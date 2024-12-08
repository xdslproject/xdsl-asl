import re
from enum import Enum, auto
from string import hexdigits
from typing import TypeAlias, cast

from xdsl.utils.exceptions import ParseError
from xdsl.utils.lexer import Lexer, Position, Span, Token
from xdsl.utils.mlir_lexer import StringLiteral


class ASLTokenKind(Enum):
    SEMICOLON = auto()
    PARENTHESE_OPEN = auto()
    PARENTHESE_CLOSE = auto()
    BRACKET_OPEN = auto()
    BRACKET_CLOSE = auto()
    SBRACKET_OPEN = auto()
    SBRACKET_CLOSE = auto()
    LT = auto()
    GT = auto()
    EQ = auto()
    COMMA = auto()

    EOF = auto()

    RETURN = auto()
    VAR = auto()
    DEF = auto()

    IDENTIFIER = auto()
    NUMBER = auto()
    OPERATOR = auto()
    STRING_LIT = auto()

    def get_string_literal_value(self, span: Span) -> str:
        """
        Translate the token text into a string literal value.
        This will remove the quotes around the string literal, and unescape
        the string.
        This function will raise an exception if the token is not a string literal.
        """
        if self != ASLTokenKind.STRING_LIT:
            raise ValueError("Token is not a string literal!")
        return StringLiteral.from_span(span).string_contents


SINGLE_CHAR_TOKENS = {
    ";": ASLTokenKind.SEMICOLON,
    "(": ASLTokenKind.PARENTHESE_OPEN,
    ")": ASLTokenKind.PARENTHESE_CLOSE,
    "{": ASLTokenKind.BRACKET_OPEN,
    "}": ASLTokenKind.BRACKET_CLOSE,
    "[": ASLTokenKind.SBRACKET_OPEN,
    "]": ASLTokenKind.SBRACKET_CLOSE,
    "<": ASLTokenKind.LT,
    ">": ASLTokenKind.GT,
    "=": ASLTokenKind.EQ,
    ",": ASLTokenKind.COMMA,
    "+": ASLTokenKind.OPERATOR,
    "-": ASLTokenKind.OPERATOR,
    "*": ASLTokenKind.OPERATOR,
    "/": ASLTokenKind.OPERATOR,
}

ASLToken: TypeAlias = Token[ASLTokenKind]


class ASLLexer(Lexer[ASLTokenKind]):
    def _is_in_bounds(self, size: Position = 1) -> bool:
        """
        Check if the current position is within the bounds of the input.
        """
        return self.pos + size - 1 < self.input.len

    def _get_chars(self, size: int = 1) -> str | None:
        """
        Get the character at the current location, or multiple characters ahead.
        Return None if the position is out of bounds.
        """
        res = self.input.slice(self.pos, self.pos + size)
        self.pos += size
        return res

    def _peek_chars(self, size: int = 1) -> str | None:
        """
        Peek at the character at the current location, or multiple characters ahead.
        Return None if the position is out of bounds.
        """
        return self.input.slice(self.pos, self.pos + size)

    def _consume_chars(self, size: int = 1) -> None:
        """
        Advance the lexer position in the input by the given amount.
        """
        self.pos += size

    def _consume_regex(self, regex: re.Pattern[str]) -> re.Match[str] | None:
        """
        Advance the lexer position to the end of the next match of the given
        regular expression.
        """
        match = regex.match(self.input.content, self.pos)
        if match is None:
            return None
        self.pos = match.end()
        return match

    _whitespace_regex = re.compile(r"((#[^\n]*(\n)?)|(\s+))*", re.ASCII)

    def _consume_whitespace(self) -> None:
        """
        Consume whitespace and comments.
        """
        self._consume_regex(self._whitespace_regex)

    def lex(self) -> ASLToken:
        # First, skip whitespaces
        self._consume_whitespace()

        start_pos = self.pos
        current_char = self._get_chars()

        # Handle end of file
        if current_char is None:
            return self._form_token(ASLTokenKind.EOF, start_pos)

        # bare identifier
        if current_char.isalpha() or current_char == "_":
            return self._lex_bare_identifier(start_pos)

        if current_char == '"':
            return self._lex_string_literal(start_pos)

        # single-char punctuation that are not part of a multi-char token
        single_char_token_kind = SINGLE_CHAR_TOKENS.get(current_char)
        if single_char_token_kind is not None:
            return self._form_token(single_char_token_kind, start_pos)

        if current_char.isnumeric():
            return self._lex_number(start_pos)

        raise ParseError(
            Span(start_pos, start_pos + 1, self.input),
            f"Unexpected character: {current_char}",
        )

    IDENTIFIER_SUFFIX = r"[a-zA-Z0-9_$.]*"
    bare_identifier_suffix_regex = re.compile(IDENTIFIER_SUFFIX)

    def _lex_bare_identifier(self, start_pos: Position) -> ASLToken:
        """
        Lex a bare identifier with the following grammar:
        `bare-id ::= (letter|[_]) (letter|digit|[_$.])*`

        The first character is expected to have already been parsed.
        """
        self._consume_regex(self.bare_identifier_suffix_regex)

        return self._form_token(ASLTokenKind.IDENTIFIER, start_pos)

    _hexdigits_star_regex = re.compile(r"[0-9a-fA-F]*")
    _digits_star_regex = re.compile(r"[0-9]*")
    _fractional_suffix_regex = re.compile(r"\.[0-9]*([eE][+-]?[0-9]+)?")

    def _lex_number(self, start_pos: Position) -> ASLToken:
        """
        Lex a number literal, which is either a decimal or an hexadecimal.
        The first character is expected to have already been parsed.
        """
        first_digit = self.input.at(self.pos - 1)

        # Hexadecimal case, we only parse it if we see the first '0x' characters,
        # and then a first digit.
        # Otherwise, a string like '0xi32' would not be parsed correctly.
        if (
            first_digit == "0"
            and self._peek_chars() == "x"
            and self._is_in_bounds(2)
            and cast(str, self.input.at(self.pos + 1)) in hexdigits
        ):
            self._consume_chars(2)
            self._consume_regex(self._hexdigits_star_regex)
            return self._form_token(ASLTokenKind.NUMBER, start_pos)

        # Decimal case
        self._consume_regex(self._digits_star_regex)

        # Check if we are lexing a floating point
        match = self._consume_regex(self._fractional_suffix_regex)
        if match is not None:
            return self._form_token(ASLTokenKind.NUMBER, start_pos)
        return self._form_token(ASLTokenKind.NUMBER, start_pos)

    _unescaped_characters_regex = re.compile(r'[^"\\\n\v\f]*')

    def _lex_string_literal(self, start_pos: Position) -> ASLToken:
        """
        Lex a string literal.
        The first character `"` is expected to have already been parsed.
        """

        while self._is_in_bounds():
            self._consume_regex(self._unescaped_characters_regex)
            current_char = self._get_chars()

            # end of string literal
            if current_char == '"':
                return self._form_token(ASLTokenKind.STRING_LIT, start_pos)

            # newline character in string literal (not allowed)
            if current_char in ["\n", "\v", "\f"]:
                raise ParseError(
                    Span(start_pos, self.pos, self.input),
                    "Newline character not allowed in string literal.",
                )

            # escape character
            # TODO: handle unicode escape
            if current_char == "\\":
                escaped_char = self._get_chars()
                if escaped_char not in ['"', "\\", "n", "t"]:
                    raise ParseError(
                        Span(start_pos, self.pos, self.input),
                        "Unknown escape in string literal.",
                    )

        raise ParseError(
            Span(start_pos, self.pos, self.input),
            "End of file reached before closing string literal.",
        )
