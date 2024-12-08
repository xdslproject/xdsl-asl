import re
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from xdsl.parser import Input
from xdsl.utils.lexer import Position, Span, StringLiteral

T = TypeVar("T")


class ParseError(Exception):
    position: Position
    message: str

    def __init__(self, position: Position, message: str) -> None:
        self.position = position
        self.message = message
        super().__init__()

    def __str__(self):
        return f"ParseError at {self.position}: {self.message}"


class Parser:
    input: Input
    pos: int

    def __init__(self, input: Input | str, pos: int = 0):
        if isinstance(input, str):
            input = Input(input, "<unknown>")
        self.input = input
        self.pos = pos

    @property
    def remaining(self) -> str:
        return self.input.content[self.pos :]

    # region Base parsing functions

    def parse_optional_characters(self, chars: str):
        if self.input.content.startswith(chars, self.pos):
            self.pos += len(chars)
            return chars

    def parse_characters(self, chars: str) -> str:
        return self.expect(
            lambda parser: parser.parse_optional_characters(chars), chars
        )

    def peek_optional(self, pattern: re.Pattern[str]):
        if (match := pattern.match(self.input.content, self.pos)) is not None:
            end_pos = match.regs[0][1]
            return self.input.content[match.pos : end_pos], end_pos

    def parse_optional_pattern(self, pattern: re.Pattern[str]):
        res = self.peek_optional(pattern)
        if res is not None:
            self.pos = res[1]
            return res[0]

    def parse_pattern(self, pattern: re.Pattern[str], message: str = ""):
        return self.expect(
            lambda parser: self.parse_optional_pattern(pattern),
            message if message else str(pattern),
        )

    # endregion
    # region: Helpers

    _unescaped_characters_regex = re.compile(r'[^"\\\n\v\f]*')

    def _is_in_bounds(self, size: Position = 1) -> bool:
        """
        Check if the current position is within the bounds of the input.
        """
        return self.pos + size - 1 < self.input.len

    def _lex_string_literal(self, start_pos: Position) -> str:
        """
        Lex a string literal.
        The first character `"` is expected to have already been parsed.
        """

        while self._is_in_bounds():
            self.parse_pattern(self._unescaped_characters_regex)
            current_char = self.input.at(self.pos)
            self.pos += 1

            # end of string literal
            if current_char == '"':
                return StringLiteral.from_span(
                    Span(start_pos, self.pos, self.input)
                ).bytes_contents.decode()

            # newline character in string literal (not allowed)
            if current_char in ["\n", "\v", "\f"]:
                raise ParseError(
                    start_pos,
                    "Newline character not allowed in string literal.",
                )

            # escape character
            # TODO: handle unicode escape
            if current_char == "\\":
                escaped_char = self.input.at(self.pos)
                self.pos += 1
                if escaped_char not in ['"', "\\", "n", "t"]:
                    raise ParseError(
                        start_pos,
                        "Unknown escape in string literal.",
                    )

        raise ParseError(
            start_pos,
            "End of file reached before closing string literal.",
        )

    def parse_optional_str_literal(self) -> str | None:
        """
        Parse a string literal with the format `"..."`, if present.

        Returns the string contents without the quotes and with escape sequences
        resolved.
        """
        pos = self.pos
        if self.parse_optional_characters('"'):
            return self._lex_string_literal(pos)

    def parse_str_literal(self, context_msg: str = "") -> str:
        """
        Parse a string literal with the format `"..."`.

        Returns the string contents without the quotes and with escape sequences
        resolved.
        """
        return self.expect(
            Parser.parse_optional_str_literal,
            "string literal expected" + context_msg,
        )

    IDENTIFIER_SUFFIX = r"[a-zA-Z0-9_$.]*"
    BARE_IDENDITIER_REGEX = re.compile(r"[a-zA-Z_]" + IDENTIFIER_SUFFIX)

    def parse_many(self, element: Callable[["Parser"], T | None]) -> tuple[T, ...]:
        if (first := element(self)) is None:
            return ()
        res = [first]
        while (el := element(self)) is not None:
            res.append(el)
        return tuple(res)

    def parse_many_separated(
        self,
        element: Callable[["Parser"], T | None],
        separator: Callable[["Parser"], Any | None],
    ) -> tuple[T, ...]:
        if (first := element(self)) is None:
            return ()
        res = [first]
        while separator(self) is not None:
            el = self.expect(element, "element")
            res.append(el)
        return tuple(res)

    def parse_optional_list(
        self,
        el: Callable[["Parser"], T | None],
        separator: Callable[["Parser"], Any | None],
        end: Callable[["Parser"], Any | None],
    ) -> list[T] | None:
        if end(self) is not None:
            return []
        if (first := el(self)) is None:
            return None
        res = [first]
        while end(self) is None:
            self.expect(separator, "separator")
            element = self.expect(el, "element")
            res.append(element)
        return res

    IDENTIFIER = re.compile("[A-z_][A-z_\\d]*")

    def parse_optional_identifier(self) -> str | None:
        return self.parse_optional_pattern(Parser.IDENTIFIER)

    def parse_identifier(self) -> str:
        return self.expect(Parser.parse_optional_identifier, "identifier")

    def parse_list(
        self,
        el: Callable[["Parser"], T],
        separator: Callable[["Parser"], Any | None],
        end: Callable[["Parser"], Any | None],
    ) -> list[T]:
        if end(self) is not None:
            return []
        first = el(self)
        res = [first]
        while end(self) is None:
            self.expect(separator, "separator")
            element = self.expect(el, "element")
            res.append(element)
        return res

    def expect(self, parse: Callable[["Parser"], T | None], message: str) -> T:
        if (parsed := parse(self)) is None:
            raise ParseError(self.pos, message)
        return parsed

    def parse_one_of(
        self, parsers: Sequence[Callable[["Parser"], T | None]]
    ) -> T | None:
        for parser in parsers:
            if (parsed := parser(self)) is not None:
                return parsed

    # endregion
