from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from xdsl.parser import GenericParser, Input, ParserState

from asl_xdsl.frontend.ast import (
    AST,
    SB_ASL,
    Annotated,
    D_Func,
    D_TypeDecl,
    Decl,
    E_Literal,
    Expr,
    Field,
    L_Int,
    Literal,
    SPass,
    SReturn,
    Stmt,
    StmtDesc,
    SubprogramBody,
    SubprogramType,
    T_Exception,
    T_Int,
    T_Record,
    Ty,
    TypeDesc,
    UnConstrained,
)
from asl_xdsl.frontend.lexer import ASLLexer, ASLTokenKind

_T = TypeVar("_T")


class BaseASLParser(GenericParser[ASLTokenKind]):
    def __init__(self, input: Input | str):
        if isinstance(input, str):
            input = Input(input, "<unknown>")
        super().__init__(ParserState(ASLLexer(input)))

    def peek(self) -> str | None:
        if self._current_token.kind == ASLTokenKind.EOF:
            return None
        return self._current_token.text

    def parse_optional_identifier(self) -> str | None:
        if (tok := self._parse_optional_token(ASLTokenKind.IDENTIFIER)) is not None:
            return tok.text

    def parse_identifier(self) -> str:
        return self.expect(self.parse_optional_identifier, "Expected identifier")

    def parse_optional_str_literal(self) -> str | None:
        """
        Parse a string literal with the format `"..."`, if present.

        Returns the string contents without the quotes and with escape sequences
        resolved.
        """

        if (token := self._parse_optional_token(ASLTokenKind.STRING_LIT)) is None:
            return None
        try:
            return token.kind.get_string_literal_value(token.span)
        except UnicodeDecodeError:
            return None

    def parse_str_literal(self, context_msg: str = "") -> str:
        """
        Parse a string literal with the format `"..."`.

        Returns the string contents without the quotes and with escape sequences
        resolved.
        """
        return self.expect(
            self.parse_optional_str_literal,
            "string literal expected" + context_msg,
        )


class ASTParser(BaseASLParser):
    def parse_field(self) -> Field:
        raise NotImplementedError()

    def parse_exception(self) -> T_Exception:
        self.parse_characters("T_Exception")
        fields = self.parse_comma_separated_list(
            self.Delimiter.SQUARE, self.parse_field
        )
        return T_Exception(tuple(fields))

    def parse_record(self) -> T_Record:
        self.parse_characters("T_Record")
        fields = self.parse_comma_separated_list(
            self.Delimiter.SQUARE, self.parse_field
        )
        return T_Record(tuple(fields))

    TYPE_PARSER = {
        "T_Exception": parse_exception,
        "T_Record": parse_record,
    }

    def parse_type_desc(self) -> TypeDesc:
        ty_key = self.peek()
        assert ty_key is not None
        p = self.TYPE_PARSER.get(ty_key)
        if p is None:
            raise NotImplementedError(f"Unimplemented type {ty_key}")
        return p(self)

    def parse_annotated(self, inner: Callable[[ASTParser], _T]) -> Annotated[_T]:
        self.parse_characters("annot")
        self.parse_characters("(")
        res = inner(self)
        self.parse_characters(")")
        return Annotated(res)

    def parse_ty(self) -> Ty:
        return self.parse_annotated(ASTParser.parse_type_desc)

    def parse_optional_type_decl_field(self) -> tuple[str, tuple[Field, ...]] | None:
        if self.parse_characters("None"):
            return None
        else:
            raise NotImplementedError()

    def parse_type_decl(self) -> D_TypeDecl:
        self.parse_characters("D_TypeDecl")
        self.parse_characters("(")
        id = self.parse_str_literal()
        self.parse_characters(",")
        ty = self.parse_ty()
        self.parse_characters(",")
        fields = self.parse_optional_type_decl_field()
        self.parse_characters(")")
        return D_TypeDecl(id, ty, fields)

    def parse_decl(self) -> Decl:
        id = self.peek()
        if id != D_TypeDecl.__name__:
            raise NotImplementedError(f"Unimplemented declaration {id}")
        decl = self.parse_type_decl()
        return decl

    def parse_ast(self) -> AST:
        decls = self.parse_comma_separated_list(self.Delimiter.SQUARE, self.parse_decl)
        return AST(tuple(decls))


class ASLParser(BaseASLParser):
    def parse_integer(self) -> T_Int:
        self.parse_characters("integer")
        # TODO: parse constraints
        return T_Int(UnConstrained())

    def parse_exception(self) -> T_Exception:
        self.parse_characters("exception")
        # TODO: parse fields
        return T_Exception(())

    def parse_record(self) -> T_Record:
        self.parse_characters("record")
        # TODO: parse fields
        return T_Record(())

    TYPE_PARSER = {
        "exception": parse_exception,
        "record": parse_record,
        "integer": parse_integer,
    }

    def parse_type_desc(self) -> TypeDesc:
        ty_key = self.peek()
        assert ty_key is not None
        p = self.TYPE_PARSER.get(ty_key)
        if p is None:
            raise NotImplementedError(f"Unimplemented type {ty_key}")
        return p(self)

    def parse_ty(self) -> Ty:
        return Annotated(self.parse_type_desc())

    def parse_type_decl(self) -> D_TypeDecl:
        self.parse_characters("type")
        id = self.parse_identifier()
        self.parse_characters("of")
        ty = self.parse_ty()
        self._parse_optional_token(ASLTokenKind.SEMICOLON)
        return D_TypeDecl(id, ty, None)

    def parse_optional_literal(self) -> Literal | None:
        # TODO: proper integer literal
        if (tok := self._parse_optional_token(ASLTokenKind.NUMBER)) is not None:
            return L_Int(int(tok.text))

    def parse_optional_expr(self) -> Expr | None:
        if (lit := self.parse_optional_literal()) is not None:
            return Annotated(E_Literal(lit))

    def parse_return(self) -> SReturn:
        self.parse_characters("return")
        expr = self.parse_optional_expr()
        self.parse_characters(";")
        return SReturn(expr)

    STMT_PARSER = {
        "return": parse_return,
    }

    def parse_stmt_desc(self) -> StmtDesc:
        next_token = self.peek()
        if next_token == "end":
            return SPass()
        if next_token is None:
            raise NotImplementedError(f"Stmt beginning with {next_token}")
        p = self.STMT_PARSER.get(next_token)
        if p is None:
            raise NotImplementedError(f"Stmt beginning with {next_token}")
        return p(self)

    def parse_stmt(self) -> Stmt:
        return Annotated(self.parse_stmt_desc())

    def parse_subprogram_body(self) -> SubprogramBody:
        return SB_ASL(self.parse_stmt())

    def parse_func_decl(self) -> D_Func:
        self.parse_characters("func")
        name = self.parse_identifier()
        self.parse_characters("(")
        self.parse_characters(")")
        if self._parse_optional_token(ASLTokenKind.ARROW):
            return_type = self.parse_ty()
        else:
            return_type = None
        self.parse_characters("begin")
        body = self.parse_subprogram_body()
        self.parse_characters("end")
        self.parse_characters(";")
        return D_Func(
            name,
            None,
            body,
            return_type,
            None,
            SubprogramType.ST_Procedure,
        )

    DECL_PARSER = {
        "type": parse_type_decl,
        "func": parse_func_decl,
    }

    def parse_decl(self) -> Decl:
        decl_key = self.peek()
        assert decl_key is not None
        p = self.DECL_PARSER.get(decl_key)
        if p is None:
            raise NotImplementedError(f"Unimplemented type {decl_key}")
        decl = p(self)
        return decl

    def parse_ast(self) -> AST:
        decls: list[Decl] = []

        while self.peek() is not None:
            decls.append(self.parse_decl())

        return AST(tuple(decls))
