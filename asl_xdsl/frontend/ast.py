from __future__ import annotations

import re
from typing import NamedTuple

from xdsl.parser import BaseParser
from xdsl.parser.base_parser import ParserState
from xdsl.utils.lexer import Input, Lexer

IDENTIFIER = re.compile("[A-z_][A-z_\\d]*")


class Ty(NamedTuple):
    ty: T_Exception

    @staticmethod
    def parse_ast(parser: BaseParser) -> Ty:
        parser.parse_characters("annot")
        parser.parse_punctuation("(")
        id = parser.expect(parser.parse_optional_identifier, "Ty")
        if id != T_Exception.__name__:
            raise NotImplementedError(f"Unimplemented type {id}")
        ty = T_Exception.parse_ast_tail(parser)
        parser.parse_punctuation(")")
        return Ty(ty)


class Field(NamedTuple):
    id: str
    ty: Ty

    @staticmethod
    def parse_ast(parser: BaseParser) -> Field:
        raise NotImplementedError


class T_Exception(NamedTuple):
    fields: tuple[Field, ...]

    @staticmethod
    def parse_ast_tail(parser: BaseParser) -> T_Exception:
        # parser.parse_characters(" ")
        fields = parser.parse_comma_separated_list(
            BaseParser.Delimiter.SQUARE, lambda: Field.parse_ast(parser)
        )
        return T_Exception(tuple(fields))

    @staticmethod
    def parse_ast(parser: BaseParser) -> T_Exception:
        parser.parse_characters(T_Exception.__name__)
        return T_Exception.parse_ast_tail(parser)


class D_TypeDecl(NamedTuple):
    id: str
    ty: Ty
    fields: tuple[str, tuple[Field, ...]] | None

    @staticmethod
    def parse_optional_field(
        parser: BaseParser,
    ) -> tuple[str, tuple[Field, ...]] | None:
        if parser.parse_characters("None"):
            return None
        else:
            raise NotImplementedError()

    @staticmethod
    def parse_ast_tail(parser: BaseParser) -> D_TypeDecl:
        parser.parse_punctuation("(")
        id = parser.parse_str_literal()
        parser.parse_punctuation(",")
        ty = Ty.parse_ast(parser)
        parser.parse_punctuation(",")
        fields = D_TypeDecl.parse_optional_field(parser)
        parser.parse_punctuation(")")
        return D_TypeDecl(id, ty, fields)

    @staticmethod
    def parse_ast(parser: BaseParser) -> D_TypeDecl:
        parser.parse_characters(D_TypeDecl.__name__)
        return D_TypeDecl.parse_ast_tail(parser)


class Decl(NamedTuple):
    decl: D_TypeDecl

    @staticmethod
    def parse_ast(parser: BaseParser) -> Decl:
        id = parser.expect(parser.parse_optional_identifier, "Decl")
        if id != D_TypeDecl.__name__:
            raise NotImplementedError(f"Unimplemented declaration {id}")
        decl = D_TypeDecl.parse_ast_tail(parser)
        return Decl(decl)


class AST(NamedTuple):
    decls: tuple[Decl, ...]

    @staticmethod
    def parse_ast(parser: BaseParser) -> AST:
        return AST(
            tuple(
                parser.parse_comma_separated_list(
                    BaseParser.Delimiter.SQUARE, lambda: Decl.parse_ast(parser)
                )
            )
        )


def base_parser(input: str) -> BaseParser:
    return BaseParser(ParserState(Lexer(Input(input, "<unknown>"))))


def parse_serialized_ast(ast: str) -> AST:
    parser = base_parser(ast)
    return AST.parse_ast(parser)
