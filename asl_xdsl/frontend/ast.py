from __future__ import annotations

import re
from typing import NamedTuple

from xdsl.parser import Input

from asl_xdsl.frontend.parser import Parser

IDENTIFIER = re.compile("[A-z_][A-z_\\d]*")


def parse_optional_identifier(parser: Parser) -> str | None:
    return parser.parse_optional_pattern(IDENTIFIER)


def parse_identifier(parser: Parser) -> str:
    return parser.expect(parse_optional_identifier, "identifier")


class Ty(NamedTuple):
    ty: T_Exception

    @staticmethod
    def parse_ast(parser: Parser) -> Ty:
        parser.parse_characters("annot (")
        id = parser.expect(lambda parser: parser.peek_optional(IDENTIFIER), "Ty")[0]
        if id != T_Exception.__name__:
            raise NotImplementedError(f"Unimplemented type {id}")
        ty = T_Exception.parse_ast(parser)
        parser.parse_characters(")")
        return Ty(ty)


class Field(NamedTuple):
    id: str
    ty: Ty

    @staticmethod
    def parse_ast(parser: Parser) -> Field:
        raise NotImplementedError


class T_Exception(NamedTuple):
    fields: tuple[Field, ...]

    @staticmethod
    def parse_ast_tail(parser: Parser) -> T_Exception:
        parser.parse_characters(" [")
        fields = parser.parse_list(
            Field.parse_ast,
            lambda parser: parser.parse_characters(", "),
            lambda parser: parser.parse_optional_characters("]"),
        )
        return T_Exception(tuple(fields))

    @staticmethod
    def parse_ast(parser: Parser) -> T_Exception:
        parser.parse_characters(T_Exception.__name__)
        return T_Exception.parse_ast_tail(parser)


class D_TypeDecl(NamedTuple):
    id: str
    ty: Ty
    fields: tuple[str, tuple[Field, ...]] | None

    @staticmethod
    def parse_optional_field(
        parser: Parser,
    ) -> tuple[str, tuple[Field, ...]] | None:
        if parser.parse_characters("None"):
            return None
        else:
            raise NotImplementedError()

    @staticmethod
    def parse_ast_tail(parser: Parser) -> D_TypeDecl:
        parser.parse_characters(" (")
        id = parser.parse_str_literal()
        parser.parse_characters(", ")
        ty = Ty.parse_ast(parser)
        parser.parse_characters(", ")
        fields = D_TypeDecl.parse_optional_field(parser)
        parser.parse_characters(")")
        return D_TypeDecl(id, ty, fields)

    @staticmethod
    def parse_ast(parser: Parser) -> D_TypeDecl:
        parser.parse_characters(D_TypeDecl.__name__)
        return D_TypeDecl.parse_ast_tail(parser)


class Decl(NamedTuple):
    decl: D_TypeDecl

    @staticmethod
    def parse_ast(parser: Parser) -> Decl:
        id = parser.expect(parse_optional_identifier, "Decl")
        if id != D_TypeDecl.__name__:
            raise NotImplementedError(f"Unimplemented declaration {id}")
        decl = D_TypeDecl.parse_ast_tail(parser)
        return Decl(decl)


class AST(NamedTuple):
    decls: tuple[Decl, ...]

    @staticmethod
    def parse_ast(parser: Parser) -> AST:
        parser.parse_characters("[")
        return AST(
            tuple(
                parser.parse_list(
                    lambda parser: Decl.parse_ast(parser),
                    lambda parser: parser.parse_optional_characters(", "),
                    lambda parser: parser.parse_optional_characters("]"),
                )
            )
        )


def base_parser(input: str) -> Parser:
    return Parser(Input(input, "<unknown>"))


def parse_serialized_ast(ast: str) -> AST:
    parser = base_parser(ast)
    return AST.parse_ast(parser)
