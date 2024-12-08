from __future__ import annotations

from typing import NamedTuple

from xdsl.parser import Input

from asl_xdsl.frontend.parser import Parser
from asl_xdsl.frontend.printer import Printer


class Ty(NamedTuple):
    ty: T_Exception

    @staticmethod
    def parse_ast(parser: Parser) -> Ty:
        parser.parse_characters("annot (")
        id = parser.parse_identifier()
        if id != T_Exception.__name__:
            raise NotImplementedError(f"Unimplemented type {id}")
        ty = T_Exception.parse_ast_tail(parser)
        parser.parse_characters(")")
        return Ty(ty)

    def print_asl(self, printer: Printer):
        self.ty.print_asl(printer)

    @staticmethod
    def parse_asl(parser: Parser) -> Ty:
        if parser.parse_optional_identifier() == "exception":
            return Ty(T_Exception.parse_asl_tail(parser))
        else:
            raise NotImplementedError()


class Field(NamedTuple):
    id: str
    ty: Ty

    @staticmethod
    def parse_ast(parser: Parser) -> Field:
        raise NotImplementedError()

    def print_asl(self, printer: Printer) -> None:
        raise NotImplementedError()


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

    @staticmethod
    def parse_asl_tail(parser: Parser) -> T_Exception:
        """
        Everything after `exception`
        """
        # TODO: parse fields
        return T_Exception(())

    def print_asl(self, printer: Printer) -> None:
        if self.fields:
            raise NotImplementedError()
        printer.print_string("exception")


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

    def print_asl(self, printer: Printer) -> None:
        printer.print_string("type ")
        printer.print_string(self.id)
        printer.print_string(" of ")
        self.ty.print_asl(printer)
        printer.print_string(";\n")

    @staticmethod
    def parse_asl_tail(parser: Parser) -> D_TypeDecl:
        """
        Parse everything after `type`
        """
        # TODO: be more flexible with whitespace
        parser.parse_characters(" ")
        id = parser.parse_identifier()
        parser.parse_characters(" of ")
        ty = Ty.parse_asl(parser)
        parser.parse_characters(";\n")
        return D_TypeDecl(id, ty, None)


class Decl(NamedTuple):
    decl: D_TypeDecl

    @staticmethod
    def parse_ast(parser: Parser) -> Decl:
        id = parser.parse_identifier()
        if id != D_TypeDecl.__name__:
            raise NotImplementedError(f"Unimplemented declaration {id}")
        decl = D_TypeDecl.parse_ast_tail(parser)
        return Decl(decl)

    def print_asl(self, printer: Printer) -> None:
        self.decl.print_asl(printer)

    @staticmethod
    def parse_optional_asl(parser: Parser) -> Decl | None:
        pos = parser.pos
        if parser.parse_optional_identifier() == "type":
            return Decl(D_TypeDecl.parse_asl_tail(parser))
        parser.pos = pos


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

    def print_asl(self, printer: Printer) -> None:
        for decl in self.decls:
            decl.print_asl(printer)

    @staticmethod
    def parse_asl(parser: Parser) -> AST:
        return AST(parser.parse_many(Decl.parse_optional_asl))


def base_parser(input: str) -> Parser:
    return Parser(Input(input, "<unknown>"))


def parse_serialized_ast(ast: str) -> AST:
    parser = base_parser(ast)
    return AST.parse_ast(parser)
