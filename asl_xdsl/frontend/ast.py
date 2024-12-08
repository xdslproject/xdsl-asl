from __future__ import annotations

from typing import NamedTuple

from asl_xdsl.frontend.printer import Printer


class Ty(NamedTuple):
    ty: T_Exception | T_Record

    def print_asl(self, printer: Printer):
        self.ty.print_asl(printer)


class Field(NamedTuple):
    id: str
    ty: Ty

    def print_asl(self, printer: Printer) -> None:
        raise NotImplementedError()


class T_Exception(NamedTuple):
    fields: tuple[Field, ...]

    def print_asl(self, printer: Printer) -> None:
        if self.fields:
            raise NotImplementedError()
        printer.print_string("exception")


class T_Record(NamedTuple):
    fields: tuple[Field, ...]

    def print_asl(self, printer: Printer) -> None:
        if self.fields:
            raise NotImplementedError()
        printer.print_string("record")


class D_TypeDecl(NamedTuple):
    id: str
    ty: Ty
    fields: tuple[str, tuple[Field, ...]] | None

    def print_asl(self, printer: Printer) -> None:
        printer.print_string("type ")
        printer.print_string(self.id)
        printer.print_string(" of ")
        self.ty.print_asl(printer)
        printer.print_string(";\n")


class Decl(NamedTuple):
    decl: D_TypeDecl

    def print_asl(self, printer: Printer) -> None:
        self.decl.print_asl(printer)


class AST(NamedTuple):
    decls: tuple[Decl, ...]

    def print_asl(self, printer: Printer) -> None:
        for decl in self.decls:
            decl.print_asl(printer)
