from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, NamedTuple, TypeAlias, TypeVar

from asl_xdsl.frontend.printer import Printer

T = TypeVar("T", covariant=True)


@dataclass
class Annotated(Generic[T]):
    val: T

    def print_asl(self, printer: Printer):
        print_asl = getattr(self.val, "print_asl")
        print_asl(printer)


class Field(NamedTuple):
    id: str
    ty: Ty

    def print_asl(self, printer: Printer) -> None:
        raise NotImplementedError()


# region: Types


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


TypeDesc: TypeAlias = T_Exception | T_Record


Ty: TypeAlias = Annotated[TypeDesc]

# endregion


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


class TypingRule(Enum):
    SPass = auto()


class SPass(NamedTuple): ...


StmtDesc: TypeAlias = SPass

Stmt: TypeAlias = Annotated[StmtDesc]


class SB_ASL(NamedTuple):
    stmt: Stmt


class SB_Primitive(NamedTuple): ...


SubprogramBody: TypeAlias = SB_ASL | SB_Primitive


class SubprogramType(Enum):
    ST_Procedure = auto()
    """
    A procedure is a subprogram without return type, called from a statement.
    """
    ST_Function = auto()
    """
    A function is a subprogram with a return type, called from an expression.
    """
    ST_Getter = auto()
    """
    A getter is a special function called with a syntax similar to slices.
    """
    ST_EmptyGetter = auto()
    """
    An empty getter is a special function called with a syntax similar to a variable.
    This is relevant only for V0.
    """
    ST_Setter = auto()
    """
    A setter is a special procedure called with a syntax similar to slice assignment.
    """
    ST_EmptySetter = auto()
    """
    An empty setter is a special procedure called with a syntax similar to an assignment
    to a variable. This is relevant only for V0.
    """


class D_Func(NamedTuple):
    name: str
    args: None
    body: SubprogramBody
    return_type: None
    parameters: None
    subprogram_type: SubprogramType

    def print_subprogram_body(self, printer: Printer):
        if isinstance(self.body, SB_Primitive):
            raise NotImplementedError()
        sb = self.body.stmt.val
        match sb:
            case SPass():
                return

    def print_asl(self, printer: Printer):
        if self.args is not None:
            raise NotImplementedError()
        if self.return_type is not None:
            raise NotImplementedError()
        if self.parameters is not None:
            raise NotImplementedError()
        if self.subprogram_type != SubprogramType.ST_Procedure:
            raise NotImplementedError()
        printer.print_string("func ")
        printer.print_string(self.name)
        printer.print_string("()\n")
        printer.print_string("begin\n")
        self.print_subprogram_body(printer)
        printer.print_string("end;\n")


class Decl(NamedTuple):
    decl: D_TypeDecl | D_Func

    def print_asl(self, printer: Printer) -> None:
        self.decl.print_asl(printer)


class AST(NamedTuple):
    decls: tuple[Decl, ...]

    def print_asl(self, printer: Printer) -> None:
        for decl in self.decls:
            decl.print_asl(printer)
