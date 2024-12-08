from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, NamedTuple, TypeAlias, TypeVar

from asl_xdsl.frontend.printer import Printer

T = TypeVar("T", covariant=True)


# region Helpers
@dataclass
class Annotated(Generic[T]):
    val: T

    def print_asl(self, printer: Printer):
        print_asl = getattr(self.val, "print_asl")
        print_asl(printer)


# endregion

# region Literal


class L_Int(NamedTuple):
    val: int

    def print_asl(self, printer: Printer):
        printer.print_string(f"{self.val}")


Literal: TypeAlias = L_Int

# endregion


# region Expressions


class E_Literal(NamedTuple):
    literal: Literal

    def print_asl(self, printer: Printer):
        self.literal.print_asl(printer)


ExprDesc: TypeAlias = E_Literal

Expr: TypeAlias = Annotated[ExprDesc]

# endregion


class Field(NamedTuple):
    id: str
    ty: Ty

    def print_asl(self, printer: Printer) -> None:
        raise NotImplementedError()


# region: Constraints


class UnConstrained(NamedTuple):
    """
    The normal, unconstrained, integer type.
    """

    def print_asl(self, printer: Printer) -> None:
        printer.print_string("integer")


class WellConstrained(NamedTuple):
    """
    An integer type constrained from ASL syntax: it is the union of each constraint in
    the list.
    """

    # of int_constraint list

    def print_asl(self, printer: Printer) -> None:
        raise NotImplementedError()


class PendingConstrained(NamedTuple):
    """
    An integer type whose constraint will be inferred during type-checking.
    """

    def print_asl(self, printer: Printer) -> None:
        raise NotImplementedError()


class Parameterized(NamedTuple):
    """
    A parameterized integer, the default type for parameters of function at compile
    time, with a unique identifier and the variable bearing its name.
    """

    # of uid * identifier

    def print_asl(self, printer: Printer) -> None:
        raise NotImplementedError()


ConstraintKind: TypeAlias = (
    UnConstrained | WellConstrained | PendingConstrained | Parameterized
)
"""
The constraint_kind constrains an integer type to a certain subset.
"""

# endregion


# region: Types


class T_Int(NamedTuple):
    kind: ConstraintKind

    def print_asl(self, printer: Printer) -> None:
        self.kind.print_asl(printer)


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


TypeDesc: TypeAlias = T_Int | T_Exception | T_Record


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


class SPass(NamedTuple):
    def print_asl(self, printer: Printer): ...


class SReturn(NamedTuple):
    expr: Expr | None

    def print_asl(self, printer: Printer):
        # TODO: proper indentation
        printer.print_string("    ")
        printer.print_string("return")
        if self.expr is not None:
            printer.print_string(" ")
            self.expr.print_asl(printer)
        printer.print_string(";\n")


StmtDesc: TypeAlias = SPass | SReturn

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
    return_type: Ty | None
    parameters: None
    subprogram_type: SubprogramType

    def print_subprogram_body(self, printer: Printer):
        if isinstance(self.body, SB_Primitive):
            raise NotImplementedError()
        sb = self.body.stmt.val
        # TODO: indentation
        sb.print_asl(printer)

    def print_asl(self, printer: Printer):
        if self.args is not None:
            raise NotImplementedError()
        if self.parameters is not None:
            raise NotImplementedError()
        if self.subprogram_type != SubprogramType.ST_Procedure:
            raise NotImplementedError()
        printer.print_string("func ")
        printer.print_string(self.name)
        printer.print_string("()")
        if self.return_type is not None:
            printer.print_string(" => ")
            self.return_type.print_asl(printer)
        printer.print_string("\n")
        printer.print_string("begin\n")
        self.print_subprogram_body(printer)
        printer.print_string("end;\n")


Decl: TypeAlias = D_TypeDecl | D_Func


class AST(NamedTuple):
    decls: tuple[Decl, ...]

    def print_asl(self, printer: Printer) -> None:
        for decl in self.decls:
            decl.print_asl(printer)
