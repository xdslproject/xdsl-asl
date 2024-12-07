from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects import builtin
from xdsl.ir import (
    Attribute,
    Data,
    Dialect,
    ParametrizedAttribute,
    TypeAttribute,
    VerifyException,
)
from xdsl.irdl import ParameterDef, irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer


@irdl_attr_definition
class BoolType(ParametrizedAttribute, TypeAttribute):
    """A boolean type."""

    name = "asl.bool"


@irdl_attr_definition
class BoolAttr(Data[bool]):
    """A boolean attribute."""

    name = "asl.bool_attr"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> bool:
        """Parse the attribute parameter."""
        parser.parse_characters("<")
        value = parser.parse_boolean()
        parser.parse_characters(">")
        return value

    def print_parameter(self, printer: Printer) -> None:
        """Print the attribute parameter."""
        printer.print("true" if self.data else "false")


@irdl_attr_definition
class IntegerType(ParametrizedAttribute, TypeAttribute):
    """An arbitrary-precision integer type."""

    name = "asl.int"


@irdl_attr_definition
class IntegerAttr(ParametrizedAttribute):
    """An arbitrary-precision integer attribute."""

    name = "asl.int_attr"

    value: ParameterDef[builtin.IntAttr]

    def __init__(self, value: int | builtin.IntAttr):
        if isinstance(value, int):
            value = builtin.IntAttr(value)
        super().__init__([value])

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """Parse the attribute parameters."""
        parser.parse_characters("<")
        value = builtin.IntAttr(parser.parse_integer())
        parser.parse_characters(">")
        return [value]

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        printer.print("<")
        printer.print(self.value.data)
        printer.print(">")


@irdl_attr_definition
class BitVectorType(ParametrizedAttribute, TypeAttribute):
    """A bit vector type."""

    name = "asl.bits"

    width: ParameterDef[builtin.IntAttr]

    def __init__(self, width: int | builtin.IntAttr):
        if isinstance(width, int):
            width = builtin.IntAttr(width)
        super().__init__([width])

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """Parse the attribute parameters."""
        parser.parse_characters("<")
        width = builtin.IntAttr(parser.parse_integer())
        parser.parse_characters(">")
        return [width]

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        printer.print("<")
        printer.print(self.width.data)
        printer.print(">")


@irdl_attr_definition
class BitVectorAttr(ParametrizedAttribute):
    """A bit vector attribute."""

    name = "asl.bits_attr"

    value: ParameterDef[builtin.IntAttr]
    width: ParameterDef[builtin.IntAttr]

    def maximum_value(self) -> int:
        """Return the maximum value that can be represented."""
        return (1 << self.width.data) - 1

    @staticmethod
    def normalize_value(value: int, width: int) -> int:
        """Normalize the value to the range [0, 2^width)."""
        max_value = 1 << width
        return ((value % max_value) + max_value) % max_value

    def __init__(self, value: int | builtin.IntAttr, width: int | builtin.IntAttr):
        if isinstance(value, int):
            value = builtin.IntAttr(value)
        if isinstance(width, int):
            width = builtin.IntAttr(width)

        value_int = value.data
        value = builtin.IntAttr(self.normalize_value(value_int, width.data))
        super().__init__([value, width])

    def _verify(self) -> None:
        if self.value.data < 0 or self.value.data >= self.maximum_value():
            raise VerifyException(
                f"Value {self.value.data} is out of range for width {self.width.data}"
            )

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """Parse the attribute parameters."""
        parser.parse_characters("<")
        value = builtin.IntAttr(parser.parse_integer())
        parser.parse_characters(":")
        width = builtin.IntAttr(parser.parse_integer())
        parser.parse_characters(">")
        return [value, width]

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        printer.print("<")
        printer.print(self.value.data)
        printer.print(" : ")
        printer.print(self.width.data)
        printer.print(">")


ASLDialect = Dialect(
    "asl",
    [],
    [BoolType, BoolAttr, IntegerType, IntegerAttr, BitVectorType, BitVectorAttr],
)
