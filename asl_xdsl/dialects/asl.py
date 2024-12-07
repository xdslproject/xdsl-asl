from __future__ import annotations

from collections.abc import Mapping, Sequence

from xdsl.dialects import builtin
from xdsl.ir import (
    Attribute,
    Data,
    Dialect,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
    VerifyException,
)
from xdsl.irdl import (
    IRDLOperation,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
)
from xdsl.parser import AttrParser, Parser
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
class BitVectorType(ParametrizedAttribute, TypeAttribute):
    """A bit vector type."""

    name = "asl.bits"

    width: ParameterDef[builtin.IntAttr]

    def __init__(self, width: int):
        super().__init__([builtin.IntAttr(width)])

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
    type: ParameterDef[BitVectorType]

    def maximum_value(self) -> int:
        """Return the maximum value that can be represented."""
        return (1 << self.type.width.data) - 1

    @staticmethod
    def normalize_value(value: int, width: int) -> int:
        """Normalize the value to the range [0, 2^width)."""
        max_value = 1 << width
        return ((value % max_value) + max_value) % max_value

    def __init__(self, value: int, type: BitVectorType):
        value = self.normalize_value(value, type.width.data)
        super().__init__([builtin.IntAttr(value), type])

    def _verify(self) -> None:
        if self.value.data < 0 or self.value.data >= self.maximum_value():
            raise VerifyException(
                f"Value {self.value.data} is out of range "
                f"for width {self.type.width.data}"
            )

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """Parse the attribute parameters."""
        parser.parse_characters("<")
        value = builtin.IntAttr(parser.parse_integer())
        parser.parse_characters(":")
        width = parser.parse_attribute()
        parser.parse_characters(">")
        return [value, width]

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        printer.print("<")
        printer.print(self.value.data)
        printer.print(" : ")
        printer.print(self.type.width.data)
        printer.print(">")


@irdl_op_definition
class ConstantBoolOp(IRDLOperation):
    """A constant boolean operation."""

    name = "asl.constant_bool"

    value = prop_def(BoolAttr)
    res = result_def(BoolType())

    def __init__(self, value: bool, attr_dict: Mapping[str, Attribute] = {}):
        super().__init__(
            result_types=[BoolType()],
            properties={"value": BoolAttr(value)},
            attributes=attr_dict,
        )

    @classmethod
    def parse(cls, parser: Parser) -> ConstantBoolOp:
        """Parse the operation."""
        value = parser.parse_boolean()
        attr_dict = parser.parse_optional_attr_dict()
        return ConstantBoolOp(value, attr_dict)

    def print(self, printer: Printer) -> None:
        """Print the operation."""
        printer.print(" ", "true" if self.value else "false")
        if self.attributes:
            printer.print(" ")
            printer.print_attr_dict(self.attributes)


@irdl_op_definition
class ConstantIntOp(IRDLOperation):
    """A constant arbitrary-sized integer operation."""

    name = "asl.constant_int"

    value = prop_def(builtin.IntAttr)
    res = result_def(IntegerType())

    def __init__(
        self, value: int | builtin.IntAttr, attr_dict: Mapping[str, Attribute] = {}
    ):
        if isinstance(value, int):
            value = builtin.IntAttr(value)
        super().__init__(
            result_types=[IntegerType()],
            properties={"value": value},
            attributes=attr_dict,
        )

    @classmethod
    def parse(cls, parser: Parser) -> ConstantIntOp:
        """Parse the operation."""
        value = parser.parse_integer(allow_boolean=False, allow_negative=False)
        attr_dict = parser.parse_optional_attr_dict()
        return ConstantIntOp(value, attr_dict)

    def print(self, printer: Printer) -> None:
        """Print the operation."""
        printer.print(" ", self.value.data)
        if self.attributes:
            printer.print(" ")
            printer.print_attr_dict(self.attributes)


@irdl_op_definition
class ConstantBitVectorOp(IRDLOperation):
    """A constant bit vector operation."""

    name = "asl.constant_bits"

    value = prop_def(BitVectorAttr)
    res = result_def(BitVectorType)

    def __init__(
        self,
        value: BitVectorAttr,
        attr_dict: Mapping[str, Attribute] = {},
    ) -> None:
        super().__init__(
            result_types=[value.type],
            properties={"value": value},
            attributes=attr_dict,
        )

    @classmethod
    def parse(cls, parser: Parser) -> ConstantBitVectorOp:
        """Parse the operation."""
        value = parser.parse_integer()
        parser.parse_characters(":")

        type = parser.parse_attribute()
        if not isinstance(type, BitVectorType):
            parser.raise_error(f"Expected bit vector type, got {type}")

        value = BitVectorAttr(value, type)
        attr_dict = parser.parse_optional_attr_dict()
        return ConstantBitVectorOp(value, attr_dict)

    def print(self, printer: Printer) -> None:
        """Print the operation."""
        printer.print(" ", self.value.value.data, " : ", self.res.type)
        if self.attributes:
            printer.print(" ")
            printer.print_attr_dict(self.attributes)


@irdl_op_definition
class NotOp(IRDLOperation):
    """A bitwise NOT operation."""

    name = "asl.not_bool"

    arg = operand_def(BoolType())
    res = result_def(BoolType())

    assembly_format = "$arg attr-dict"

    def __init__(self, arg: SSAValue, attr_dict: Mapping[str, Attribute] = {}):
        super().__init__(
            operands=[arg],
            result_types=[BoolType()],
            attributes=attr_dict,
        )


class BinaryBoolOp(IRDLOperation):
    """A binary boolean operation."""

    lhs = operand_def(BoolType())
    rhs = operand_def(BoolType())
    res = result_def(BoolType())

    assembly_format = "$lhs `,` $rhs attr-dict"

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[BoolType()],
            attributes=attr_dict,
        )


@irdl_op_definition
class AndBoolOp(BinaryBoolOp):
    """A boolean AND operation."""

    name = "asl.and_bool"


@irdl_op_definition
class OrBoolOp(BinaryBoolOp):
    """A boolean OR operation."""

    name = "asl.or_bool"


@irdl_op_definition
class EqBoolOp(BinaryBoolOp):
    """A boolean EQ operation."""

    name = "asl.eq_bool"


@irdl_op_definition
class NeBoolOp(BinaryBoolOp):
    """A boolean NE operation."""

    name = "asl.ne_bool"


@irdl_op_definition
class ImpliesBoolOp(BinaryBoolOp):
    """A boolean IMPLIES operation."""

    name = "asl.implies_bool"


@irdl_op_definition
class EquivBoolOp(BinaryBoolOp):
    """A boolean EQUIV operation."""

    name = "asl.equiv_bool"


ASLDialect = Dialect(
    "asl",
    [
        ConstantBoolOp,
        ConstantIntOp,
        ConstantBitVectorOp,
        NotOp,
        AndBoolOp,
        OrBoolOp,
        EqBoolOp,
        NeBoolOp,
        ImpliesBoolOp,
        EquivBoolOp,
    ],
    [BoolType, BoolAttr, IntegerType, BitVectorType, BitVectorAttr],
)
