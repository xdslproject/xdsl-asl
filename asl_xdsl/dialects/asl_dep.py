"""A dependently-typed high-level ASL dialect."""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects import builtin
from xdsl.ir import Dialect, ParametrizedAttribute, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer


@irdl_attr_definition
class BitsType(ParametrizedAttribute):
    """
    A bitvector type.
    While bitvector types have a bitwith paramater, these are attached in the
    operations that are using them, as these can be encoded by SSA values.
    See `BitsSSAValue` to see how dependent integer values are passed around.
    """

    name = "dep_asl.bits"


@irdl_attr_definition
class IntegerType(ParametrizedAttribute):
    """
    An integer type.
    While integer types have additional annotations, these are done in the
    operations that are using them.
    See `IntegerSSAValue` to see how dependent integer values are passed around.
    """

    name = "dep_asl.int"


@dataclass
class BitsSSAValue:
    """
    A dependent bitvector SSA value.
    The bitvector width is represented as a dependent integer.
    """

    value: SSAValue
    width: SSAValue


@dataclass
class IntegerSSAValue:
    """
    A dependent integer SSA value.
    It is represented as a pair of two values: the value and the constraint.
    """

    value: SSAValue
    constraint: SSAValue | None


@irdl_op_definition
class ConstantIntOp(IRDLOperation):
    """A constant integer operation."""

    name = "asl_dep.constant_int"

    value_attr = prop_def(builtin.IntAttr)
    res_value = result_def(IntegerType)

    @property
    def value(self) -> int:
        return self.value_attr.data

    @property
    def res(self) -> IntegerSSAValue:
        # The value is constrained by itself, as it is a constant.
        return IntegerSSAValue(self.res_value, self.res_value)

    def __init__(self, value: int) -> None:
        super().__init__(
            properties={"value_attr": builtin.IntAttr(value)},
            result_types=[IntegerType()],
        )

    @classmethod
    def parse(cls, parser: Parser) -> ConstantIntOp:
        """Parse the operation."""
        value = parser.parse_integer(allow_boolean=False, allow_negative=False)
        op = ConstantIntOp(value)

        if attr_dict := parser.parse_optional_attr_dict():
            op.attributes = attr_dict

        return op

    def print(self, printer: Printer) -> None:
        """Print the operation."""
        printer.print_string(" ")
        printer.print_int(self.value_attr.data)
        if self.attributes:
            printer.print_string(" ")
            printer.print_attr_dict(self.attributes)


@irdl_op_definition
class ConstantBitsOp(IRDLOperation):
    """A constant bit vector operation."""

    name = "asl_dep.constant_bits"

    value_attr = prop_def(builtin.IntAttr)
    value_width = operand_def(IntegerType)

    res_value = result_def(BitsType())

    def __init__(
        self,
        value: int,
        value_width: SSAValue,
    ) -> None:
        super().__init__(
            result_types=[BitsType()],
            properties={"value_attr": builtin.IntAttr(value)},
            operands=[value_width],
        )

    @classmethod
    def parse(cls, parser: Parser) -> ConstantBitsOp:
        value = parser.parse_integer(allow_boolean=False, allow_negative=False)
        parser.parse_characters(":")
        parser.parse_identifier("bits")
        parser.parse_characters("<")
        value_width = parser.parse_operand()
        parser.parse_characters(">")
        attributes = parser.parse_optional_attr_dict()

        op = ConstantBitsOp(value, value_width)
        op.attributes = attributes
        return op

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_int(self.value_attr.data)
        printer.print_string(" : bits<")
        printer.print_ssa_value(self.value_width)
        printer.print_string(">")
        if self.attributes:
            printer.print_string(" ")
            printer.print_attr_dict(self.attributes)


class BinaryBitsOp(IRDLOperation):
    """A dependent binary bit vector operation."""

    lhs_value = operand_def(BitsType())
    lhs_width = operand_def(IntegerType())

    rhs_value = operand_def(BitsType())
    rhs_width = operand_def(IntegerType())

    res_width = operand_def(IntegerType())
    res = result_def(BitsType())

    assembly_format = """
      $lhs_value `,` $rhs_value `:`
      `(` `bits` `<` $lhs_width `>` `,` `bits` `<` $rhs_width `>` `)`
      `->` `bits` `<` $res_width `>` attr-dict
    """

    def __init__(
        self,
        lhs_value: SSAValue,
        lhs_width: SSAValue,
        rhs_value: SSAValue,
        rhs_width: SSAValue,
        res_width: SSAValue,
    ):
        super().__init__(
            operands=[lhs_value, lhs_width, rhs_value, rhs_width, res_width],
            result_types=[BitsType()],
        )


@irdl_op_definition
class AddBitsOp(BinaryBitsOp):
    """A bit vector addition operation."""

    name = "asl_dep.add_bits"


@irdl_op_definition
class SubBitsOp(BinaryBitsOp):
    """A bit vector subtraction operation."""

    name = "asl_dep.sub_bits"


@irdl_op_definition
class AndBitsOp(BinaryBitsOp):
    """A bit vector AND operation."""

    name = "asl_dep.and_bits"


@irdl_op_definition
class OrBitsOp(BinaryBitsOp):
    """A bit vector OR operation."""

    name = "asl_dep.or_bits"


@irdl_op_definition
class XorBitsOp(BinaryBitsOp):
    """A bit vector XOR operation."""

    name = "asl_dep.xor_bits"


ASLDepDialect = Dialect(
    "asl_dep",
    [
        ConstantIntOp,
        ConstantBitsOp,
        AddBitsOp,
        SubBitsOp,
        AndBitsOp,
        OrBitsOp,
        XorBitsOp,
    ],
    [
        BitsType,
        IntegerType,
    ],
)
