from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence
from typing import ClassVar, TypeAlias

from xdsl.dialects import builtin
from xdsl.dialects.builtin import (
    IntAttr,
    StringAttr,
    SymbolRefAttr,
    TupleType,
)
from xdsl.dialects.utils import parse_func_op_like, print_func_op_like
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
    VerifyException,
)
from xdsl.irdl import (
    AnyAttr,
    BaseAttr,
    GenericAttrConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    ParameterDef,
    VarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import (
    CallableOpInterface,
    ConstantLike,
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    Pure,
    SymbolOpInterface,
    SymbolTable,
    SymbolUserOpInterface,
)

from asl_xdsl.analysis.integer_range import (
    IntegerRange,
    IntegerRangeAnalysis,
    IntegerRangeTrait,
)


@irdl_attr_definition
class ConstraintExactAttr(ParametrizedAttribute):
    """A constraint on an integer attribute to be exactly equal to a value."""

    name = "asl.int_constraint_exact"

    value_attr: ParameterDef[builtin.IntAttr]

    def __init__(self, value: int):
        super().__init__([builtin.IntAttr(value)])

    @property
    def value(self) -> int:
        return self.value_attr.data

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> int:
        """Parse the attribute parameter."""
        with parser.in_angle_brackets():
            value = parser.parse_integer()
        return value

    def print_parameter(self, printer: Printer) -> None:
        """Print the attribute parameter."""
        printer.print("<", self.value, ">")


@irdl_attr_definition
class ConstraintRangeAttr(ParametrizedAttribute):
    """A constraint on an integer attribute to be within a range."""

    name = "asl.int_constraint_range"

    min_value_attr: ParameterDef[builtin.IntAttr]
    max_value_attr: ParameterDef[builtin.IntAttr]

    def __init__(self, min_value: int, max_value: int):
        super().__init__([builtin.IntAttr(min_value), builtin.IntAttr(max_value)])

    @property
    def min_value(self) -> int:
        return self.min_value_attr.data

    @property
    def max_value(self) -> int:
        return self.max_value_attr.data

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """Parse the attribute parameters."""
        with parser.in_angle_brackets():
            min_value = parser.parse_integer()
            parser.parse_characters(":")
            max_value = parser.parse_integer()
        return [builtin.IntAttr(min_value), builtin.IntAttr(max_value)]

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        printer.print("<", self.min_value, ":", self.max_value, ">")


def _parse_integer_constraint(
    parser: AttrParser,
) -> ConstraintExactAttr | ConstraintRangeAttr:
    """Parse an integer constraint using a shorthand syntax."""
    value = parser.parse_integer()
    if parser.parse_optional_characters(":") is None:
        return ConstraintExactAttr(value)
    second_value = parser.parse_integer()
    return ConstraintRangeAttr(value, second_value)


def _print_integer_constraint(
    constraint: ConstraintExactAttr | ConstraintRangeAttr, printer: Printer
):
    """Print an integer constraint using a shorthand syntax."""
    if isinstance(constraint, ConstraintExactAttr):
        printer.print(constraint.value)
    else:
        printer.print(constraint.min_value, ":", constraint.max_value)


@irdl_attr_definition
class StringType(ParametrizedAttribute, TypeAttribute):
    """A string type."""

    name = "asl.string"


@irdl_attr_definition
class IntegerType(ParametrizedAttribute, TypeAttribute):
    """An arbitrary-precision integer type."""

    name = "asl.int"

    constraints_attr: ParameterDef[
        builtin.ArrayAttr[ConstraintExactAttr | ConstraintRangeAttr]
    ]

    def __init__(self, constraints: Sequence[Attribute] = ()):
        super().__init__([builtin.ArrayAttr(constraints)])

    @property
    def constraints(self) -> Sequence[ConstraintExactAttr | ConstraintRangeAttr]:
        return self.constraints_attr.data

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        # Integer types with no constraints have no angle brackets
        if parser.parse_optional_characters("<") is None:
            return [builtin.ArrayAttr(())]

        constraints = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, lambda: _parse_integer_constraint(parser)
        )
        parser.parse_characters(">")
        return [builtin.ArrayAttr(constraints)]

    def print_parameters(self, printer: Printer) -> None:
        if not self.constraints:
            return
        printer.print("<")
        printer.print_list(
            self.constraints,
            lambda constr: _print_integer_constraint(constr, printer),
        )
        printer.print(">")


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


ArrayElementType: TypeAlias = BitVectorType


@irdl_attr_definition
class ArrayType(
    ParametrizedAttribute,
    TypeAttribute,
    builtin.ShapedType,
    builtin.ContainerType[ArrayElementType],
):
    """An array type."""

    name = "asl.array"

    shape: ParameterDef[builtin.ArrayAttr[builtin.IntAttr]]
    element_type: ParameterDef[ArrayElementType]

    def __init__(
        self,
        shape: builtin.ArrayAttr[builtin.IntAttr],
        element_type: ArrayElementType,
    ):
        super().__init__([shape, element_type])

    def verify(self) -> None:
        if not self.shape.data:
            raise VerifyException("asl.array shape must not be empty")

        for dim_attr in self.shape.data:
            if dim_attr.data < 0:
                raise VerifyException(
                    "asl.array array dimensions must have non-negative size"
                )

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.data for i in self.shape.data)

    def get_element_type(self) -> ArrayElementType:
        return self.element_type

    @classmethod
    def parse_parameters(cls, parser: AttrParser):
        with parser.in_angle_brackets():
            shape, type = parser.parse_ranked_shape()
            return builtin.ArrayAttr(builtin.IntAttr(dim) for dim in shape), type

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_list(
                self.shape, lambda dim: printer.print_string(f"{dim.data}"), "x"
            )
            printer.print_string("x")
            printer.print_attribute(self.element_type)

    @classmethod
    def constr(
        cls,
        element_type: IRDLAttrConstraint | None = None,
        *,
        shape: IRDLGenericAttrConstraint[builtin.ArrayAttr[IntAttr]] | None = None,
    ) -> GenericAttrConstraint[ArrayType]:
        if element_type is None and shape is None:
            return BaseAttr[ArrayType](ArrayType)
        shape_constr = AnyAttr() if shape is None else shape
        return ParamAttrConstraint[ArrayType](
            ArrayType,
            (
                shape_constr,
                element_type,
            ),
        )


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
        if self.value.data < 0 or self.value.data > self.maximum_value():
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


class ConstantIntIntegerRangeTrait(IntegerRangeTrait):
    @staticmethod
    def compute_analysis(op: Operation, analysis: IntegerRangeAnalysis) -> None:
        assert isinstance(op, ConstantIntOp), "Expected ConstantIntOp"
        analysis.set_range(op.res, IntegerRange(op.value.data, op.value.data))


@irdl_op_definition
class ConstantIntOp(IRDLOperation):
    """A constant arbitrary-sized integer operation."""

    name = "asl.constant_int"

    value = prop_def(builtin.IntAttr)
    res = result_def(IntegerType)

    traits = traits_def(ConstantIntIntegerRangeTrait())

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
        value = parser.parse_integer(allow_boolean=False, allow_negative=True)
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
class ConstantStringOp(IRDLOperation):
    """A constant string operation."""

    name = "asl.constant_string"

    value = prop_def(builtin.StringAttr)
    res = result_def(StringType)

    assembly_format = "$value attr-dict"

    def __init__(
        self, value: str | builtin.StringAttr, attr_dict: Mapping[str, Attribute] = {}
    ):
        if isinstance(value, str):
            value = builtin.StringAttr(value)
        super().__init__(
            result_types=[StringType()],
            properties={"value": value},
            attributes=attr_dict,
        )


class UnaryIntOp(IRDLOperation, ABC):
    """A unary integer operation."""

    arg = operand_def(IntegerType)
    res = result_def(IntegerType)

    assembly_format = "$arg `:` type($arg) `->` type($res) attr-dict"

    def __init__(
        self,
        arg: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[arg],
            result_types=[IntegerType()],
            attributes=attr_dict,
        )


@irdl_op_definition
class NegateIntOp(UnaryIntOp):
    """An integer negation operation."""

    name = "asl.neg_int"


@irdl_op_definition
class Pow2IntOp(UnaryIntOp):
    """
    An integer exponentiation operation.
    result == 2 ** x
    """

    name = "asl.pow2_int"


class BinaryIntOp(IRDLOperation, ABC):
    """A binary integer operation."""

    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    res = result_def(IntegerType)

    assembly_format = (
        "$lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) attr-dict"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[IntegerType()],
            attributes=attr_dict,
        )


@irdl_op_definition
class AddIntOp(BinaryIntOp):
    """An integer addition operation."""

    name = "asl.add_int"


@irdl_op_definition
class SubIntOp(BinaryIntOp):
    """An integer subtraction operation."""

    name = "asl.sub_int"


@irdl_op_definition
class MulIntOp(BinaryIntOp):
    """An integer multiplication operation."""

    name = "asl.mul_int"


@irdl_op_definition
class ExpIntOp(BinaryIntOp):
    """An integer exponentiation operation."""

    name = "asl.exp_int"


@irdl_op_definition
class ShiftLeftIntOp(BinaryIntOp):
    """An integer left shift operation."""

    name = "asl.shl_int"


@irdl_op_definition
class ShiftRightIntOp(BinaryIntOp):
    """An integer right shift operation."""

    name = "asl.shr_int"


@irdl_op_definition
class ExactDivIntOp(BinaryIntOp):
    """
    An integer division operation.
    The rhs is expected to be positive, and to divide the lhs exactly.
    """

    name = "asl.exact_div_int"


@irdl_op_definition
class FloorDivIntOp(BinaryIntOp):
    """
    An integer division operation.
    Calculates "floor(x / y)"
    That is, the result is rounded down to negative infinity.
    """

    name = "asl.fdiv_int"


@irdl_op_definition
class FloorRemIntOp(BinaryIntOp):
    """
    An integer division remainder operation.
    Pairs with asl.fdiv_int: "result == x - y * asl.fdiv_int(x, y)"
    """

    name = "asl.frem_int"


@irdl_op_definition
class ZeroDivIntOp(BinaryIntOp):
    """
    An integer division operation.
    Calculates "round_to_zero(x / y)"
    """

    name = "asl.zdiv_int"


@irdl_op_definition
class ZeroRemIntOp(BinaryIntOp):
    """
    An integer division remainder operation.
    Pairs with asl.zdiv_int: "result == x - y * asl.zdiv_int(x, y)"
    """

    name = "asl.zrem_int"


@irdl_op_definition
class AlignIntOp(BinaryIntOp):
    """
    An integer alignment operation.
    Rounds x down to a multiple of 2**y.
    result == asl.fdiv_int(x, 2**y)
    """

    name = "asl.align_int"


class MoxPow2IntIntegerRangeTrait(IntegerRangeTrait):
    @staticmethod
    def compute_analysis(op: Operation, analysis: IntegerRangeAnalysis) -> None:
        assert isinstance(op, ModPow2IntOp), "Expected ModPow2IntOp"
        rhs_upper = analysis.get_range(op.rhs).upper_bound
        lhs_bounds = analysis.get_range(op.lhs)
        if rhs_upper is None:
            # If the rhs is unbounded, the result is not more bounded than the lhs.
            analysis.set_range(op.res, lhs_bounds)
            return

        bound_from_rhs = IntegerRange(0, 2**rhs_upper - 1)
        analysis.set_range(op.res, lhs_bounds & bound_from_rhs)


@irdl_op_definition
class ModPow2IntOp(BinaryIntOp):
    """
    An integer division remainder operation.
    Pairs with asl.align_int: "result == asl.frem_int(x, 2**y)"
    """

    name = "asl.mod_pow2_int"

    traits = traits_def(MoxPow2IntIntegerRangeTrait())


@irdl_op_definition
class IsPow2IntOp(IRDLOperation):
    """An integer power-of-two predicate operation."""

    arg = operand_def(IntegerType)
    res = result_def(builtin.IntegerType(1))

    name = "asl.is_pow2_int"

    assembly_format = "$arg `:` type($arg) `->` type($res) attr-dict"

    def __init__(
        self,
        arg: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[arg],
            result_types=[builtin.IntegerType(1)],
            attributes=attr_dict,
        )


class PredicateIntOp(IRDLOperation, ABC):
    """An integer predicate operation."""

    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    res = result_def(builtin.IntegerType(1))

    assembly_format = (
        "$lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) attr-dict"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[builtin.IntegerType(1)],
            attributes=attr_dict,
        )


@irdl_op_definition
class EqIntOp(PredicateIntOp):
    """An integer equality operation."""

    name = "asl.eq_int"


@irdl_op_definition
class NeIntOp(PredicateIntOp):
    """An integer inequality operation."""

    name = "asl.ne_int"


@irdl_op_definition
class LeIntOp(PredicateIntOp):
    """An integer less-than-or-equal operation."""

    name = "asl.le_int"


@irdl_op_definition
class LtIntOp(PredicateIntOp):
    """An integer less-than operation."""

    name = "asl.lt_int"


@irdl_op_definition
class GeIntOp(PredicateIntOp):
    """An integer greater-than-or-equal operation."""

    name = "asl.ge_int"


@irdl_op_definition
class GtIntOp(PredicateIntOp):
    """An integer greater-than operation."""

    name = "asl.gt_int"


class BinaryBitsOp(IRDLOperation, ABC):
    """A binary bit vector operation."""

    T: ClassVar = VarConstraint("T", BaseAttr(BitVectorType))

    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(T)

    assembly_format = (
        "$lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) attr-dict"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[lhs.type],
            attributes=attr_dict,
        )


@irdl_op_definition
class AddBitsOp(BinaryBitsOp):
    """A bit vector addition operation."""

    name = "asl.add_bits"


@irdl_op_definition
class SubBitsOp(BinaryBitsOp):
    """A bit vector subtraction operation."""

    name = "asl.sub_bits"


@irdl_op_definition
class MulBitsOp(BinaryBitsOp):
    """A bit vector multiplication operation."""

    name = "asl.mul_bits"


@irdl_op_definition
class AndBitsOp(BinaryBitsOp):
    """A bit vector AND operation."""

    name = "asl.and_bits"


@irdl_op_definition
class OrBitsOp(BinaryBitsOp):
    """A bit vector OR operation."""

    name = "asl.or_bits"


@irdl_op_definition
class XorBitsOp(BinaryBitsOp):
    """A bit vector XOR operation."""

    name = "asl.xor_bits"


@irdl_op_definition
class AddBitsIntOp(IRDLOperation):
    """A bit vector addition operation with an integer."""

    name = "asl.add_bits_int"

    T: ClassVar = VarConstraint("T", BaseAttr(BitVectorType))

    lhs = operand_def(T)
    rhs = operand_def(IntegerType())
    res = result_def(T)

    assembly_format = (
        "$lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) attr-dict"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[lhs.type],
            attributes=attr_dict,
        )


@irdl_op_definition
class SubBitsIntOp(IRDLOperation):
    """A bit vector subtraction operation with an integer."""

    name = "asl.sub_bits_int"

    T: ClassVar = VarConstraint("T", BaseAttr(BitVectorType))

    lhs = operand_def(T)
    rhs = operand_def(IntegerType())
    res = result_def(T)

    assembly_format = (
        "$lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) attr-dict"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[lhs.type],
            attributes=attr_dict,
        )


@irdl_op_definition
class LslBitsOp(IRDLOperation):
    """A bit vector logical left shift operation."""

    name = "asl.lsl_bits"

    T: ClassVar = VarConstraint("T", BaseAttr(BitVectorType))

    lhs = operand_def(T)
    rhs = operand_def(IntegerType())
    res = result_def(T)

    assembly_format = (
        "$lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) attr-dict"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[lhs.type],
            attributes=attr_dict,
        )


@irdl_op_definition
class LsrBitsOp(IRDLOperation):
    """A bit vector logical shift right operation."""

    name = "asl.lsr_bits"

    T: ClassVar = VarConstraint("T", BaseAttr(BitVectorType))

    lhs = operand_def(T)
    rhs = operand_def(IntegerType())
    res = result_def(T)

    assembly_format = (
        "$lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) attr-dict"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[lhs.type],
            attributes=attr_dict,
        )


@irdl_op_definition
class AsrBitsOp(IRDLOperation):
    """A bit vector arithmetic shift right operation."""

    name = "asl.asr_bits"

    T: ClassVar = VarConstraint("T", BaseAttr(BitVectorType))

    lhs = operand_def(T)
    rhs = operand_def(IntegerType())
    res = result_def(T)

    assembly_format = (
        "$lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) attr-dict"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[lhs.type],
            attributes=attr_dict,
        )


@irdl_op_definition
class ZeroExtendBitsOp(IRDLOperation):
    """A bit vector zero-extend operation."""

    name = "asl.zero_extend_bits"

    lhs = operand_def(BitVectorType)
    rhs = operand_def(IntegerType())
    res = result_def(BitVectorType)

    assembly_format = (
        "$lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) attr-dict"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        res: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[res.type],
            attributes=attr_dict,
        )


@irdl_op_definition
class SignExtendBitsOp(IRDLOperation):
    """A bit vector zero-extend operation."""

    name = "asl.sign_extend_bits"

    lhs = operand_def(BitVectorType)
    rhs = operand_def(IntegerType())
    res = result_def(BitVectorType)

    assembly_format = (
        "$lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) attr-dict"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        res: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[res.type],
            attributes=attr_dict,
        )


@irdl_op_definition
class AppendBitsOp(IRDLOperation):
    """A bit vector append operation."""

    name = "asl.append_bits"

    lhs = operand_def(BitVectorType)
    rhs = operand_def(BitVectorType)
    res = result_def(BitVectorType)

    assembly_format = (
        "$lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) attr-dict"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        res: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[res.type],
            attributes=attr_dict,
        )


@irdl_op_definition
class ReplicateBitsOp(IRDLOperation):
    """A bit vector replication operation."""

    name = "asl.replicate_bits"

    lhs = operand_def(BitVectorType)
    rhs = operand_def(IntegerType())
    res = result_def(BitVectorType)

    assembly_format = (
        "$lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) attr-dict"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        res: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[res.type],
            attributes=attr_dict,
        )


@irdl_op_definition
class ZerosBitsOp(IRDLOperation):
    """A bit vector all-zeros operation."""

    name = "asl.zeros_bits"

    arg = operand_def(IntegerType())
    res = result_def(BitVectorType)

    assembly_format = "$arg `:` type($arg) `->` type($res) attr-dict"

    def __init__(
        self,
        arg: SSAValue,
        res: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[arg],
            result_types=[res.type],
            attributes=attr_dict,
        )


@irdl_op_definition
class OnesBitsOp(IRDLOperation):
    """A bit vector all-ones operation."""

    name = "asl.ones_bits"

    arg = operand_def(IntegerType())
    res = result_def(BitVectorType)

    assembly_format = "$arg `:` type($arg) `->` type($res) attr-dict"

    def __init__(
        self,
        arg: SSAValue,
        res: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[arg],
            result_types=[res.type],
            attributes=attr_dict,
        )


@irdl_op_definition
class MkMaskBitsOp(IRDLOperation):
    """
    A bit vector mask generation operation.
    `mk_mask(x, N) : bits(N)` consists of `x` ones.
    For example, `mk_mask(3, 8) == '0000 0111'`.
    """

    name = "asl.mk_mask"

    lhs = operand_def(IntegerType())
    rhs = operand_def(IntegerType())
    res = result_def(BitVectorType)

    assembly_format = (
        "$lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) attr-dict"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        res: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[res.type],
            attributes=attr_dict,
        )


@irdl_op_definition
class NotBitsOp(IRDLOperation):
    """A bitwise NOT operation."""

    name = "asl.not_bits"

    T: ClassVar = VarConstraint("T", BaseAttr(BitVectorType))

    arg = operand_def(T)
    res = result_def(T)

    assembly_format = "$arg `:` type($arg) `->` type($res) attr-dict"

    def __init__(self, arg: SSAValue, attr_dict: Mapping[str, Attribute] = {}):
        super().__init__(
            operands=[arg],
            result_types=[arg.type],
            attributes=attr_dict,
        )


@irdl_op_definition
class CvtBitsSIntOp(IRDLOperation):
    """A conversion operation from bitvectors to signed integers."""

    name = "asl.cvt_bits_sint"

    arg = operand_def(BitVectorType)
    res = result_def(IntegerType)

    assembly_format = "$arg `:` type($arg) `->` type($res) attr-dict"

    def __init__(self, arg: SSAValue, attr_dict: Mapping[str, Attribute] = {}):
        super().__init__(
            operands=[arg],
            result_types=[IntegerType()],
            attributes=attr_dict,
        )


@irdl_op_definition
class CvtBitsUIntOp(IRDLOperation):
    """A conversion operation from bitvectors to unsigned integers."""

    name = "asl.cvt_bits_uint"

    arg = operand_def(BitVectorType)
    res = result_def(IntegerType)

    assembly_format = "$arg `:` type($arg) `->` type($res) attr-dict"

    def __init__(self, arg: SSAValue, attr_dict: Mapping[str, Attribute] = {}):
        super().__init__(
            operands=[arg],
            result_types=[IntegerType()],
            attributes=attr_dict,
        )


@irdl_op_definition
class EqBitsOp(IRDLOperation):
    """A bit vector EQ operation."""

    name = "asl.eq_bits"

    T: ClassVar = VarConstraint("T", BaseAttr(BitVectorType))

    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(builtin.IntegerType(1))

    assembly_format = (
        "$lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) attr-dict"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[builtin.IntegerType(1)],
            attributes=attr_dict,
        )


@irdl_op_definition
class NeBitsOp(IRDLOperation):
    """A bit vector NE operation."""

    name = "asl.ne_bits"

    T: ClassVar = VarConstraint("T", BaseAttr(BitVectorType))

    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(builtin.IntegerType(1))

    assembly_format = (
        "$lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) attr-dict"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[builtin.IntegerType(1)],
            attributes=attr_dict,
        )


@irdl_op_definition
class PrintBitsHexOp(IRDLOperation):
    """A bit vector print function."""

    # Eventually, this should be an external function
    # This is just a workaround until we can cope with
    # bitwidth polymorphism.

    name = "asl.print_bits_hex"

    arg = operand_def(BitVectorType)

    assembly_format = "$arg `:` type($arg) `->` `(` `)` attr-dict"

    def __init__(
        self,
        arg: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[arg],
            result_types=[],
            attributes=attr_dict,
        )


@irdl_op_definition
class PrintSIntNHexOp(IRDLOperation):
    """A sized integer print function."""

    # Eventually, this should be an external function
    # This is just a workaround until we can cope with
    # bitwidth polymorphism.

    name = "asl.print_sintN_hex"

    arg = operand_def(builtin.IntegerType)

    assembly_format = "$arg `:` type($arg) `->` `(` `)` attr-dict"

    def __init__(
        self,
        arg: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[arg],
            result_types=[],
            attributes=attr_dict,
        )


@irdl_op_definition
class PrintSIntNDecOp(IRDLOperation):
    """A sized integer print function."""

    # Eventually, this should be an external function
    # This is just a workaround until we can cope with
    # bitwidth polymorphism.

    name = "asl.print_sintN_dec"

    arg = operand_def(builtin.IntegerType)

    assembly_format = "$arg `:` type($arg) `->` `(` `)` attr-dict"

    def __init__(
        self,
        arg: SSAValue,
        attr_dict: Mapping[str, Attribute] = {},
    ):
        super().__init__(
            operands=[arg],
            result_types=[],
            attributes=attr_dict,
        )


class FuncOpCallableInterface(CallableOpInterface):
    @classmethod
    def get_callable_region(cls, op: Operation) -> Region:
        assert isinstance(op, FuncOp)
        return op.body

    @classmethod
    def get_argument_types(cls, op: Operation) -> tuple[Attribute, ...]:
        assert isinstance(op, FuncOp)
        return op.function_type.inputs.data

    @classmethod
    def get_result_types(cls, op: Operation) -> tuple[Attribute, ...]:
        assert isinstance(op, FuncOp)
        return op.function_type.outputs.data


@irdl_op_definition
class FuncOp(IRDLOperation):
    """A function operation."""

    name = "asl.func"

    body = region_def()
    sym_name = prop_def(builtin.StringAttr)
    function_type = prop_def(builtin.FunctionType)

    traits = traits_def(
        IsolatedFromAbove(), SymbolOpInterface(), FuncOpCallableInterface()
    )

    def __init__(
        self,
        name: str,
        function_type: builtin.FunctionType,
        region: Region | None = None,
    ):
        if region is None:
            region = Region(Block(arg_types=function_type.inputs))
        properties: dict[str, Attribute | None] = {
            "sym_name": builtin.StringAttr(name),
            "function_type": function_type,
        }
        super().__init__(properties=properties, regions=[region])

    def verify_(self) -> None:
        # If this is an empty region (external function), then return
        if len(self.body.blocks) == 0:
            return

        entry_block = self.body.blocks.first
        assert entry_block is not None
        block_arg_types = entry_block.arg_types
        if self.function_type.inputs.data != tuple(block_arg_types):
            raise VerifyException(
                "Expected entry block arguments to have the same types as the function "
                "input types"
            )

        if not isinstance(last_op := entry_block.last_op, ReturnOp):
            raise VerifyException("Expected last operation of function to be a return")
        arg_types = () if last_op.arg is None else (last_op.arg.type,)
        if arg_types != self.function_type.outputs.data:
            raise VerifyException(
                "Expected return types to match the function output types"
            )

    @classmethod
    def parse(cls, parser: Parser) -> FuncOp:
        (
            name,
            input_types,
            return_types,
            region,
            extra_attrs,
            _,
            _,
        ) = parse_func_op_like(
            parser, reserved_attr_names=("sym_name", "function_type")
        )
        func = FuncOp(
            name=name,
            function_type=builtin.FunctionType.from_lists(input_types, return_types),
            region=region,
        )
        if extra_attrs is not None:
            func.attributes |= extra_attrs.data
        return func

    def print(self, printer: Printer):
        print_func_op_like(
            printer,
            self.sym_name,
            self.function_type,
            self.body,
            self.attributes,
            reserved_attr_names=(
                "sym_name",
                "function_type",
            ),
        )


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """
    A return operation.
    Should be the last operation of a function
    """

    name = "asl.return"

    arg = opt_operand_def()

    traits = traits_def(HasParent(FuncOp), IsTerminator())

    assembly_format = "($arg^ `:` type($arg))? attr-dict"

    def __init__(self, value: SSAValue | None = None):
        super().__init__(operands=[value])


class CallOpSymbolUserOpInterface(SymbolUserOpInterface):
    def verify(self, op: Operation) -> None:
        assert isinstance(op, CallOp)

        found_callee = SymbolTable.lookup_symbol(op, op.callee)
        if not found_callee:
            raise VerifyException(f"'{op.callee}' could not be found in symbol table")

        if not isinstance(found_callee, FuncOp):
            raise VerifyException(f"'{op.callee}' does not reference a valid function")

        if len(found_callee.function_type.inputs) != len(op.arguments):
            raise VerifyException("incorrect number of operands for callee")

        if len(found_callee.function_type.outputs) != len(op.result_types):
            raise VerifyException("incorrect number of results for callee")

        for idx, (found_operand, operand) in enumerate(
            zip(found_callee.function_type.inputs, (arg.type for arg in op.arguments))
        ):
            if found_operand != operand:
                raise VerifyException(
                    f"operand type mismatch: expected operand type {found_operand}, but"
                    f" provided {operand} for operand number {idx}"
                )

        for idx, (found_res, res) in enumerate(
            zip(found_callee.function_type.outputs, op.result_types)
        ):
            if found_res != res:
                raise VerifyException(
                    f"result type mismatch: expected result type {found_res}, but"
                    f" provided {res} for result number {idx}"
                )

        return


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "asl.call"
    arguments = var_operand_def()
    callee = prop_def(builtin.FlatSymbolRefAttrConstr)
    res = var_result_def()

    traits = traits_def(
        CallOpSymbolUserOpInterface(),
    )

    assembly_format = (
        "$callee `(` $arguments `)` attr-dict `:` functional-type($arguments, $res)"
    )

    def __init__(
        self,
        callee: str | builtin.SymbolRefAttr,
        arguments: Sequence[SSAValue | Operation],
        return_types: Sequence[Attribute],
    ):
        if isinstance(callee, str):
            callee = builtin.SymbolRefAttr(callee)
        super().__init__(
            operands=[arguments],
            result_types=[return_types],
            properties={"callee": callee},
        )


@irdl_op_definition
class GetSliceOp(IRDLOperation):
    """Extract a slice from a bit vector."""

    name = "asl.get_slice"

    bits = operand_def(BitVectorType)
    index = operand_def(IntegerType)
    width = operand_def(IntegerType)

    res = result_def(BitVectorType)

    assembly_format = (
        "$bits `,` $index `,` $width "
        "`:` `(` type($bits) `,` type($index) `,` type($width) `)` "
        "`->` type($res) attr-dict"
    )

    def __init__(
        self,
        bits: SSAValue,
        index: SSAValue,
        width: SSAValue,
        res: SSAValue,
    ):
        super().__init__(
            operands=[bits, index, width],
            result_types=[res.type],
        )


@irdl_op_definition
class SetSliceOp(IRDLOperation):
    """Insert a slice into a bit vector."""

    name = "asl.set_slice"

    S: ClassVar = VarConstraint("S", BaseAttr(BitVectorType))

    bits = operand_def(S)
    index = operand_def(IntegerType)
    width = operand_def(IntegerType)
    rhs = operand_def(BitVectorType)
    res = result_def(S)

    assembly_format = (
        "$bits `,` $index `,` $width `,` $rhs "
        "`:` `(` type($bits) `,` type($index) `,` type($width) `,` type($rhs) `)` "
        "`->` type($res) attr-dict"
    )

    def __init__(
        self,
        bits: SSAValue,
        index: SSAValue,
        width: SSAValue,
        rhs: SSAValue,
    ):
        super().__init__(
            operands=[bits, index, width, rhs],
            result_types=[bits.type],
        )


@irdl_attr_definition
class ReferenceType(ParametrizedAttribute, TypeAttribute):
    """
    The type of a reference to an object
    """

    name = "asl.ref"
    type: ParameterDef[Attribute]

    def print_parameters(self, printer: Printer) -> None:
        # We need this to pretty print a tuple and its members if
        # this is referencing one, otherwise just let the type
        # handle its own printing
        printer.print("<")
        printer.print(self.type)
        printer.print(">")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        # This is complicated by the fact we need to parse tuple
        # here also as the buildin dialect does not support this
        # yet
        parser.parse_characters("<")
        has_tuple = parser.parse_optional_keyword("tuple")
        if has_tuple is None:
            param_type = parser.parse_type()
            parser.parse_characters(">")
            return [param_type]
        else:
            # If its a tuple then there are any number of types
            def parse_types():
                return parser.parse_type()

            param_types = parser.parse_comma_separated_list(
                parser.Delimiter.ANGLE, parse_types
            )
            parser.parse_characters(">")
            return [TupleType(param_types)]

    @classmethod
    def constr(
        cls,
        type: IRDLAttrConstraint | None = None,
    ) -> GenericAttrConstraint[ReferenceType]:
        if type is None:
            return BaseAttr[ReferenceType](ReferenceType)
        return ParamAttrConstraint[ReferenceType](
            ReferenceType,
            (type,),
        )


@irdl_op_definition
class GlobalOp(IRDLOperation):
    name = "asl.global"

    assembly_format = "$sym_name `:` $global_type attr-dict"

    global_type = prop_def(Attribute)
    sym_name = prop_def(StringAttr)

    traits = traits_def(SymbolOpInterface())

    def __init__(
        self,
        global_type: Attribute,
        sym_name: str | StringAttr,
    ):
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)

        props: dict[str, Attribute] = {
            "global_type": global_type,
            "sym_name": sym_name,
        }

        super().__init__(properties=props)


@irdl_op_definition
class AddressOfOp(IRDLOperation):
    """
    Convert a global reference to an SSA-value to be
    used in other operations.

    %p = asl.address_of @symbol : !asl.ref<i1>
    """

    name = "asl.address_of"

    traits = traits_def(
        ConstantLike(),
        Pure(),
    )

    symbol = prop_def(SymbolRefAttr)
    res = result_def()
    
    assembly_format = "`(` $symbol `)` `:` type($res) attr-dict"

    assembly_format = "$symbol `:` type($res) attr-dict"


@irdl_op_definition
class ArrayRefOp(IRDLOperation):
    """
    Create a ref for an array element.

    %element_ref = asl.array_ref %array_ref [ %index ]
        : !asl.ref<!asl.array<16 x i8>> -> !asl.ref<i8>
    """

    name = "asl.array_ref"

    traits = traits_def(
        Pure(),
    )

    T: ClassVar = VarConstraint("T", AnyAttr())
    A: ClassVar = VarConstraint("A", ArrayType.constr(T))
    ref = operand_def(ReferenceType.constr(A))
    index = operand_def(IntegerType())
    res = result_def(ReferenceType.constr(T))

    assembly_format = "$ref `[` $index `]` `:` type($ref) `->` type($res) attr-dict"


@irdl_op_definition
class LoadOp(IRDLOperation):
    """
    Load from a reference.
    """

    name = "asl.load"

    T: ClassVar = VarConstraint("T", AnyAttr())
    ref = operand_def(ReferenceType.constr(T))
    res = result_def(T)

    assembly_format = "`from` $ref `:` type($res) attr-dict"


@irdl_op_definition
class StoreOp(IRDLOperation):
    """
    Store from a reference.
    """

    name = "asl.store"

    T: ClassVar = VarConstraint("T", AnyAttr())
    ref = operand_def(ReferenceType.constr(T))
    value = operand_def(T)

    assembly_format = "$value `to` $ref `:` type($value) attr-dict"


ASLDialect = Dialect(
    "asl",
    [
        # Constants
        ConstantIntOp,
        ConstantBitVectorOp,
        ConstantStringOp,
        # Integer operations
        NegateIntOp,
        AddIntOp,
        SubIntOp,
        MulIntOp,
        ExpIntOp,
        ShiftLeftIntOp,
        ShiftRightIntOp,
        ExactDivIntOp,
        FloorDivIntOp,
        FloorRemIntOp,
        ZeroDivIntOp,
        ZeroRemIntOp,
        AlignIntOp,
        ModPow2IntOp,
        IsPow2IntOp,
        Pow2IntOp,
        EqIntOp,
        NeIntOp,
        LeIntOp,
        LtIntOp,
        GeIntOp,
        GtIntOp,
        # Bits operations
        AddBitsOp,
        SubBitsOp,
        MulBitsOp,
        AndBitsOp,
        OrBitsOp,
        XorBitsOp,
        LslBitsOp,
        LsrBitsOp,
        AsrBitsOp,
        ZeroExtendBitsOp,
        SignExtendBitsOp,
        AppendBitsOp,
        ReplicateBitsOp,
        ZerosBitsOp,
        OnesBitsOp,
        MkMaskBitsOp,
        AddBitsIntOp,
        SubBitsIntOp,
        NotBitsOp,
        CvtBitsSIntOp,
        CvtBitsUIntOp,
        EqBitsOp,
        NeBitsOp,
        PrintBitsHexOp,
        PrintSIntNHexOp,
        PrintSIntNDecOp,
        # Functions
        ReturnOp,
        FuncOp,
        CallOp,
        # Slices
        GetSliceOp,
        SetSliceOp,
        # References
        GlobalOp,
        AddressOfOp,
        ArrayRefOp,
        LoadOp,
        StoreOp,
    ],
    [
        IntegerType,
        BitVectorType,
        BitVectorAttr,
        StringType,
        ArrayType,
        ReferenceType,
    ],
)
