from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

from xdsl.dialects import builtin
from xdsl.dialects.utils import parse_func_op_like, print_func_op_like
from xdsl.ir import (
    Attribute,
    Block,
    Data,
    Dialect,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
    VerifyException,
)
from xdsl.irdl import (
    BaseAttr,
    IRDLOperation,
    ParameterDef,
    VarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import (
    CallableOpInterface,
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    SymbolOpInterface,
)


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


@irdl_op_definition
class NegateIntOp(IRDLOperation):
    """An integer negation operation."""

    name = "asl.negate_int"

    arg = operand_def(IntegerType())
    res = result_def(IntegerType())

    assembly_format = "$arg attr-dict"

    def __init__(self, arg: SSAValue, attr_dict: Mapping[str, Attribute] = {}):
        super().__init__(
            operands=[arg],
            result_types=[IntegerType()],
            attributes=attr_dict,
        )


class BinaryIntOp(IRDLOperation):
    """A binary integer operation."""

    lhs = operand_def(IntegerType())
    rhs = operand_def(IntegerType())
    res = result_def(IntegerType())

    assembly_format = "$lhs `,` $rhs attr-dict"

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

    name = "asl.shiftleft_int"


@irdl_op_definition
class ShiftRightIntOp(BinaryIntOp):
    """An integer right shift operation."""

    name = "asl.shiftright_int"


@irdl_op_definition
class DivIntOp(BinaryIntOp):
    """
    An integer division operation.
    The rhs is expected to be positive, and to divide the lhs exactly.
    """

    name = "asl.div_int"


@irdl_op_definition
class FDivIntOp(BinaryIntOp):
    """
    An integer division remainder operation.
    The rhs is expected to be positive, and the result is rounded down.
    """

    name = "asl.fdiv_int"


@irdl_op_definition
class FRemIntOp(BinaryIntOp):
    """
    An integer division remainder operation.
    The rhs is expected to be positive, and the result is positive as well.
    """

    name = "asl.frem_int"


class PredicateIntOp(IRDLOperation):
    """An integer predicate operation."""

    lhs = operand_def(IntegerType())
    rhs = operand_def(IntegerType())
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


class BinaryBitsOp(IRDLOperation):
    """A binary bit vector operation."""

    T: ClassVar = VarConstraint("T", BaseAttr(BitVectorType))

    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(T)

    assembly_format = "$lhs `,` $rhs `:` type($res) attr-dict"

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

    assembly_format = "$lhs `,` $rhs `:` type($res) attr-dict"

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

    assembly_format = "$lhs `,` $rhs `:` type($res) attr-dict"

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
class NotBitsOp(IRDLOperation):
    """A bitwise NOT operation."""

    name = "asl.not_bits"

    T: ClassVar = VarConstraint("T", BaseAttr(BitVectorType))

    arg = operand_def(T)
    res = result_def(T)

    assembly_format = "$arg `:` type($res) attr-dict"

    def __init__(self, arg: SSAValue, attr_dict: Mapping[str, Attribute] = {}):
        super().__init__(
            operands=[arg],
            result_types=[arg.type],
            attributes=attr_dict,
        )


@irdl_op_definition
class EqBitsOp(IRDLOperation):
    """A bit vector EQ operation."""

    name = "asl.eq_bits"

    T: ClassVar = VarConstraint("T", BaseAttr(BitVectorType))

    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(BoolType())

    assembly_format = "$lhs `,` $rhs `:` type($lhs) attr-dict"

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
class NeBitsOp(IRDLOperation):
    """A bit vector NE operation."""

    name = "asl.ne_bits"

    T: ClassVar = VarConstraint("T", BaseAttr(BitVectorType))

    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(BoolType())

    assembly_format = "$lhs `,` $rhs `:` type($lhs) attr-dict"

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
        if tuple(arg.type for arg in last_op.args) != self.function_type.outputs.data:
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

    args = var_operand_def()

    traits = traits_def(HasParent(FuncOp), IsTerminator())

    assembly_format = "($args^ `:` type($args))? attr-dict"

    def __init__(self, value: Sequence[SSAValue]):
        super().__init__(operands=[value])


@irdl_op_definition
class SliceSingleOp(IRDLOperation):
    """Slice a single element from a bit vector."""

    name = "asl.slice_single"

    bits = operand_def(BitVectorType)
    index = operand_def(IntegerType)

    res = result_def(BitVectorType(1))

    assembly_format = "$bits `[` $index `]` `:` type($bits) attr-dict"

    def __init__(
        self,
        bits: SSAValue,
        index: SSAValue,
    ):
        super().__init__(
            operands=[bits, index],
            result_types=[BitVectorType(1)],
        )


ASLDialect = Dialect(
    "asl",
    [
        # Constants
        ConstantBoolOp,
        ConstantIntOp,
        ConstantBitVectorOp,
        # Boolean operations
        NotOp,
        AndBoolOp,
        OrBoolOp,
        EqBoolOp,
        NeBoolOp,
        ImpliesBoolOp,
        EquivBoolOp,
        # Integer operations
        NegateIntOp,
        AddIntOp,
        SubIntOp,
        MulIntOp,
        ExpIntOp,
        ShiftLeftIntOp,
        ShiftRightIntOp,
        DivIntOp,
        FDivIntOp,
        FRemIntOp,
        EqIntOp,
        NeIntOp,
        LeIntOp,
        LtIntOp,
        GeIntOp,
        GtIntOp,
        # Bits operations
        AddBitsOp,
        SubBitsOp,
        AndBitsOp,
        OrBitsOp,
        XorBitsOp,
        AddBitsIntOp,
        SubBitsIntOp,
        NotBitsOp,
        EqBitsOp,
        NeBitsOp,
        # Functions
        ReturnOp,
        FuncOp,
        # Slices
        SliceSingleOp,
    ],
    [BoolType, BoolAttr, IntegerType, BitVectorType, BitVectorAttr],
)
