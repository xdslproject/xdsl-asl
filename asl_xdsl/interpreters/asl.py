from typing import Any

from xdsl.dialects import builtin
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    TerminatorValue,
    impl,
    impl_callable,
    impl_external,
    impl_terminator,
    register_impls,
)
from xdsl.ir import Operation
from xdsl.utils.comparisons import to_signed, to_unsigned
from xdsl.utils.hints import isa

from asl_xdsl.dialects import asl


@register_impls
class ASLFunctions(InterpreterFunctions):
    @impl_terminator(asl.ReturnOp)
    def run_return(
        self, interpreter: Interpreter, op: asl.ReturnOp, args: tuple[Any, ...]
    ) -> tuple[TerminatorValue, PythonValues]:
        return ReturnedValues(args), ()

    @impl_callable(asl.FuncOp)
    def call_func(
        self, interpreter: Interpreter, op: asl.FuncOp, args: tuple[Any, ...]
    ):
        if (first_block := op.body.blocks.first) is None or not first_block.ops:
            return interpreter.call_external(op.sym_name.data, op, args)
        else:
            return interpreter.run_ssacfg_region(op.body, args, op.sym_name.data)

    @impl(asl.CallOp)
    def run_call(
        self, interpreter: Interpreter, op: asl.CallOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return interpreter.call_op(op.callee.string_value(), args)

    @impl(asl.NegateIntOp)
    def run_neg_int(
        self, interpreter: Interpreter, op: asl.NegateIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        arg: int
        [arg] = args
        return (0 - arg,)

    @impl(asl.AddIntOp)
    def run_add_int(
        self, interpreter: Interpreter, op: asl.AddIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        return (lhs + rhs,)

    @impl(asl.SubIntOp)
    def run_sub_int(
        self, interpreter: Interpreter, op: asl.SubIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        return (lhs - rhs,)

    @impl(asl.MulIntOp)
    def run_mul_int(
        self, interpreter: Interpreter, op: asl.MulIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        return (lhs * rhs,)

    @impl(asl.ExactDivIntOp)
    def run_exact_div_int(
        self, interpreter: Interpreter, op: asl.ExactDivIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert rhs != 0
        assert lhs % rhs == 0
        return (lhs // rhs,)

    @impl(asl.FloorDivIntOp)
    def run_floor_div_int(
        self, interpreter: Interpreter, op: asl.FloorDivIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert rhs != 0
        return (lhs // rhs,)

    @impl(asl.FloorRemIntOp)
    def run_floor_rem_int(
        self, interpreter: Interpreter, op: asl.FloorRemIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert rhs != 0
        return (lhs % rhs,)

    @impl(asl.ZeroDivIntOp)
    def run_zero_div_int(
        self, interpreter: Interpreter, op: asl.ZeroDivIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert rhs != 0
        div = abs(lhs) // abs(rhs)
        if (lhs > 0) != (rhs > 0):
            div = -div
        return (div,)

    @impl(asl.ZeroRemIntOp)
    def run_zero_rem_int(
        self, interpreter: Interpreter, op: asl.ZeroRemIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert rhs != 0
        div = abs(lhs) // abs(rhs)
        if (lhs > 0) != (rhs > 0):
            div = -div
        return (lhs - div * rhs,)

    @impl(asl.AlignIntOp)
    def run_align_int(
        self, interpreter: Interpreter, op: asl.AlignIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert lhs >= 0
        assert rhs >= 0
        return ((lhs >> rhs) << rhs,)

    @impl(asl.ModPow2IntOp)
    def run_mod_pow2_int(
        self, interpreter: Interpreter, op: asl.ModPow2IntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert lhs >= 0
        assert rhs >= 0
        return (lhs & ((1 << rhs) - 1),)

    @impl(asl.IsPow2IntOp)
    def run_is_pow2_int(
        self, interpreter: Interpreter, op: asl.IsPow2IntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        arg: int
        [arg] = args
        return (arg > 0 and arg & (arg - 1) == 0,)

    @impl(asl.Pow2IntOp)
    def run_pow2_int(
        self, interpreter: Interpreter, op: asl.Pow2IntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        arg: int
        [arg] = args
        return (1 << arg,)

    @impl(asl.ShiftLeftIntOp)
    def run_shl_int(
        self, interpreter: Interpreter, op: asl.ShiftLeftIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert rhs >= 0
        return (lhs << rhs,)

    @impl(asl.ShiftRightIntOp)
    def run_shr_int(
        self, interpreter: Interpreter, op: asl.ShiftRightIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert rhs >= 0
        return (lhs >> rhs,)

    @impl(asl.EqIntOp)
    def run_eq_int(
        self, interpreter: Interpreter, op: asl.EqIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        return (lhs == rhs,)

    @impl(asl.NeIntOp)
    def run_ne_int(
        self, interpreter: Interpreter, op: asl.NeIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        return (lhs != rhs,)

    @impl(asl.LeIntOp)
    def run_le_int(
        self, interpreter: Interpreter, op: asl.LeIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        return (lhs <= rhs,)

    @impl(asl.LtIntOp)
    def run_lt_int(
        self, interpreter: Interpreter, op: asl.LtIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        return (lhs < rhs,)

    @impl(asl.GeIntOp)
    def run_ge_int(
        self, interpreter: Interpreter, op: asl.GeIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        return (lhs >= rhs,)

    @impl(asl.GtIntOp)
    def run_gt_int(
        self, interpreter: Interpreter, op: asl.GtIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        return (lhs > rhs,)

    @impl(asl.AddBitsOp)
    def run_add_bits(
        self, interpreter: Interpreter, op: asl.AddBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(op.lhs.type, asl.BitVectorType)
        width = op.lhs.type.width.data
        lhs: int
        rhs: int
        (lhs, rhs) = args
        result = to_unsigned(lhs + rhs, width)
        return (result,)

    @impl(asl.SubBitsOp)
    def run_sub_bits(
        self, interpreter: Interpreter, op: asl.SubBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(op.lhs.type, asl.BitVectorType)
        width = op.lhs.type.width.data
        lhs: int
        rhs: int
        (lhs, rhs) = args
        result = to_unsigned(lhs - rhs, width)
        return (result,)

    @impl(asl.MulBitsOp)
    def run_mul_bits(
        self, interpreter: Interpreter, op: asl.MulBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(op.lhs.type, asl.BitVectorType)
        width = op.lhs.type.width.data
        lhs: int
        rhs: int
        (lhs, rhs) = args
        result = to_unsigned(lhs * rhs, width)
        return (result,)

    @impl(asl.AndBitsOp)
    def run_and_bits(
        self, interpreter: Interpreter, op: asl.AndBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        return (lhs & rhs,)

    @impl(asl.OrBitsOp)
    def run_or_bits(
        self, interpreter: Interpreter, op: asl.OrBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        return (lhs | rhs,)

    @impl(asl.XorBitsOp)
    def run_xor_bits(
        self, interpreter: Interpreter, op: asl.XorBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        return (lhs ^ rhs,)

    @impl(asl.LslBitsOp)
    def run_lsl_bits(
        self, interpreter: Interpreter, op: asl.LslBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(op.lhs.type, asl.BitVectorType)
        width = op.lhs.type.width.data
        lhs: int
        rhs: int
        (lhs, rhs) = args
        result = to_unsigned(lhs << rhs, width)
        return (result,)

    @impl(asl.LsrBitsOp)
    def run_lsr_bits(
        self, interpreter: Interpreter, op: asl.LsrBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(op.lhs.type, asl.BitVectorType)
        width = op.lhs.type.width.data
        lhs: int
        rhs: int
        (lhs, rhs) = args
        result = to_unsigned(lhs >> rhs, width)
        return (result,)

    @impl(asl.AsrBitsOp)
    def run_asr_bits(
        self, interpreter: Interpreter, op: asl.AsrBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(op.lhs.type, asl.BitVectorType)
        width = op.lhs.type.width.data
        lhs: int
        rhs: int
        (lhs, rhs) = args
        if lhs >= (1 << (width - 1)):
            lhs = lhs - (1 << width)
        result = to_unsigned(lhs >> rhs, width)
        return (result,)

    @impl(asl.ZeroExtendBitsOp)
    def run_zext_bits(
        self, interpreter: Interpreter, op: asl.ZeroExtendBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        (lhs, _) = args
        result = lhs
        return (result,)

    @impl(asl.SignExtendBitsOp)
    def run_sext_bits(
        self, interpreter: Interpreter, op: asl.SignExtendBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(op.lhs.type, asl.BitVectorType)
        width = op.lhs.type.width.data
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert rhs >= width
        result = lhs
        if result >= (1 << (width - 1)):
            result |= (1 << rhs) - (1 << width)
        return (result,)

    @impl(asl.AppendBitsOp)
    def run_append_bits(
        self, interpreter: Interpreter, op: asl.AppendBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(op.rhs.type, asl.BitVectorType)
        width = op.rhs.type.width.data
        lhs: int
        rhs: int
        (lhs, rhs) = args
        result = (lhs << width) | rhs
        return (result,)

    @impl(asl.ReplicateBitsOp)
    def run_replicate_bits(
        self, interpreter: Interpreter, op: asl.ReplicateBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(op.lhs.type, asl.BitVectorType)
        width = op.lhs.type.width.data
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert rhs >= 0
        result = 0
        for i in range(rhs):
            result |= lhs << (i * width)
        return (result,)

    @impl(asl.ZerosBitsOp)
    def run_zeros_bits(
        self, interpreter: Interpreter, op: asl.ZerosBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        arg: int
        [arg] = args
        assert arg >= 0
        result = 0
        return (result,)

    @impl(asl.OnesBitsOp)
    def run_ones_bits(
        self, interpreter: Interpreter, op: asl.OnesBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        arg: int
        [arg] = args
        assert arg >= 0
        result = (1 << arg) - 1
        return (result,)

    @impl(asl.MkMaskBitsOp)
    def run_mk_mask_bits(
        self, interpreter: Interpreter, op: asl.MkMaskBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert lhs <= rhs
        result = (1 << lhs) - 1
        return (result,)

    @impl(asl.NotBitsOp)
    def run_not_bits(
        self, interpreter: Interpreter, op: asl.NotBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(op.arg.type, asl.BitVectorType)
        width = op.arg.type.width.data
        arg: int
        [arg] = args
        result = (1 << width) - 1 - arg
        return (result,)

    @impl(asl.CvtBitsUIntOp)
    def run_cvt_bits_uint_bits(
        self, interpreter: Interpreter, op: asl.CvtBitsUIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(op.arg.type, asl.BitVectorType)
        width = op.arg.type.width.data
        arg: int
        [arg] = args
        result = to_unsigned(arg, width)
        return (result,)

    @impl(asl.CvtBitsSIntOp)
    def run_cvt_bits_sint_bits(
        self, interpreter: Interpreter, op: asl.CvtBitsSIntOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(op.arg.type, asl.BitVectorType)
        width = op.arg.type.width.data
        arg: int
        [arg] = args
        result = to_signed(arg, width)
        return (result,)

    @impl(asl.EqBitsOp)
    def run_eq_bits(
        self, interpreter: Interpreter, op: asl.EqBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        return (lhs == rhs,)

    @impl(asl.NeBitsOp)
    def run_ne_bits(
        self, interpreter: Interpreter, op: asl.NeBitsOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs: int
        rhs: int
        (lhs, rhs) = args
        return (lhs != rhs,)

    @impl(asl.GetSliceOp)
    def run_get_slice(
        self, interpreter: Interpreter, op: asl.GetSliceOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        x: int
        index: int
        width: int
        (x, index, width) = args
        result = (x >> index) & ((1 << width) - 1)
        return (result,)

    @impl(asl.SetSliceOp)
    def run_set_slice(
        self, interpreter: Interpreter, op: asl.SetSliceOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        x: int
        index: int
        width: int
        y: int
        (x, index, width, y) = args
        mask = ((1 << width) - 1) << index
        result = (x & ~mask) | (y << index)
        return (result,)

    @impl(asl.ConstantIntOp)
    def run_constant_int(
        self, interpreter: Interpreter, op: asl.ConstantIntOp, args: PythonValues
    ) -> PythonValues:
        value = op.value
        return (value.data,)

    @impl(asl.ConstantBitVectorOp)
    def run_constant_bits(
        self, interpreter: Interpreter, op: asl.ConstantBitVectorOp, args: PythonValues
    ) -> PythonValues:
        value = op.value
        return (value.value.data,)

    @impl(asl.ConstantStringOp)
    def run_constant_string(
        self, interpreter: Interpreter, op: asl.ConstantStringOp, args: PythonValues
    ) -> PythonValues:
        value = op.value
        return (value.data,)

    @impl(asl.PrintBitsHexOp)
    def asl_print_bits_hex(
        self, interpreter: Interpreter, op: asl.PrintBitsHexOp, args: PythonValues
    ) -> PythonValues:
        assert isinstance(op.arg.type, asl.BitVectorType)
        width = op.arg.type.width.data
        arg: int
        (arg,) = args
        output = f"{width:d}'x{arg:x}"
        interpreter.print(output)
        return ()

    @impl(asl.PrintSIntNHexOp)
    def asl_print_sintN_hex(
        self, interpreter: Interpreter, op: asl.PrintSIntNHexOp, args: PythonValues
    ) -> PythonValues:
        assert isa(op.arg.type, builtin.IntegerType)
        width = op.arg.type.width.data
        arg: int
        (arg,) = args
        if arg >= 0:
            output = f"i{width:d}'x{arg:x}"
        else:
            output = f"-i{width:d}'x{-arg:x}"
        interpreter.print(output)
        return ()

    @impl(asl.PrintSIntNDecOp)
    def asl_print_sintN_dec(
        self, interpreter: Interpreter, op: asl.PrintSIntNDecOp, args: PythonValues
    ) -> PythonValues:
        assert isa(op.arg.type, builtin.IntegerType)
        width = op.arg.type.width.data
        arg: int
        (arg,) = args
        if arg >= 0:
            output = f"i{width:d}'d{arg:d}"
        else:
            output = f"-i{width:d}'d{-arg:d}"
        interpreter.print(output)
        return ()

    # region built-in function implementations

    @impl_external("print_int_dec.0")
    def asl_print_int_dec(
        self, interpreter: Interpreter, op: Operation, args: PythonValues
    ) -> PythonValues:
        arg: int
        (arg,) = args
        interpreter.print(arg)
        return ()

    @impl_external("print_int_hex.0")
    def asl_print_int_hex(
        self, interpreter: Interpreter, op: Operation, args: PythonValues
    ) -> PythonValues:
        arg: int
        (arg,) = args
        interpreter.print(hex(arg))
        return ()

    @impl_external("print_char.0")
    def asl_print_char(
        self, interpreter: Interpreter, op: Operation, args: PythonValues
    ) -> PythonValues:
        arg: int
        (arg,) = args
        interpreter.print(chr(arg))
        return ()

    @impl_external("print_str.0")
    def asl_print_string(
        self, interpreter: Interpreter, op: Operation, args: PythonValues
    ) -> PythonValues:
        arg: str
        (arg,) = args
        interpreter.print(arg)
        return ()

    # endregion
