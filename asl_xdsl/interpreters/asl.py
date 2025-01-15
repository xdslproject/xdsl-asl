from typing import Any

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

    @impl(asl.ConstantIntOp)
    def run_constant_int(
        self, interpreter: Interpreter, op: asl.ConstantIntOp, args: PythonValues
    ) -> PythonValues:
        value = op.value
        return (value.data,)

    @impl(asl.ConstantStringOp)
    def run_constant_string(
        self, interpreter: Interpreter, op: asl.ConstantStringOp, args: PythonValues
    ) -> PythonValues:
        value = op.value
        return (value.data,)

    # region built-in function implementations

    @impl_external("print_bits_hex.0")
    def asl_print_bits_hex(
        self, interpreter: Interpreter, op: Operation, args: PythonValues
    ) -> PythonValues:
        arg: int
        (arg,) = args
        interpreter.print(hex(arg))
        return ()

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
