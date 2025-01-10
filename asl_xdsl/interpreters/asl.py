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

    @impl(asl.ConstantIntOp)
    def run_constant(
        self, interpreter: Interpreter, op: asl.ConstantIntOp, args: PythonValues
    ) -> PythonValues:
        value = op.value
        return (value.data,)

    # region built-in function implementations

    @impl_external("asl_print_int_dec")
    def asl_print_int_dec(
        self, interpreter: Interpreter, op: Operation, args: PythonValues
    ) -> PythonValues:
        arg: int
        (arg,) = args
        interpreter.print(arg)
        return ()

    @impl_external("asl_print_char")
    def asl_print_char(
        self, interpreter: Interpreter, op: Operation, args: PythonValues
    ) -> PythonValues:
        arg: int
        (arg,) = args
        interpreter.print(chr(arg))
        return ()

    # endregion
