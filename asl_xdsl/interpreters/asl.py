from typing import Any

from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    TerminatorValue,
    impl,
    impl_callable,
    impl_terminator,
    register_impls,
)

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

    @impl(asl.ConstantIntOp)
    def run_constant(
        self, interpreter: Interpreter, op: asl.ConstantIntOp, args: PythonValues
    ) -> PythonValues:
        value = op.value
        return (value.data,)
