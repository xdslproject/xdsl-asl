from typing import IO

from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import Input
from xdsl.traits import CallableOpInterface
from xdsl.xdsl_opt_main import xDSLOptMain

from asl_xdsl.dialects.asl import ASLDialect
from asl_xdsl.dialects.asl_dep import ASLDepDialect
from asl_xdsl.frontend.parser import ASLParser


class ASLOptMain(xDSLOptMain):
    def register_all_dialects(self):
        super().register_all_dialects()
        self.ctx.load_dialect(ASLDialect)
        self.ctx.load_dialect(ASLDepDialect)

    def register_all_passes(self):
        return super().register_all_passes()

    def register_all_targets(self):
        super().register_all_targets()

        def interpret_target(module: ModuleOp, output: IO[str]):
            from xdsl.interpreter import Interpreter
            from xdsl.interpreters import arith, scf

            from asl_xdsl.interpreters.asl import ASLFunctions

            interpreter = Interpreter(module, file=output)
            interpreter.register_implementations(ASLFunctions())
            interpreter.register_implementations(arith.ArithFunctions())
            interpreter.register_implementations(scf.ScfFunctions())
            op = interpreter.get_op_for_symbol("main.0")
            trait = op.get_trait(CallableOpInterface)
            assert trait is not None

            result = interpreter.call_op(op)
            if result:
                if len(result) == 1:
                    print(f"result: {result[0]}", file=output)
                else:
                    print("result: (", file=output)
                    print(",\n".join(f"    {res}" for res in result), file=output)
                    print(")", file=output)
            else:
                print("result: ()", file=output)

        self.available_targets["exec"] = interpret_target

    def register_all_frontends(self):
        super().register_all_frontends()
        from asl_xdsl.frontend.irgen import IRGen

        def asl_frontend(stream: IO[str]):
            source = stream.read()
            ast = ASLParser(Input(source, self.get_input_name())).parse_ast()
            irgen = IRGen().ir_gen_module(ast)
            return irgen

        self.available_frontends["asl"] = asl_frontend


def main():
    ASLOptMain().run()


if __name__ == "__main__":
    main()
