from typing import IO

from xdsl.parser import Input
from xdsl.xdsl_opt_main import xDSLOptMain

from asl_xdsl.dialects.asl import ASLDialect
from asl_xdsl.frontend.parser import ASLParser


class ASLOptMain(xDSLOptMain):
    def register_all_dialects(self):
        super().register_all_dialects()
        self.ctx.load_dialect(ASLDialect)

    def register_all_passes(self):
        return super().register_all_passes()

    def register_all_targets(self):
        return super().register_all_targets()

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
