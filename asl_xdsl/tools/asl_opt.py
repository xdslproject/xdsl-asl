from xdsl.xdsl_opt_main import xDSLOptMain

from asl_xdsl.dialects.asl import ASLDialect


class ASLOptMain(xDSLOptMain):
    def register_all_dialects(self):
        super().register_all_dialects()
        self.ctx.load_dialect(ASLDialect)

    def register_all_passes(self):
        return super().register_all_passes()

    def register_all_targets(self):
        return super().register_all_targets()


def main():
    ASLOptMain().run()


if __name__ == "__main__":
    main()
