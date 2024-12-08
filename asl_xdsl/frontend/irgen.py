from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import SSAValue
from xdsl.utils.scoped_dict import ScopedDict

# from asl_xdsl.dialects.asl import FuncOp
from asl_xdsl.frontend.ast import AST, Decl


class IRGenError(Exception):
    pass


@dataclass(init=False)
class IRGen:
    """
    Implementation of a simple MLIR emission from the Toy AST.

    This will emit operations that are specific to the Toy language, preserving
    the semantics of the language and (hopefully) allow to perform accurate
    analysis and transformation based on these high level semantics.
    """

    module: ModuleOp
    """A "module" matches a Toy source file: containing a list of functions."""

    builder: Builder
    """
    The builder is a helper class to create IR inside a function. The builder
    is stateful, in particular it keeps an "insertion point": this is where
    the next operations will be introduced."""

    symbol_table: ScopedDict[str, SSAValue] | None = None
    """
    The symbol table maps a variable name to a value in the current scope.
    Entering a function creates a new scope, and the function arguments are
    added to the mapping. When the processing of a function is terminated, the
    scope is destroyed and the mappings created in this scope are dropped."""

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder.at_end(self.module.body.blocks[0])

    def ir_gen_module(self, module_ast: AST) -> ModuleOp:
        """
        Convert the AST for an ASL source file to an MLIR Module operation.
        """

        # for decl in module_ast.decls:
        #     self.ir_gen_decl(decl)

        self.module.verify()

        return self.module

    def ir_gen_decl(self, decl: Decl) -> None:
        """Generate IR for a declaration."""
        # Currently declarations are not used in the IR
        pass
