from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.dialects.builtin import FunctionType, ModuleOp
from xdsl.ir import Block, Region, SSAValue, TypeAttribute
from xdsl.rewriter import InsertPoint
from xdsl.utils.scoped_dict import ScopedDict

from asl_xdsl.dialects.asl import (
    ConstantIntOp,
    FuncOp,
    IntegerType,
    ReturnOp,
)
from asl_xdsl.frontend import ast


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
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))

    def ir_gen_module(self, module_ast: ast.AST) -> ModuleOp:
        """
        Convert the AST for an ASL source file to an MLIR Module operation.
        """

        for decl in module_ast.decls:
            self.ir_gen_decl(decl)

        self.module.verify()

        return self.module

    def ir_gen_decl(self, decl: ast.Decl) -> None:
        """Generate IR for a declaration."""
        # Currently declarations are not used in the IR
        match decl:
            case ast.D_TypeDecl():
                self.ir_gen_type_decl(decl)
            case ast.D_Func():
                self.ir_gen_func_decl(decl)

    def ir_gen_type_decl(self, decl: ast.D_TypeDecl) -> None:
        """Generate IR for a type declaration."""
        raise NotImplementedError()

    @staticmethod
    def get_type(t: ast.TypeDesc) -> TypeAttribute:
        match t:
            case ast.T_Int():
                return IRGen.get_int_type(t)
            case ast.T_Exception():
                return IRGen.get_exception_type(t)
            case ast.T_Record():
                return IRGen.get_record_type(t)

    @staticmethod
    def get_int_type(t: ast.T_Int) -> TypeAttribute:
        match t.kind:
            case ast.UnConstrained():
                return IntegerType()
            case ast.WellConstrained() | ast.PendingConstrained() | ast.Parameterized():
                raise NotImplementedError()

    @staticmethod
    def get_exception_type(t: ast.T_Exception) -> TypeAttribute:
        raise NotImplementedError()

    @staticmethod
    def get_record_type(t: ast.T_Record) -> TypeAttribute:
        raise NotImplementedError()

    @staticmethod
    def get_function_type(args: None, return_type: ast.Ty | None) -> FunctionType:
        # For now we only support functions with no arguments
        if args is not None:
            raise NotImplementedError()

        # If no return type, function returns void (empty tuple)
        if return_type is None:
            return_types = ()
        else:
            return_types = (IRGen.get_type(return_type.val),)

        # Otherwise create function type with single return value
        return FunctionType.from_lists([], return_types)

    def ir_gen_func_decl(self, decl: ast.D_Func) -> None:
        """Generate IR for a function declaration."""

        # Create entry block and switch builder to it
        entry_block = Block()
        self.builder = Builder(InsertPoint.at_start(entry_block))

        # Create new scope for function
        outer_symbol_table = self.symbol_table
        self.symbol_table = ScopedDict(outer_symbol_table)

        # Generate IR for function body
        self.ir_gen_subprogram_body(decl.body)

        # Restore outer scope
        self.symbol_table = outer_symbol_table

        # Return builder to module level
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))

        # Create function operation
        function_type = IRGen.get_function_type(decl.args, decl.return_type)
        func = FuncOp(decl.name, function_type, Region(entry_block))
        self.builder.insert(func)

    def ir_gen_subprogram_body(self, body: ast.SubprogramBody) -> None:
        """Generate IR for a subprogram body."""
        if isinstance(body, ast.SB_Primitive):
            raise NotImplementedError()

        # Get the statement from the ASL subprogram body
        stmt = body.stmt.val

        # Generate IR for the statement
        match stmt:
            case ast.SPass():
                # For pass statement, no IR needed
                pass
            case ast.SReturn():
                # For return statement, create return operation
                expr_val = self.ir_gen_optional_expr(stmt.expr)
                self.builder.insert(ReturnOp(expr_val))

    def ir_gen_optional_expr(self, expr: ast.Expr | None) -> SSAValue | None:
        """Generate IR for an optional expression."""
        if expr is not None:
            return self.ir_gen_expr(expr)

    def ir_gen_expr(self, expr: ast.Expr) -> SSAValue:
        """Generate IR for an expression."""
        match expr.val:
            case ast.E_Literal():
                return self.ir_gen_literal(expr.val.literal)

    def ir_gen_literal(self, lit: ast.Literal) -> SSAValue:
        """Generate IR for a literal value."""
        match lit:
            case ast.L_Int():
                return self.builder.insert(ConstantIntOp(lit.val)).res
            # case ast.L_Bool():
            #     return self.builder.insert(ConstantBoolOp(lit.val)).res
