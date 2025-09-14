from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence
from typing import ClassVar, TypeAlias

from xdsl.dialects import builtin
from xdsl.dialects.utils import parse_func_op_like, print_func_op_like
from xdsl.ir import (
    Attribute,
    Block,
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
    VarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    param_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.irdl.attributes import eq, irdl_to_attr_constraint
from xdsl.irdl.constraints import AnyAttr, EqIntConstraint, RangeOf, SingleOf
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import (
    CallableOpInterface,
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    SymbolOpInterface,
    SymbolTable,
    SymbolUserOpInterface,
)


@irdl_attr_definition
class KindIntegerType(ParametrizedAttribute, TypeAttribute):
    """An arbitrary-precision integer type."""

    name = "dt.int"


@irdl_op_definition
class TypeAddOp(IRDLOperation):
    name = "dt.type_add"
    lhs = operand_def(KindIntegerType())
    rhs = operand_def(KindIntegerType())
    res = result_def(KindIntegerType())

    assembly_format = "$rhs `,` $lhs attr-dict"


@irdl_attr_definition
class KindConstraint(ParametrizedAttribute, TypeAttribute):
    name = "dt.constraint"

@irdl_op_definition
class ConstraintCmpOp(IRDLOperation):
    name = "dt.constr_cmp"
    cmp = prop_def(builtin.StringAttr) # FIXME: only comparison ops
    lhs = operand_def(KindIntegerType())
    rhs = operand_def(KindIntegerType())
    res = result_def(KindConstraint())
    assembly_format = "$cmp $lhs $rhs attr-dict"

@irdl_attr_definition
class KindTypeType(ParametrizedAttribute, TypeAttribute):
    name = "dt.type"

@irdl_op_definition
class RangeTypeOp(IRDLOperation):
    """A range type"""

    name = "dt.range"
    low = operand_def(KindIntegerType())
    high = operand_def(KindIntegerType())
    res = result_def(KindTypeType())

    assembly_format = "$low `to` $high attr-dict"


@irdl_op_definition
class ArrowTypeOp(IRDLOperation):
    """Function type"""

    name = "dt.arrow"
    src = operand_def(KindTypeType())
    tgt = operand_def(KindTypeType())
    res = result_def(KindTypeType())

    assembly_format = "$src `,` $tgt attr-dict"


@irdl_op_definition
class ConstraintTypeOp(IRDLOperation):
    name = "dt.constr"

    assembly_format = "$constraint $type attr-dict"

    constraint = operand_def(KindConstraint())
    type = operand_def(KindTypeType())
    res = result_def(KindTypeType())

@irdl_op_definition
class ForallTypeOp(IRDLOperation):
    "Universal quantification"

    name = "dt.forall"

    assembly_format = "$kind $type attr-dict"

    T: ClassVar[VarConstraint] = VarConstraint(
        "T", eq(KindIntegerType()) | eq(KindTypeType())
    )
    kind = prop_def(T)
    type = region_def("single_block", entry_args=SingleOf(T))
    res = result_def(KindTypeType())


@irdl_op_definition
class YieldTypeOp(IRDLOperation):
    name = "dt.yield"
    arg = operand_def(KindTypeType())
    assembly_format = "$arg attr-dict"

    traits = traits_def(HasParent(ForallTypeOp), IsTerminator())


DTDialect = Dialect(
    "dt",
    [
        TypeAddOp,
        RangeTypeOp,
        ArrowTypeOp,
        ForallTypeOp,
        ConstraintTypeOp,
        YieldTypeOp,
        ConstraintCmpOp,
    ],
    [
        KindTypeType,
        KindIntegerType,
        KindConstraint,
    ],
)
