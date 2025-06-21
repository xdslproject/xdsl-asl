from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp, NoneAttr
from xdsl.ir import Attribute
from xdsl.passes import ModulePass

from asl_xdsl.analysis.integer_range import IntegerRange, IntegerRangeAnalysis


def bound_to_attr(bound: int | None) -> Attribute:
    """Convert a bound (int or None) to an attribute."""
    if bound is None:
        return NoneAttr()
    return IntAttr(bound)


def range_to_attr(integer_range: IntegerRange) -> Attribute:
    """Convert an IntegerRange to an attribute."""
    return ArrayAttr(
        [
            bound_to_attr(integer_range.lower_bound),
            bound_to_attr(integer_range.upper_bound),
        ]
    )


@dataclass(frozen=True)
class TestIntegerRangeAnalysis(ModulePass):
    """
    Test the integer range analysis pass by adding integer range analysis information
    to the module using attributes.
    """

    name = "test-integer-range-analysis"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        analysis = IntegerRangeAnalysis()
        analysis.compute_single_block_region_analysis(op.body)

        for sub_op in op.walk():
            if not sub_op.results:
                continue
            result_ranges = [analysis.get_range(result) for result in sub_op.results]
            result_ranges_attrs = ArrayAttr(
                [range_to_attr(result) for result in result_ranges]
            )
            sub_op.attributes["__integer_ranges"] = result_ranges_attrs
