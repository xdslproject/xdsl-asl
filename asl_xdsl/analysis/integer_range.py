from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from xdsl.ir import Operation, Region, SSAValue
from xdsl.traits import OpTrait


@dataclass(frozen=True)
class IntegerRange:
    """
    The range of an integer value, defined by its lower and upper bounds (inclusive).
    If either bound is None, it means that the bound is not known.
    """

    lower_bound: int | None
    upper_bound: int | None

    def is_empty(self) -> bool:
        """Check if the range is empty."""
        if self.lower_bound is None or self.upper_bound is None:
            return False
        return self.lower_bound > self.upper_bound

    def get_as_constant(self) -> int | None:
        """
        Get the range as a constant integer value if it is a single value range.
        If the range is not a single value, returns None.
        """
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound == self.upper_bound:
                return self.lower_bound
        return None

    @staticmethod
    def top() -> IntegerRange:
        """
        Get the top range, which is the range that covers all possible integer values.
        This is used when no specific range is known for a value.
        """
        return IntegerRange(None, None)

    @staticmethod
    def bottom() -> IntegerRange:
        """
        Get the bottom range, which is the empty range.
        This is used when a value is known to be outside of any possible integer range.
        """
        return IntegerRange(1, -1)

    def __contains__(self, value: int) -> bool:
        """Check if a value is within the range."""
        if self.lower_bound is not None:
            if value < self.lower_bound:
                return False
        if self.upper_bound is not None:
            if value > self.upper_bound:
                return False
        return True

    def __or__(self, other: IntegerRange) -> IntegerRange:
        """Combine two integer ranges into one that covers both ranges."""
        if self.lower_bound is None:
            lower_bound = other.lower_bound
        elif other.lower_bound is None:
            lower_bound = self.lower_bound
        else:
            lower_bound = min(self.lower_bound, other.lower_bound)

        if self.upper_bound is None:
            upper_bound = other.upper_bound
        elif other.upper_bound is None:
            upper_bound = self.upper_bound
        else:
            upper_bound = max(self.upper_bound, other.upper_bound)

        return IntegerRange(lower_bound, upper_bound)

    def __and__(self, other: IntegerRange) -> IntegerRange:
        """Intersect two integer ranges into one that covers the intersection."""
        if self.lower_bound is None:
            lower_bound = other.lower_bound
        elif other.lower_bound is None:
            lower_bound = self.lower_bound
        else:
            lower_bound = max(self.lower_bound, other.lower_bound)

        if self.upper_bound is None:
            upper_bound = other.upper_bound
        elif other.upper_bound is None:
            upper_bound = self.upper_bound
        else:
            upper_bound = min(self.upper_bound, other.upper_bound)

        return IntegerRange(lower_bound, upper_bound)


@dataclass
class IntegerRangeAnalysis:
    """An analysis that contains the integer ranges of integer values."""

    ranges: dict[SSAValue, IntegerRange] = field(
        default_factory=dict[SSAValue, IntegerRange]
    )

    def get_range(self, value: SSAValue) -> IntegerRange:
        """
        Get the integer range of a value.
        If the value is not an integer type, returns None.
        """
        return self.ranges.get(value, IntegerRange.top())

    def set_range(self, value: SSAValue, integer_range: IntegerRange) -> None:
        """
        Set the integer range of a value.
        If the value is not an integer type, this will raise an error.
        """
        # No need to set the top range, as it is the default
        if integer_range == IntegerRange.top():
            return
        self.ranges[value] = integer_range

    def compute_operation_analysis(self, op: Operation) -> None:
        """
        Compute the integer ranges of the operation's results.
        This function will call the IntegerRangeTrait.compute_analysis method
        if the operation has the IntegerRangeTrait trait.
        """
        trait = op.get_trait(IntegerRangeTrait)
        if trait is None:
            return

        trait.compute_analysis(op, self)

    def compute_single_block_region_analysis(self, region: Region) -> None:
        """
        Compute the integer ranges of all the SSA values in an SSACFG region
        with a single block.
        """
        assert len(region.blocks) == 1, "Region must have a single block"
        block = region.block

        for op in block.ops:
            self.compute_operation_analysis(op)


class IntegerRangeTrait(OpTrait, ABC):
    """
    A trait that indicates that an operation can be used for an integer range analysis.
    In practice, this means that we can compute the lower and upper bounds of the
    operation's integer results based on its operands and attributes ranges.
    """

    @staticmethod
    @abstractmethod
    def compute_analysis(op: Operation, analysis: IntegerRangeAnalysis):
        """
        Compute the integer ranges of the operation's results based on the
        integer ranges of its operands and attributes.
        """
        ...
