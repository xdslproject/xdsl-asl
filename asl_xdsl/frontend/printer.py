from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar


@dataclass(eq=False, repr=False)
class Printer:
    stream: Any | None = field(default=None)

    def print_string(self, text: str) -> None:
        print(text, end="", file=self.stream)

    @contextmanager
    def in_braces(self):
        self.print_string("{")
        yield
        self.print_string("}")

    @contextmanager
    def in_parens(self):
        self.print_string("(")
        yield
        self.print_string(") ")

    T = TypeVar("T")

    def print_list(
        self, elems: Iterable[T], print_fn: Callable[[T], Any], delimiter: str = ", "
    ) -> None:
        for i, elem in enumerate(elems):
            if i:
                self.print_string(delimiter)
            print_fn(elem)
