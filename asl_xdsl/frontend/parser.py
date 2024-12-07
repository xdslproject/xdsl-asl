import re
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from xdsl.parser import Input
from xdsl.utils.lexer import Position

T = TypeVar("T")


class ParseError(Exception):
    position: Position
    message: str

    def __init__(self, position: Position, message: str) -> None:
        self.position = position
        self.message = message
        super().__init__()

    def __str__(self):
        return f"ParseError at {self.position}: {self.message}"


class Parser:
    input: Input
    pos: int

    def __init__(self, input: Input | str, pos: int = 0):
        if isinstance(input, str):
            input = Input(input, "<unknown>")
        self.input = input
        self.pos = pos

    @property
    def remaining(self) -> str:
        return self.input.content[self.pos :]

    # region Base parsing functions

    def parse_optional_chars(self, chars: str):
        if self.input.content.startswith(chars, self.pos):
            self.pos += len(chars)
            return chars

    def parse_optional_pattern(self, pattern: re.Pattern[str]):
        if (match := pattern.match(self.input.content, self.pos)) is not None:
            self.pos = match.regs[0][1]
            return self.input.content[match.pos : self.pos]

    # endregion
    # region: Helpers

    def parse_many(self, element: Callable[["Parser"], T | None]) -> tuple[T, ...]:
        if (first := element(self)) is None:
            return ()
        res = [first]
        while (el := element(self)) is not None:
            res.append(el)
        return tuple(res)

    def parse_many_separated(
        self,
        element: Callable[["Parser"], T | None],
        separator: Callable[["Parser"], Any | None],
    ) -> tuple[T, ...]:
        if (first := element(self)) is None:
            return ()
        res = [first]
        while separator(self) is not None:
            el = self.expect("element", element)
            res.append(el)
        return tuple(res)

    def parse_optional_list(
        self,
        el: Callable[["Parser"], T | None],
        separator: Callable[["Parser"], Any | None],
        end: Callable[["Parser"], Any | None],
    ) -> list[T] | None:
        if end(self) is not None:
            return []
        if (first := el(self)) is None:
            return None
        res = [first]
        while end(self) is None:
            self.expect("separator", separator)
            element = self.expect("element", el)
            res.append(element)
        return res

    def expect(self, message: str, parse: Callable[["Parser"], T | None]) -> T:
        if (parsed := parse(self)) is None:
            raise ParseError(self.pos, message)
        return parsed

    def parse_one_of(
        self, parsers: Sequence[Callable[["Parser"], T | None]]
    ) -> T | None:
        for parser in parsers:
            if (parsed := parser(self)) is not None:
                return parsed

    # endregion
