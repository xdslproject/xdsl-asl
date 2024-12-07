from __future__ import annotations

import re
from typing import NamedTuple

from asl_xdsl.frontend.parser import Parser

IDENTIFIER = re.compile("[A-z_][A-z_\\d]*")


def parse_optional_identifier(parser: Parser) -> str | None:
    return parser.parse_optional_pattern(IDENTIFIER)


def parse_identifier(parser: Parser) -> str:
    return parser.expect("identifier", parse_optional_identifier)


class Ty(NamedTuple):
    ty: T_Exception


class Field(NamedTuple):
    id: str
    ty: Ty

    @staticmethod
    def parse_ast(parser: Parser) -> Field:
        raise NotImplementedError


class T_Exception(NamedTuple):
    fields: tuple[Field, ...]

    @staticmethod
    def parse_ast(parser: Parser) -> T_Exception:
        parser.parse_chars(T_Exception.__name__)
        parser.parse_chars(" [")
        fields = parser.parse_list(
            Field.parse_ast,
            lambda parser: parser.parse_chars(", "),
            lambda parser: parser.parse_optional_chars("]"),
        )
        return T_Exception(tuple(fields))
