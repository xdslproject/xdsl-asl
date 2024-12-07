from __future__ import annotations

import re
from typing import NamedTuple

from xdsl.parser import BaseParser

IDENTIFIER = re.compile("[A-z_][A-z_\\d]*")


class Ty(NamedTuple):
    ty: T_Exception

    @staticmethod
    def parse_ast(parser: BaseParser) -> Ty:
        parser.parse_characters("annot")
        parser.parse_punctuation("(")
        id = parser.expect(parser.parse_optional_identifier, "Ty")
        if id != T_Exception.__name__:
            raise NotImplementedError(f"Unimplemented type {id}")
        ty = T_Exception.parse_ast_tail(parser)
        parser.parse_punctuation(")")
        return Ty(ty)


class Field(NamedTuple):
    id: str
    ty: Ty

    @staticmethod
    def parse_ast(parser: BaseParser) -> Field:
        raise NotImplementedError


class T_Exception(NamedTuple):
    fields: tuple[Field, ...]

    @staticmethod
    def parse_ast_tail(parser: BaseParser) -> T_Exception:
        # parser.parse_characters(" ")
        fields = parser.parse_comma_separated_list(
            BaseParser.Delimiter.SQUARE, lambda: Field.parse_ast(parser)
        )
        return T_Exception(tuple(fields))

    @staticmethod
    def parse_ast(parser: BaseParser) -> T_Exception:
        parser.parse_characters(T_Exception.__name__)
        return T_Exception.parse_ast_tail(parser)
