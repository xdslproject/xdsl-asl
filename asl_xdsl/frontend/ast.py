import re

from asl_xdsl.frontend.parser import Parser

IDENTIFIER = re.compile("[A-z_][A-z_\\d]*")


def parse_optional_identifier(parser: Parser) -> str | None:
    return parser.parse_optional_pattern(IDENTIFIER)


def parse_identifier(parser: Parser) -> str:
    return parser.expect("identifier", parse_optional_identifier)
