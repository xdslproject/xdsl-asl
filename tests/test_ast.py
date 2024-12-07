import pytest
from xdsl.parser import BaseParser, ParserState
from xdsl.utils.lexer import Input, Lexer

from asl_xdsl.frontend.ast import T_Exception, Ty


def p(input: str) -> BaseParser:
    return BaseParser(ParserState(Lexer(Input(input, "<unknown>"))))


@pytest.mark.parametrize(
    "identifier",
    [
        ("_"),
        ("a"),
        ("_a1"),
    ],
)
def test_parse_identifier(identifier: str):
    parser = p(identifier)
    assert parser.parse_identifier() == identifier


def test_parse_exception():
    parser = p("T_Exception []")
    assert T_Exception.parse_ast(parser) == T_Exception(())


def test_parse_type():
    parser = p("annot (T_Exception [])")
    assert Ty.parse_ast(parser) == Ty(T_Exception(()))
