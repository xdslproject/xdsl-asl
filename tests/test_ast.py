import pytest

from asl_xdsl.frontend.ast import T_Exception, parse_identifier
from asl_xdsl.frontend.parser import Parser


@pytest.mark.parametrize(
    "identifier",
    [
        ("_"),
        ("a"),
        ("_a1"),
    ],
)
def test_parse_identifier(identifier: str):
    parser = Parser(identifier)
    assert parse_identifier(parser) == identifier


def test_parse_exception():
    parser = Parser("T_Exception []")
    assert T_Exception.parse_ast(parser) == T_Exception(())
