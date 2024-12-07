import pytest

from asl_xdsl.frontend.ast import (
    D_TypeDecl,
    Decl,
    T_Exception,
    Ty,
    base_parser,
    parse_serialized_ast,
)


@pytest.mark.parametrize(
    "identifier",
    [
        ("_"),
        ("a"),
        ("_a1"),
    ],
)
def test_parse_identifier(identifier: str):
    parser = base_parser(identifier)
    assert parser.parse_identifier() == identifier


def test_parse_exception():
    parser = base_parser("T_Exception []")
    assert T_Exception.parse_ast(parser) == T_Exception(())


def test_parse_type():
    parser = base_parser("annot (T_Exception [])")
    assert Ty.parse_ast(parser) == Ty(T_Exception(()))


def test_type_decl():
    parser = base_parser('D_TypeDecl ("except", annot (T_Exception []), None)')
    assert D_TypeDecl.parse_ast(parser) == D_TypeDecl(
        "except", Ty(T_Exception(())), None
    )


def test_parse_decl():
    parser = base_parser('D_TypeDecl ("except", annot (T_Exception []), None)')
    assert Decl.parse_ast(parser) == Decl(
        D_TypeDecl("except", Ty(T_Exception(())), None)
    )


def test_parse_ast():
    assert parse_serialized_ast(
        '[D_TypeDecl ("except", annot (T_Exception []), None)]'
    ) == (Decl(D_TypeDecl("except", Ty(T_Exception(())), None)),)
