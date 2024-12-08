import pytest

from asl_xdsl.frontend.ast import (
    AST,
    D_TypeDecl,
    Decl,
    T_Exception,
    Ty,
)
from asl_xdsl.frontend.parser import ASTParser


@pytest.mark.parametrize(
    "identifier",
    [
        ("_"),
        ("a"),
        ("_a1"),
    ],
)
def test_parse_identifier(identifier: str):
    parser = ASTParser(identifier)
    assert parser.parse_identifier() == identifier
    assert parser.peek() is None


def test_parse_exception():
    parser = ASTParser("T_Exception []")
    assert parser.parse_exception() == T_Exception(())
    assert parser.peek() is None


def test_parse_type():
    parser = ASTParser("annot (T_Exception [])")
    assert parser.parse_type() == Ty(T_Exception(()))
    assert parser.peek() is None


def test_type_decl():
    parser = ASTParser('D_TypeDecl ("except", annot (T_Exception []), None)')
    assert parser.parse_type_decl() == D_TypeDecl("except", Ty(T_Exception(())), None)
    assert parser.peek() is None


def test_parse_decl():
    parser = ASTParser('D_TypeDecl ("except", annot (T_Exception []), None)')
    assert parser.parse_decl() == Decl(D_TypeDecl("except", Ty(T_Exception(())), None))
    assert parser.peek() is None


def test_parse_ast():
    parser = ASTParser('[D_TypeDecl ("except", annot (T_Exception []), None)]')

    assert parser.parse_ast() == AST(
        (Decl(D_TypeDecl("except", Ty(T_Exception(())), None)),)
    )
    assert parser.peek() is None
