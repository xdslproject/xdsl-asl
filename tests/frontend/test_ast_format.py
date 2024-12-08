import pytest

from asl_xdsl.frontend.ast import (
    AST,
    Annotated,
    D_TypeDecl,
    Decl,
    T_Exception,
    T_Record,
    TypeDesc,
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


def test_parse_record():
    parser = ASTParser("T_Record []")
    assert parser.parse_record() == T_Record(())
    assert parser.peek() is None


@pytest.mark.parametrize(
    "serialized, deserialized",
    [
        ("T_Exception []", TypeDesc(T_Exception(()))),
        ("T_Record []", TypeDesc(T_Record(()))),
    ],
)
def test_parse_type_desc(serialized: str, deserialized: TypeDesc):
    parser = ASTParser(serialized)
    assert parser.parse_type_desc() == deserialized
    assert parser.peek() is None


def test_type_decl():
    parser = ASTParser('D_TypeDecl ("except", annot (T_Exception []), None)')
    assert parser.parse_type_decl() == D_TypeDecl(
        "except", Annotated(TypeDesc(T_Exception(()))), None
    )
    assert parser.peek() is None


def test_parse_decl():
    parser = ASTParser('D_TypeDecl ("except", annot (T_Exception []), None)')
    assert parser.parse_decl() == Decl(
        D_TypeDecl("except", Annotated(TypeDesc(T_Exception(()))), None)
    )
    assert parser.peek() is None


def test_parse_ast():
    parser = ASTParser('[D_TypeDecl ("except", annot (T_Exception []), None)]')

    assert parser.parse_ast() == AST(
        (Decl(D_TypeDecl("except", Annotated(TypeDesc(T_Exception(()))), None)),)
    )
    assert parser.peek() is None
