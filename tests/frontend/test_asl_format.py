from collections.abc import Callable
from io import StringIO

import pytest

from asl_xdsl.frontend.ast import (
    AST,
    Annotated,
    D_TypeDecl,
    Decl,
    Field,
    T_Exception,
    TypeDesc,
)
from asl_xdsl.frontend.parser import ASLParser
from asl_xdsl.frontend.printer import Printer


def get_asl(func: Callable[[Printer], None]) -> str:
    io = StringIO()
    func(Printer(io))
    return io.getvalue()


def test_print_exception():
    assert get_asl(T_Exception(()).print_asl) == "exception"

    with pytest.raises(NotImplementedError):
        get_asl(
            T_Exception(
                (Field("field", Annotated(TypeDesc(T_Exception(())))),)
            ).print_asl
        )


def test_parse_exception():
    parser = ASLParser("exception")
    assert parser.parse_exception() == T_Exception(())


def test_print_field():
    with pytest.raises(NotImplementedError):
        get_asl(Field("field", Annotated(TypeDesc(T_Exception(())))).print_asl)


def test_print_type():
    assert get_asl(TypeDesc(T_Exception(())).print_asl) == "exception"


def test_parse_type():
    parser = ASLParser("exception")
    assert parser.parse_type_desc() == TypeDesc(T_Exception(()))


def test_print_type_decl():
    assert (
        get_asl(
            D_TypeDecl("except", Annotated(TypeDesc(T_Exception(()))), None).print_asl
        )
        == "type except of exception;\n"
    )


def test_parse_type_decl():
    parser = ASLParser("type except of exception;\n")
    assert parser.parse_type_decl() == D_TypeDecl(
        "except", Annotated(TypeDesc(T_Exception(()))), None
    )


def test_print_decl():
    assert (
        get_asl(
            Decl(
                D_TypeDecl("except", Annotated(TypeDesc(T_Exception(()))), None)
            ).print_asl
        )
        == "type except of exception;\n"
    )


def test_parse_decl():
    parser = ASLParser("type except of exception;\n")
    assert parser.parse_decl() == Decl(
        D_TypeDecl("except", Annotated(TypeDesc(T_Exception(()))), None)
    )


def test_print_ast():
    assert (
        get_asl(
            AST(
                (
                    Decl(
                        D_TypeDecl("except", Annotated(TypeDesc(T_Exception(()))), None)
                    ),
                )
            ).print_asl
        )
        == "type except of exception;\n"
    )


def test_parse_asl():
    parser = ASLParser("type except of exception;\n")
    assert parser.parse_ast() == AST(
        (Decl(D_TypeDecl("except", Annotated(TypeDesc(T_Exception(()))), None)),)
    )
