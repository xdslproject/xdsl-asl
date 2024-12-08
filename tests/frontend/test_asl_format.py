from collections.abc import Callable
from io import StringIO

import pytest

from asl_xdsl.frontend.ast import (
    AST,
    D_TypeDecl,
    Decl,
    Field,
    T_Exception,
    Ty,
    base_parser,
)
from asl_xdsl.frontend.printer import Printer


def get_asl(func: Callable[[Printer], None]) -> str:
    io = StringIO()
    func(Printer(io))
    return io.getvalue()


def test_print_exception():
    assert get_asl(T_Exception(()).print_asl) == "exception"

    with pytest.raises(NotImplementedError):
        get_asl(T_Exception((Field("field", Ty(T_Exception(()))),)).print_asl)


def test_parse_exception():
    parser = base_parser("")
    assert T_Exception.parse_asl_tail(parser) == T_Exception(())


def test_print_field():
    with pytest.raises(NotImplementedError):
        get_asl(Field("field", Ty(T_Exception(()))).print_asl)


def test_print_type():
    assert get_asl(Ty(T_Exception(())).print_asl) == "exception"


def test_parse_type():
    parser = base_parser("exception")
    assert Ty.parse_asl(parser) == Ty(T_Exception(()))


def test_print_type_decl():
    assert (
        get_asl(D_TypeDecl("except", Ty(T_Exception(())), None).print_asl)
        == "type except of exception;\n"
    )


def test_parse_type_decl():
    parser = base_parser(" except of exception;\n")
    assert D_TypeDecl.parse_asl_tail(parser) == D_TypeDecl(
        "except", Ty(T_Exception(())), None
    )


def test_print_decl():
    assert (
        get_asl(Decl(D_TypeDecl("except", Ty(T_Exception(())), None)).print_asl)
        == "type except of exception;\n"
    )


def test_parse_decl():
    parser = base_parser("type except of exception;\n")
    assert Decl.parse_optional_asl(parser) == Decl(
        D_TypeDecl("except", Ty(T_Exception(())), None)
    )


def test_print_ast():
    assert (
        get_asl(AST((Decl(D_TypeDecl("except", Ty(T_Exception(())), None)),)).print_asl)
        == "type except of exception;\n"
    )


def test_parse_asl():
    parser = base_parser("type except of exception;\n")
    assert AST.parse_asl(parser) == AST(
        (Decl(D_TypeDecl("except", Ty(T_Exception(())), None)),)
    )
