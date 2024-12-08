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


def test_print_field():
    with pytest.raises(NotImplementedError):
        get_asl(Field("field", Ty(T_Exception(()))).print_asl)


def test_print_type():
    assert get_asl(Ty(T_Exception(())).print_asl) == "exception"


def test_print_type_decl():
    assert (
        get_asl(D_TypeDecl("except", Ty(T_Exception(())), None).print_asl)
        == "type except of exception;\n"
    )


def test_print_decl():
    assert (
        get_asl(Decl(D_TypeDecl("except", Ty(T_Exception(())), None)).print_asl)
        == "type except of exception;\n"
    )


def test_print_ast():
    assert (
        get_asl(AST((Decl(D_TypeDecl("except", Ty(T_Exception(())), None)),)).print_asl)
        == "type except of exception;\n"
    )
