// RUN: asl-opt %s --verify-diagnostics | filecheck %s

builtin.module {
    asl.func @wrong_return(%x : !test.type<"type1">) -> (!test.type<"type2">) {
        asl.return %x : !test.type<"type1">
    }
    // CHECK: Expected return types to match the function output types
}
