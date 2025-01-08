// RUN: asl-opt %s | asl-opt %s | filecheck %s

builtin.module {
    asl.func @external_op() -> ()
// CHECK:    asl.func @external_op() -> ()

    asl.func @non_external_op() -> () {
        asl.return
    }

// CHECK-NEXT:    asl.func @non_external_op() {
// CHECK-NEXT:      asl.return
// CHECK-NEXT:    }

    asl.func @op_with_args(%x : !test.type<"type1">, %y : !test.type<"type2">) -> () {
        asl.return
    }

// CHECK-NEXT:    asl.func @op_with_args(%x : !test.type<"type1">, %y : !test.type<"type2">) {
// CHECK-NEXT:      asl.return
// CHECK-NEXT:    }

    asl.func @op_with_return(%x : !test.type<"type1">) -> (!test.type<"type2">) {
        %y = asl.call @op_with_return(%x) : (!test.type<"type1">) -> !test.type<"type2">
        asl.return %y : !test.type<"type2">
    }

// CHECK-NEXT:    asl.func @op_with_return(%x : !test.type<"type1">) -> !test.type<"type2"> {
// CHECK-NEXT:      %y = asl.call @op_with_return(%x) : (!test.type<"type1">) -> !test.type<"type2">
// CHECK-NEXT:      asl.return %y : !test.type<"type2">
// CHECK-NEXT:    }

}
