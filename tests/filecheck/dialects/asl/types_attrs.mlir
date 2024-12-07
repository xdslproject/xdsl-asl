// RUN: asl-opt %s | asl-opt %s | filecheck %s

builtin.module {
    "test.op"() {bool_type = !asl.bool} : () -> ()
    "test.op"() {bool_true = #asl.bool_attr<true>, bool_false = #asl.bool_attr<false>} : () -> ()
    
    "test.op"() {int_type = !asl.int} : () -> ()

    "test.op"() {bits_type = !asl.bits<32>} : () -> ()
    "test.op"() {bits_attr = #asl.bits_attr<42 : !asl.bits<32>>} : () -> ()
}

// CHECK:         "test.op"() {"bool_type" = !asl.bool} : () -> ()
// CHECK-NEXT:    "test.op"() {"bool_true" = #asl.bool_attrtrue, "bool_false" = #asl.bool_attrfalse} : () -> ()
// CHECK-NEXT:    "test.op"() {"int_type" = !asl.int} : () -> ()
// CHECK-NEXT:    "test.op"() {"bits_type" = !asl.bits<32>} : () -> ()
// CHECK-NEXT:    "test.op"() {"bits_attr" = #asl.bits_attr<42 : 32>} : () -> ()
