// RUN: asl-opt %s | asl-opt %s | filecheck %s

builtin.module {
    "test.op"() {bool_type = !asl.bool} : () -> ()
    "test.op"() {bool_true = #asl.bool_attr<true>, bool_false = #asl.bool_attr<false>} : () -> ()

// CHECK:         "test.op"() {bool_type = !asl.bool} : () -> ()
// CHECK-NEXT:    "test.op"() {bool_true = #asl.bool_attrtrue, bool_false = #asl.bool_attrfalse} : () -> ()

    "test.op"() {int_type = !asl.int} : () -> ()
    "test.op"() {constraint_int = !asl.int<42>} : () -> ()
    "test.op"() {constraint_int = !asl.int<10:24>} : () -> ()
    "test.op"() {constraint_int = !asl.int<8, 16, 32, 64>} : () -> ()
    "test.op"() {constraint_int = !asl.int<0:32, 64>} : () -> ()

// CHECK-NEXT:    "test.op"() {int_type = !asl.int} : () -> ()
// CHECK-NEXT:    "test.op"() {constraint_int = !asl.int<42>} : () -> ()
// CHECK-NEXT:    "test.op"() {constraint_int = !asl.int<10:24>} : () -> ()
// CHECK-NEXT:    "test.op"() {constraint_int = !asl.int<8, 16, 32, 64>} : () -> ()
// CHECK-NEXT:    "test.op"() {constraint_int = !asl.int<0:32, 64>} : () -> ()

    "test.op"() {bits_type = !asl.bits<32>} : () -> ()
    "test.op"() {bits_attr = #asl.bits_attr<42 : !asl.bits<32>>} : () -> ()

// CHECK-NEXT:    "test.op"() {bits_type = !asl.bits<32>} : () -> ()
// CHECK-NEXT:    "test.op"() {bits_attr = #asl.bits_attr<42 : 32>} : () -> ()
}
