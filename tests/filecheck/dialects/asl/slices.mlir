// RUN: asl-opt %s | asl-opt %s | filecheck %s

builtin.module {
    %bits, %index = "test.op"() : () -> (!asl.bits<42>, !asl.int)
    asl.slice_single %bits[%index] : !asl.bits<42>
    // CHECK: asl.slice_single %bits[%index] : !asl.bits<42>
}
