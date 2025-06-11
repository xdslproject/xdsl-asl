// RUN: asl-opt %s -p=test-integer-range-analysis | filecheck %s

builtin.module {

    // CHECK: %unknown = "test.op"() {__integer_ranges = {{[}}[#none, #none]]} : () -> !asl.int
    %unknown = "test.op"() : () -> !asl.int
    
    // CHECK-NEXT: %cst16 = asl.constant_int 16 {__integer_ranges = {{[}}[#builtin.int<16>, #builtin.int<16>]]}
    %cst16 = asl.constant_int 16
    
    // CHECK-NEXT: %res = asl.mod_pow2_int %unknown, %cst16 : (!asl.int, !asl.int) -> !asl.int {__integer_ranges = {{[}}[#builtin.int<0>, #builtin.int<65535>]]}
    %res = asl.mod_pow2_int %unknown, %cst16 : (!asl.int, !asl.int) -> !asl.int
}
