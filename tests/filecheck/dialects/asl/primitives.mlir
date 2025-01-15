// RUN: asl-opt %s | asl-opt %s | filecheck %s

builtin.module {
    %int1, %int2 = "test.op"() : () -> (!asl.int, !asl.int)
// CHECK:         %int1, %int2 = "test.op"() : () -> (!asl.int, !asl.int)

    %neg_int = asl.neg_int %int1 : !asl.int -> !asl.int
    %add_int = asl.add_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %sub_int = asl.sub_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %mul_int = asl.mul_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %exp_int = asl.exp_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %shiftleft_int = asl.shl_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %shiftright_int = asl.shr_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %exact_div_int = asl.exact_div_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %fdiv_int = asl.fdiv_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %frem_int = asl.frem_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %zdiv_int = asl.zdiv_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %zrem_int = asl.zrem_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %align_int = asl.align_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %mod_pow2_int = asl.mod_pow2_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %pow2_int = asl.pow2_int %int1 : !asl.int -> !asl.int {attr_dict}
    %is_pow2_int = asl.is_pow2_int %int1 : !asl.int -> i1 {attr_dict}

// CHECK-NEXT:    %neg_int = asl.neg_int %int1
// CHECK-NEXT:    %add_int = asl.add_int %int1, %int2
// CHECK-NEXT:    %sub_int = asl.sub_int %int1, %int2
// CHECK-NEXT:    %mul_int = asl.mul_int %int1, %int2
// CHECK-NEXT:    %exp_int = asl.exp_int %int1, %int2
// CHECK-NEXT:    %shiftleft_int = asl.shl_int %int1, %int2
// CHECK-NEXT:    %shiftright_int = asl.shr_int %int1, %int2
// CHECK-NEXT:    %exact_div_int = asl.exact_div_int %int1, %int2
// CHECK-NEXT:    %fdiv_int = asl.fdiv_int %int1, %int2
// CHECK-NEXT:    %frem_int = asl.frem_int %int1, %int2
// CHECK-NEXT:    %zdiv_int = asl.zdiv_int %int1, %int2
// CHECK-NEXT:    %zrem_int = asl.zrem_int %int1, %int2
// CHECK-NEXT:    %align_int = asl.align_int %int1, %int2
// CHECK-NEXT:    %mod_pow2_int = asl.mod_pow2_int %int1, %int2
// CHECK-NEXT:    %pow2_int = asl.pow2_int %int1
// CHECK-NEXT:    %is_pow2_int = asl.is_pow2_int %int1

    %eq_int = asl.eq_int %int1, %int2 : (!asl.int, !asl.int) -> i1
    %ne_int = asl.ne_int %int1, %int2 : (!asl.int, !asl.int) -> i1
    %le_int = asl.le_int %int1, %int2 : (!asl.int, !asl.int) -> i1
    %lt_int = asl.lt_int %int1, %int2 : (!asl.int, !asl.int) -> i1
    %ge_int = asl.ge_int %int1, %int2 : (!asl.int, !asl.int) -> i1
    %gt_int = asl.gt_int %int1, %int2 : (!asl.int, !asl.int) -> i1

// CHECK-NEXT:    %eq_int = asl.eq_int %int1, %int2
// CHECK-NEXT:    %ne_int = asl.ne_int %int1, %int2
// CHECK-NEXT:    %le_int = asl.le_int %int1, %int2
// CHECK-NEXT:    %lt_int = asl.lt_int %int1, %int2
// CHECK-NEXT:    %ge_int = asl.ge_int %int1, %int2
// CHECK-NEXT:    %gt_int = asl.gt_int %int1, %int2

    %bits1, %bits2 = "test.op"() : () -> (!asl.bits<32>, !asl.bits<32>)

    %add_bits = asl.add_bits %bits1, %bits2 : !asl.bits<32>
    %add_bits_int = asl.add_bits_int %bits1, %int1 : !asl.bits<32>
    %sub_bits = asl.sub_bits %bits1, %bits2 : !asl.bits<32>
    %sub_bits_int = asl.sub_bits_int %bits1, %int1 : !asl.bits<32>
    %not_bits = asl.not_bits %bits1 : !asl.bits<32>
    %and_bits = asl.and_bits %bits1, %bits2 : !asl.bits<32>
    %or_bits = asl.or_bits %bits1, %bits2 : !asl.bits<32>
    %xor_bits = asl.xor_bits %bits1, %bits2 : !asl.bits<32>
    %eq_bits = asl.eq_bits %bits1, %bits2 : !asl.bits<32>
    %ne_bits = asl.ne_bits %bits1, %bits2 : !asl.bits<32>
}
