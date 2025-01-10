// RUN: asl-opt %s | asl-opt %s | filecheck %s

builtin.module {
    %bool1, %bool2 = "test.op"() : () -> (!asl.bool, !asl.bool)
// CHECK:    %bool1, %bool2 = "test.op"() : () -> (!asl.bool, !asl.bool)

    %not_bool = asl.not_bool %bool1
    %and_bool = asl.and_bool %bool1, %bool2
    %or_bool = asl.or_bool %bool1, %bool2
    %eq_bool = asl.eq_bool %bool1, %bool2
    %ne_bool = asl.ne_bool %bool1, %bool2
    %implies_bool = asl.implies_bool %bool1, %bool2
    %equiv_bool = asl.equiv_bool %bool1, %bool2

// CHECK-NEXT:    %not_bool = asl.not_bool %bool1
// CHECK-NEXT:    %and_bool = asl.and_bool %bool1, %bool2
// CHECK-NEXT:    %or_bool = asl.or_bool %bool1, %bool2
// CHECK-NEXT:    %eq_bool = asl.eq_bool %bool1, %bool2
// CHECK-NEXT:    %ne_bool = asl.ne_bool %bool1, %bool2
// CHECK-NEXT:    %implies_bool = asl.implies_bool %bool1, %bool2
// CHECK-NEXT:    %equiv_bool = asl.equiv_bool %bool1, %bool2


    %int1, %int2 = "test.op"() : () -> (!asl.int, !asl.int)
// CHECK-NEXT:    %int1, %int2 = "test.op"() : () -> (!asl.int, !asl.int)

    %neg_int = asl.neg_int %int1 : !asl.int -> !asl.int
    %add_int = asl.add_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %sub_int = asl.sub_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %mul_int = asl.mul_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %exp_int = asl.exp_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %shiftleft_int = asl.shiftleft_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %shiftright_int = asl.shiftright_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %div_int = asl.div_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %fdiv_int = asl.fdiv_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int
    %frem_int = asl.frem_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.int

// CHECK-NEXT:    %neg_int = asl.neg_int %int1
// CHECK-NEXT:    %add_int = asl.add_int %int1, %int2
// CHECK-NEXT:    %sub_int = asl.sub_int %int1, %int2
// CHECK-NEXT:    %mul_int = asl.mul_int %int1, %int2
// CHECK-NEXT:    %exp_int = asl.exp_int %int1, %int2
// CHECK-NEXT:    %shiftleft_int = asl.shiftleft_int %int1, %int2
// CHECK-NEXT:    %shiftright_int = asl.shiftright_int %int1, %int2
// CHECK-NEXT:    %div_int = asl.div_int %int1, %int2
// CHECK-NEXT:    %fdiv_int = asl.fdiv_int %int1, %int2
// CHECK-NEXT:    %frem_int = asl.frem_int %int1, %int2

    %eq_int = asl.eq_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.bool
    %ne_int = asl.ne_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.bool
    %le_int = asl.le_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.bool
    %lt_int = asl.lt_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.bool
    %ge_int = asl.ge_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.bool
    %gt_int = asl.gt_int %int1, %int2 : (!asl.int, !asl.int) -> !asl.bool

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
