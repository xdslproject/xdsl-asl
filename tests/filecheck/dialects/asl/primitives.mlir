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
    %zero = asl.constant_int 4 {attr_dict}
    %four = asl.constant_int 4 {attr_dict}
    %sf = asl.constant_int 64 {attr_dict}

    %add_bits = asl.add_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> !asl.bits<32>
    %add_bits_int = asl.add_bits_int %bits1, %int1 : (!asl.bits<32>, !asl.int) -> !asl.bits<32>
    %sub_bits = asl.sub_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> !asl.bits<32>
    %sub_bits_int = asl.sub_bits_int %bits1, %int1 : (!asl.bits<32>, !asl.int) -> !asl.bits<32>
    %mul_bits = asl.mul_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> !asl.bits<32>
    %lsl_bits = asl.lsl_bits %bits1, %int1 : (!asl.bits<32>, !asl.int) -> !asl.bits<32>
    %lsr_bits = asl.lsr_bits %bits1, %int1 : (!asl.bits<32>, !asl.int) -> !asl.bits<32>
    %asr_bits = asl.asr_bits %bits1, %int1 : (!asl.bits<32>, !asl.int) -> !asl.bits<32>
    %zext_bits = asl.zero_extend_bits %bits1, %sf : (!asl.bits<32>, !asl.int) -> !asl.bits<64>
    %sext_bits = asl.sign_extend_bits %bits1, %sf : (!asl.bits<32>, !asl.int) -> !asl.bits<64>
    %append_bits = asl.append_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> !asl.bits<64>
    %replicate_bits = asl.replicate_bits %bits1, %four : (!asl.bits<32>, !asl.int) -> !asl.bits<128>
    %zeros_bits = asl.zeros_bits %int1 : !asl.int -> !asl.bits<32>
    %ones_bits = asl.ones_bits %int1 : !asl.int -> !asl.bits<32>
    %mask_bits = asl.mk_mask %int1, %sf : (!asl.int, !asl.int) -> !asl.bits<64>
    %not_bits = asl.not_bits %bits1 : !asl.bits<32> -> !asl.bits<32>
    %sint_bits = asl.cvt_bits_sint %bits1 : !asl.bits<32> -> !asl.int
    %uint_bits = asl.cvt_bits_uint %bits1 : !asl.bits<32> -> !asl.int
    %and_bits = asl.and_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> !asl.bits<32>
    %or_bits = asl.or_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> !asl.bits<32>
    %xor_bits = asl.xor_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> !asl.bits<32>
    %eq_bits = asl.eq_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> i1
    %ne_bits = asl.ne_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> i1
    asl.print_bits_hex %bits1 : !asl.bits<32> -> ()
    %slice = asl.get_slice %bits1, %four, %four : (!asl.bits<32>, !asl.int, !asl.int) -> !asl.bits<4>
    %set_slice = asl.set_slice %bits1, %zero, %four, %slice : (!asl.bits<32>, !asl.int, !asl.int, !asl.bits<4>) -> !asl.bits<32>

// CHECK:         %add_bits = asl.add_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> !asl.bits<32>
// CHECK-NEXT:    %add_bits_int = asl.add_bits_int %bits1, %int1 : (!asl.bits<32>, !asl.int) -> !asl.bits<32>
// CHECK-NEXT:    %sub_bits = asl.sub_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> !asl.bits<32>
// CHECK-NEXT:    %sub_bits_int = asl.sub_bits_int %bits1, %int1 : (!asl.bits<32>, !asl.int) -> !asl.bits<32>
// CHECK-NEXT:    %mul_bits = asl.mul_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> !asl.bits<32>
// CHECK-NEXT:    %lsl_bits = asl.lsl_bits %bits1, %int1 : (!asl.bits<32>, !asl.int) -> !asl.bits<32>
// CHECK-NEXT:    %lsr_bits = asl.lsr_bits %bits1, %int1 : (!asl.bits<32>, !asl.int) -> !asl.bits<32>
// CHECK-NEXT:    %asr_bits = asl.asr_bits %bits1, %int1 : (!asl.bits<32>, !asl.int) -> !asl.bits<32>
// CHECK-NEXT:    %zext_bits = asl.zero_extend_bits %bits1, %sf : (!asl.bits<32>, !asl.int) -> !asl.bits<64>
// CHECK-NEXT:    %sext_bits = asl.sign_extend_bits %bits1, %sf : (!asl.bits<32>, !asl.int) -> !asl.bits<64>
// CHECK-NEXT:    %append_bits = asl.append_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> !asl.bits<64>
// CHECK-NEXT:    %replicate_bits = asl.replicate_bits %bits1, %four : (!asl.bits<32>, !asl.int) -> !asl.bits<128>
// CHECK-NEXT:    %zeros_bits = asl.zeros_bits %int1 : !asl.int -> !asl.bits<32>
// CHECK-NEXT:    %ones_bits = asl.ones_bits %int1 : !asl.int -> !asl.bits<32>
// CHECK-NEXT:    %mask_bits = asl.mk_mask %int1, %sf : (!asl.int, !asl.int) -> !asl.bits<64>
// CHECK-NEXT:    %not_bits = asl.not_bits %bits1 : !asl.bits<32> -> !asl.bits<32>
// CHECK-NEXT:    %sint_bits = asl.cvt_bits_sint %bits1 : !asl.bits<32> -> !asl.int
// CHECK-NEXT:    %uint_bits = asl.cvt_bits_uint %bits1 : !asl.bits<32> -> !asl.int
// CHECK-NEXT:    %and_bits = asl.and_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> !asl.bits<32>
// CHECK-NEXT:    %or_bits = asl.or_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> !asl.bits<32>
// CHECK-NEXT:    %xor_bits = asl.xor_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> !asl.bits<32>
// CHECK-NEXT:    %eq_bits = asl.eq_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> i1
// CHECK-NEXT:    %ne_bits = asl.ne_bits %bits1, %bits2 : (!asl.bits<32>, !asl.bits<32>) -> i1
// CHECK-NEXT:    asl.print_bits_hex %bits1 : !asl.bits<32> -> ()
// CHECK-NEXT:    %slice = asl.get_slice %bits1, %four, %four : (!asl.bits<32>, !asl.int, !asl.int) -> !asl.bits<4>

    %sint1, %sint2 = "test.op"() : () -> (i8, i8)
    asl.print_sintN_hex %sint1 : i8
    asl.print_sintN_dec %sint1 : i8
// CHECK:         asl.print_sintN_hex %sint1 : i8 -> ()
// CHECK-NEXT:    asl.print_sintN_dec %sint1 : i8 -> ()

    asl.global "G" : !asl.bits<32>
    asl.global "A" : !asl.array<16 x !asl.bits<32>>

    %gref = asl.address_of @G : !asl.ref<!asl.bits<32>>
    %aref = asl.address_of @A : !asl.ref<!asl.array<16 x !asl.bits<32>>>
    %eref = asl.array_ref %aref[%int1] : !asl.ref<!asl.array<16 x !asl.bits<32>>>
    asl.store %bits1 to %gref : !asl.bits<32>
    %load = asl.load from %eref : !asl.bits<32>
// CHECK:         asl.global "G" : !asl.bits<32>
// CHECK-NEXT:    asl.global "A" : !asl.array<16x!asl.bits<32>>
// CHECK-NEXT:    %gref = asl.address_of(@G) : !asl.ref<!asl.bits<32>>
// CHECK-NEXT:    %aref = asl.address_of(@A) : !asl.ref<!asl.array<16x!asl.bits<32>>>
// CHECK-NEXT:    %eref = asl.array_ref(%aref, %int1) : !asl.ref<!asl.array<16x!asl.bits<32>>>
// CHECK-NEXT:    asl.store(%gref) = %bits1 : !asl.bits<32>
// CHECK-NEXT:    %load = asl.load(%eref) : !asl.bits<32>

}
