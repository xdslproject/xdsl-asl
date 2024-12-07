// RUN: asl-opt %s | asl-opt %s | filecheck %s

builtin.module {
    %bool1, %bool2 = "test.op"() : () -> (!asl.bool, !asl.bool)

    %not_res = asl.not_bool %bool1
    %and_res = asl.and_bool %bool1, %bool2
    %or_res = asl.or_bool %bool1, %bool2
    %eq_res = asl.eq_bool %bool1, %bool2
    %ne_res = asl.ne_bool %bool1, %bool2
    %implies_res = asl.implies_bool %bool1, %bool2
    %equiv_res = asl.equiv_bool %bool1, %bool2
}

// CHECK:         %bool1, %bool2 = "test.op"() : () -> (!asl.bool, !asl.bool)
// CHECK-NEXT:    %not_res = asl.not_bool %bool1
// CHECK-NEXT:    %and_res = asl.and_bool %bool1, %bool2
// CHECK-NEXT:    %or_res = asl.or_bool %bool1, %bool2
// CHECK-NEXT:    %eq_res = asl.eq_bool %bool1, %bool2
// CHECK-NEXT:    %ne_res = asl.ne_bool %bool1, %bool2
// CHECK-NEXT:    %implies_res = asl.implies_bool %bool1, %bool2
// CHECK-NEXT:    %equiv_res = asl.equiv_bool %bool1, %bool2
