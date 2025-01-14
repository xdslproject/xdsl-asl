// RUN: asl-opt %s | asl-opt %s | filecheck %s

builtin.module {
    %c32 = asl_dep.constant_int 32
    %c64 = asl_dep.constant_int 64

    %bv1 = asl_dep.constant_bits 1 : bits<%c32>
    %bv2 = asl_dep.add_bits %bv1, %bv1 : (bits<%c32>, bits<%c32>) -> bits<%c64>
}

// CHECK:      %c32 = asl_dep.constant_int 32
// CHECK-NEXT: %c64 = asl_dep.constant_int 64
// CHECK-NEXT: %bv1 = asl_dep.constant_bits 1 : bits<%c32>
// CHECK-NEXT: %bv2 = asl_dep.add_bits %bv1, %bv1 : (bits<%c32>, bits<%c32>) -> bits<%c64>
