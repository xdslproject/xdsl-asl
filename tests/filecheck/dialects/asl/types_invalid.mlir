// RUN: asl-opt --verify-diagnostics --split-input-file %s | filecheck %s

"test.op"() {array_type = !asl.array<!asl.bits<32>>} : () -> ()

// CHECK:    asl.array shape must not be empty

// -----

"test.op"() {array_type = !asl.array<2x?x!asl.bits<32>>} : () -> ()

// CHECK:    asl.array array dimensions must have non-negative size
