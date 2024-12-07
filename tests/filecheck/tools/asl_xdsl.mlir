// RUN: asl-opt %s | filecheck %s

builtin.module {}

// CHECK:      builtin.module {
// CHECK-NEXT: }
