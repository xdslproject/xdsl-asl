// RUN: asl-frontend %s --print | asl-frontend --print | filecheck %s

type point of record;
type except of exception;

// CHECK:      type point of record;
// CHECK-NEXT: type except of exception;
