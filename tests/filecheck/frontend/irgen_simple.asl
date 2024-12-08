// RUN: asl-opt --split-input-file %s | filecheck %s

// Empty

// CHECK:       builtin.module {
// CHECK-NEXT:  }

// -----

func main() => integer
begin
    return 42;
end;

// CHECK:       builtin.module {
// CHECK-NEXT:      asl.func @main() -> !asl.int {
// CHECK-NEXT:          %0 = asl.constant_int 42
// CHECK-NEXT:          return %0 : !asl.int
// CHECK-NEXT:      }
// CHECK-NEXT:  }
