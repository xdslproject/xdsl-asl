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
// CHECK-NEXT:  }
