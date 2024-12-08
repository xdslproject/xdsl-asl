// RUN: asl-opt %s | filecheck %s

func main() => integer
begin
    return 42;
end;

// CHECK:       builtin.module {
// CHECK-NEXT:  }
