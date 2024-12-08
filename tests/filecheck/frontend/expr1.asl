// RUN: asl-frontend %s --print | asl-frontend --print | filecheck %s

type point of record;
type except of exception;

func proc()
begin
end;

func main() => integer
begin
    return 42;
end;


// CHECK:      type point of record;
// CHECK-NEXT: type except of exception;
// CHECK-NEXT: func proc()
// CHECK-NEXT: begin
// CHECK-NEXT: end;
// CHECK-NEXT: func main() => integer
// CHECK-NEXT: begin
// CHECK-NEXT:     return 42;
// CHECK-NEXT: end;
