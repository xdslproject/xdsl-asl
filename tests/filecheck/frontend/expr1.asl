// RUN: asl-frontend %s --print | asl-frontend --print | filecheck %s

type point of record;
type except of exception;

func main()
begin
end;


// CHECK:      type point of record;
// CHECK-NEXT: type except of exception;
// CHECK-NEXT: func main()
// CHECK-NEXT: begin
// CHECK-NEXT: end;
