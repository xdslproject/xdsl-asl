// RUN: asl-opt --t exec %s | filecheck %s

func main() => integer
begin
    return 42;
end;

// CHECK: result: 42
