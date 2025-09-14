// RUN: asl-opt %s | asl-opt %s | filecheck %s

builtin.module {
    %t = dt.forall !dt.int {
      ^b(%n : !dt.int):
        %c = dt.constr_cmp "lt" %n %n
        %m = dt.type_add %n %n
        %tr = dt.range %n to %m
        %tret = dt.constr %c %tr
        dt.yield %tret
    }
}

// CHECK:        %t = dt.forall !dt.int {
// CHECK-NEXT:   ^b(%n : !dt.int):
// CHECK-NEXT:     %c = dt.constr_cmp "lt" %n %n
// CHECK-NEXT:     %m = dt.type_add %n, %n
// CHECK-NEXT:     %tr = dt.range %n to %m
// CHECK-NEXT:     %tret = dt.constr %c %tr
// CHECK-NEXT:     dt.yield %tret
// CHECK-NEXT:   }
