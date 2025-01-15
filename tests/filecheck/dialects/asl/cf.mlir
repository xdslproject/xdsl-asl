// RUN: asl-opt %s | asl-opt %s | filecheck %s

builtin.module {
    asl.func @print_str.0(%x : !asl.string) -> ()
    %c = asl.constant_bool true {attr_dict}
    %0 = asl.bool_to_i1 %c : !asl.bool -> i1
    scf.if %0 {
        %1 = asl.constant_string "TRUE" {attr_dict}
        asl.call @print_str.0(%1) : (!asl.string) -> ()
    } else {
        %2 = asl.constant_string "FALSE" {attr_dict}
        asl.call @print_str.0(%2) : (!asl.string) -> ()
    }

// CHECK:        %c = asl.constant_bool true {attr_dict}
// CHECK-NEXT:   %0 = asl.bool_to_i1 %c : !asl.bool -> i1
// CHECK-NEXT:   scf.if %0 {
// CHECK-NEXT:     %1 = asl.constant_string "TRUE" {attr_dict}
// CHECK-NEXT:     asl.call @print_str.0(%1) : (!asl.string) -> ()
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %2 = asl.constant_string "FALSE" {attr_dict}
// CHECK-NEXT:     asl.call @print_str.0(%2) : (!asl.string) -> ()
// CHECK-NEXT:   }

}
