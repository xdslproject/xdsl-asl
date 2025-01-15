// RUN: asl-opt %s | asl-opt %s | filecheck %s

builtin.module {
    %fourty_two = asl.constant_int 42 {attr_dict}

    %fourty_two_bits = asl.constant_bits 42 : !asl.bits<32> {attr_dict}

    %fourty_two_string = asl.constant_string "Forty Two" {attr_dict}
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %fourty_two = asl.constant_int 42 {attr_dict}
// CHECK-NEXT:    %fourty_two_bits = asl.constant_bits 42 : !asl.bits<32> {attr_dict}
// CHECK-NEXT:    %fourty_two_string = asl.constant_string "Forty Two" {attr_dict}
// CHECK-NEXT:  }
