// RUN: asl-opt %s | asl-opt %s | filecheck %s

builtin.module {
    %true = asl.constant_bool true {attr_dict}
    %false = asl.constant_bool false {attr_dict}

    %fourty_two = asl.constant_int 42 {attr_dict}
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %true = asl.constant_bool true {"attr_dict"}
// CHECK-NEXT:    %false = asl.constant_bool true {"attr_dict"}
// CHECK-NEXT:    %fourty_two = asl.constant_int 42 {"attr_dict"}
// CHECK-NEXT:  }
