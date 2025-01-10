// RUN: asl-opt %s --target exec | filecheck %s

// CHECK: 3

asl.func @print_int_dec.0(!asl.int)
asl.func @print_char.0(!asl.int)

asl.func @main.0() -> !asl.int {
    %0 = asl.constant_int 1 {attr_dict}
    %1 = asl.constant_int 2 {attr_dict}
    %2 = asl.call @Test.0(%0, %1) : (!asl.int, !asl.int) -> !asl.int
    asl.call @print_int_dec.0 (%2) : (!asl.int) -> ()

    asl.call @println.0() : () -> ()

    %3 = asl.constant_int 0 {attr_dict}
    asl.return %3 : !asl.int
}

asl.func @Test.0(%x : !asl.int, %y : !asl.int) -> !asl.int {
    %0 = asl.add_int %x, %y : (!asl.int, !asl.int) -> !asl.int
    asl.return %0 : !asl.int
}

asl.func @println.0() -> () {
    %0 = asl.constant_int 10 {attr_dict}
    asl.call @print_char.0(%0) : (!asl.int) -> ()
    asl.return
}
