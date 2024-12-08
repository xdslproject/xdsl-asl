// RUN: asl-frontend %s --print | asl-frontend --print | filecheck %s

type except of exception;

// CHECK: type except of exception;
