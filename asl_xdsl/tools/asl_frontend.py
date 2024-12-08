"""
Helper utility to test ASL parsing and printing.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", nargs="?", help="Input file (stdin if not provided)")
    parser.add_argument("--print", action="store_true", help="Print the parsed AST")
    args = parser.parse_args()

    if args.input:
        with open(args.input) as f:
            input_str = f.read()
    else:
        import sys

        input_str = sys.stdin.read()

    from asl_xdsl.frontend.parser import ASLParser
    from asl_xdsl.frontend.printer import Printer

    ast = ASLParser(input_str).parse_ast()

    if args.print:
        printer = Printer()
        ast.print_asl(printer)


if __name__ == "__main__":
    main()
