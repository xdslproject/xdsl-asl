MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

# make tasks run all commands in a single shell
.ONESHELL:

# set up all precommit hooks
.PHONY: precommit-install
precommit-install:
	pre-commit install

# run all precommit hooks and apply them
.PHONY: precommit
precommit:
	pre-commit run --all

# run all tests
.PHONY: tests
tests: pyright
	@echo All tests done.

# run pyright on all files in the current git commit
.PHONY: pyright
pyright:
	pyright $(shell git diff --staged --name-only  -- '*.py')
