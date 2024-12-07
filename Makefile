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
