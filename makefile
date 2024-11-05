# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: help

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Create .env file if it does not already exist
ifeq (,$(wildcard .env))
  $(shell touch .env)
endif

# Includes environment variables from the .env file
include .env

# Set gRPC environment variables, which prevents some errors with the `grpcio` package
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

# Set the shell to bash, enabling the use of `source` statements
SHELL := /bin/bash

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing the 'ragger' project..."
	@$(MAKE) --quiet install-rust
	@$(MAKE) --quiet install-uv
	@$(MAKE) --quiet install-dependencies
	@$(MAKE) --quiet setup-environment-variables
	@$(MAKE) --quiet setup-git
	@echo "Installed the 'ragger' project. If you want to use pre-commit hooks, run 'make install-pre-commit'."

install-rust:
	@if [ "$(shell which rustup)" = "" ]; then \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		source "${HOME}/.cargo/env"; \
		echo "Installed Rust."; \
	fi

install-uv:
	@if [ "$(shell which uv)" = "" ]; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		source "${HOME}/.cargo/env"; \
		echo "Installed uv."; \
	fi
	@uv self update

install-dependencies:
	@uv sync \
		--extra onprem_cpu \
		--extra postgres \
		--extra demo \
		--extra keyword_search

install-pre-commit:  ## Install pre-commit hooks
	@uv run pre-commit install

lint:  ## Lint the code
	@uv run ruff check . --fix

format:  ## Format the code
	@uv run ruff format .

type-check:  ## Run type checking
	@uv run mypy . \
		--install-types \
		--non-interactive \
		--ignore-missing-imports \
		--show-error-codes \
		--check-untyped-defs

check: lint format type-check  ## Run all checks

setup-environment-variables:
	@uv run python src/scripts/fix_dot_env_file.py

setup-environment-variables-non-interactive:
	@uv run python src/scripts/fix_dot_env_file.py --non-interactive

setup-git:
	@git config --global init.defaultBranch main
	@git init
	@git config --local user.name ${GIT_NAME}
	@git config --local user.email ${GIT_EMAIL}

test:  ## Run tests
	@uv run pytest && uv run readme-cov

publish-major:  ## Publish the major version
	@uv run python -m src.scripts.versioning --major \
		&& uv build \
		&& uv publish -u __token__ -p ${PYPI_TOKEN}
	@echo "Published major version!"

publish-minor:  ## Publish the minor version
	@uv run python -m src.scripts.versioning --minor \
		&& uv build \
		&& uv publish -u __token__ -p ${PYPI_TOKEN}
	@echo "Published minor version!"

publish-patch:  ## Publish the patch version
	@uv run python -m src.scripts.versioning --patch \
		&& uv build \
		&& uv publish -u __token__ -p ${PYPI_TOKEN}
	@echo "Published patch version!"
