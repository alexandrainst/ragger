# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: help

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Create .env file if it does not already exist
ifeq (,$(wildcard .env))
  $(shell touch .env)
endif

# Create poetry env file if it does not already exist
ifeq (,$(wildcard ${HOME}/.poetry/env))
  $(shell mkdir ${HOME}/.poetry)
  $(shell touch ${HOME}/.poetry/env)
endif

# Includes environment variables from the .env file
include .env

# Set gRPC environment variables, which prevents some errors with the `grpcio` package
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

# Ensure that `pipx` and `poetry` will be able to run, since `pip` and `brew` put these
# in the following folders on Unix systems
export PATH := ${HOME}/.local/bin:/opt/homebrew/bin:$(PATH)

# Prevent DBusErrorResponse during `poetry install`
#(see https://stackoverflow.com/a/75098703 for more information)
export PYTHON_KEYRING_BACKEND := keyring.backends.null.Keyring

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing the 'ragger' project..."
	@$(MAKE) --quiet install-brew
	@$(MAKE) --quiet install-pipx
	@$(MAKE) --quiet install-poetry
	@$(MAKE) --quiet install-dependencies
	@$(MAKE) --quiet setup-environment-variables
	@$(MAKE) --quiet setup-git
	@echo "Installed the 'ragger' project. If you want to use pre-commit hooks, run 'make install-pre-commit'."

install-brew:
	@if [ $$(uname) = "Darwin" ] && [ "$(shell which brew)" = "" ]; then \
		/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
		echo "Installed Homebrew."; \
	fi

install-pipx:
	@if [ "$(shell which pipx)" = "" ]; then \
		uname=$$(uname); \
			case $${uname} in \
				(*Darwin*) installCmd='brew install pipx'; ;; \
				(*CYGWIN*) installCmd='py -3 -m pip install --upgrade --user pipx'; ;; \
				(*) installCmd='python3 -m pip install --upgrade --user pipx'; ;; \
			esac; \
			$${installCmd}; \
		pipx ensurepath --force; \
		echo "Installed pipx."; \
	fi

install-poetry:
	@if [ ! "$(shell poetry --version)" = "Poetry (version 1.8.2)" ]; then \
        python3 -m pip uninstall -y poetry poetry-core poetry-plugin-export; \
        pipx install --force poetry==1.8.2; \
        echo "Installed Poetry."; \
    fi

install-dependencies:
	@poetry env use python3.11 && poetry install --extras default

install-pre-commit:  ## Install pre-commit hooks
	@poetry run pre-commit install

lint:  ## Lint the code
	@poetry run ruff check . --fix

format:  ## Format the code
	@poetry run ruff format .

type-check:  ## Run type checking
	@poetry run mypy . --install-types --non-interactive --ignore-missing-imports --show-error-codes --check-untyped-defs

setup-environment-variables:
	@poetry run python src/scripts/fix_dot_env_file.py

setup-environment-variables-non-interactive:
	@poetry run python src/scripts/fix_dot_env_file.py --non-interactive

setup-git:
	@git config --global init.defaultBranch main
	@git init
	@git config --local user.name ${GIT_NAME}
	@git config --local user.email ${GIT_EMAIL}

test:  ## Run tests
	@poetry run pytest && poetry run readme-cov

publish-major:  ## Publish the major version
	@poetry run python -m src.scripts.versioning --major
	@echo "Published major version!"

publish-minor:  ## Publish the minor version
	@poetry run python -m src.scripts.versioning --minor
	@echo "Published minor version!"

publish-patch:  ## Publish the patch version
	@poetry run python -m src.scripts.versioning --patch
	@echo "Published patch version!"
