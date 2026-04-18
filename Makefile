.DEFAULT_GOAL := help
DIST := dist
UV ?= uv

.PHONY: help install dev lint fmt test build clean smoke check wheel-reinstall

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Install targets use $(UV) (https://docs.astral.sh/uv/)."

install:  ## Editable install (core deps only)
	$(UV) pip install -e .

dev:  ## Editable install with dev + cli extras
	$(UV) pip install -e ".[dev,cli]"

lint:  ## Run ruff linter
	ruff check src/ tests/

fmt:  ## Auto-format with ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:  ## Run pytest
	pytest --tb=short -q

build: clean  ## Build sdist + wheel into dist/
	$(UV) build
	@echo ""
	@ls -lh $(DIST)/
	@echo ""
	@echo "Wheel: $$(ls $(DIST)/*.whl)"

wheel-reinstall: build  ## Install latest wheel via uv (uses .venv or active env; run: uv venv)
	@set -- $(DIST)/*.whl && $(UV) pip install --force-reinstall "$$1[server,local,openai,replicate]"
	@echo "Installed: `$(UV) pip show argus-lens | sed -n 's/^Version: //p'`"

clean:  ## Remove build artifacts
	rm -rf $(DIST) build src/*.egg-info src/argus_lens/*.egg-info

smoke: build  ## Build wheel, install in tmp venv via uv, smoke-test import
	$(eval TMPVENV := $(shell mktemp -d))
	$(UV) venv $(TMPVENV)
	$(UV) pip install --python $(TMPVENV)/bin/python $(DIST)/*.whl
	$(TMPVENV)/bin/python -c \
		"from argus_lens import ArgusLens, __version__; print(f'argus-lens {__version__} OK')"
	rm -rf $(TMPVENV)

check: lint test build  ## Full local CI: lint + test + build
	@echo ""
	@echo "All checks passed."
