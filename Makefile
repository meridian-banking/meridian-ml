.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install with dev dependencies
	pip install -e ".[dev]"

test: ## Run the test suite
	pytest -q

lint: ## Ruff lint + format check (same as CI)
	ruff check src tests && ruff format --check src tests

fmt: ## Auto-fix lint and formatting
	ruff check --fix src tests && ruff format src tests
