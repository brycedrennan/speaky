SHELL := /bin/bash

init: install-uv ## Setup a dev environment for local development.
	uv sync --all-extras --dev
	uv tool install ruff@0.0.287
	@echo -e "\nEnvironment setup! ✨ 🍰 ✨ 🐍 \n"
	@echo -e "The following commands are available to run in the Makefile\n"
	@make -s help

af: autoformat  ## Alias for `autoformat`
autoformat:  ## Run the autoformatter.
	@uv run -- ruff check . --fix-only
	@uv run -- ruff format .

afu: autoformat-unsafe  ## Alias for `autoformat-unsafe`.
autoformat-unsafe:  ## Run the autoformatter without --fix-only.
	@uvx ruff@0.0.287 check --select RUF001,RUF002,RUF003 --fix --isolated .
	@uv run -- ruff check . --fix-only --unsafe-fixes
	@uv run -- ruff format .

lint:  ## Run the code linter.
	@uv run -- ruff check .
	@echo -e "✅ No linting errors - well done! ✨ 🍰 ✨"

typecheck: ## Run the type checker.
	@uv run -- ty check
	@echo -e "✅ No type errors - well done! ✨ 🍰 ✨"

test:  ## Run the tests.
	@uv run -- pytest
	@echo -e "✅ The tests pass! ✨ 🍰 ✨"

check: af lint typecheck test ## Run all checks.

checku: afu lint typecheck test ## Run all checks with unsafe autoformatting.

publish:  ## Build and upload the package to PyPI.
	@echo -e "\n\033[0;34m📦 Building and uploading to PyPI...\033[0m\n"
	@rm -rf dist
	@uv run --frozen -- python -m build
	@uv run --frozen -- twine upload dist/* --repository pypi -u __token__
	@echo "\n\033[0;32m✅ 📦 Package published successfully to pypi! ✨ 🍰 ✨\033[0m\n"

install-uv:  # Install uv if not already installed
	@if ! uv --help >/dev/null 2>&1; then \
		echo "Installing uv..."; \
		wget -qO- https://astral.sh/uv/install.sh | sh; \
		echo -e "\033[0;32m ✔️  uv installed \033[0m"; \
	fi

strip-voice-metadata: ## Remove metadata from built-in voice MP3s.
	@for f in speaky/voices/*.mp3; do \
		ffmpeg -y -i "$$f" -map_metadata -1 -c copy -f mp3 "$$f.tmp" && mv "$$f.tmp" "$$f"; \
	done

.PHONY: sync
sync:
	@# ── guard required env-vars ──────────────────────────────────────────────
	@: $${LOCAL_PROJ_ROOT?"LOCAL_PROJ_ROOT is not set"}
	@: $${REMOTE_PROJ_ROOT?"REMOTE_PROJ_ROOT is not set"}

	@# ── ensure we're somewhere underneath $$LOCAL_PROJ_ROOT ──────────────────
	@case "$$PWD/" in "$${LOCAL_PROJ_ROOT}"*) ;; *) \
		echo "Error: run make inside $${LOCAL_PROJ_ROOT}  (current $$PWD)"; \
		exit 1 ;; \
	esac

	@# ── derive pieces we need ────────────────────────────────────────────────
	@local="$$PWD"; \
	rel="$${local#$${LOCAL_PROJ_ROOT}}"; \
	remote_prefix="$${REMOTE_PROJ_ROOT#ssh://}"; \
	host="$${remote_prefix%%/*}"; \
	base="$${remote_prefix#*/}"; \
	remote_dir="$${base%/}/$$rel"; \
	echo "Checking remote base: $$host:$$base"; \
	ssh "$$host" "[ -d \"$$base\" ]" || { \
		echo \"Error: remote base '$$base' does not exist on $$host\"; exit 1; }; \
	echo "Ensuring remote path exists: $$host:$$remote_dir"; \
	ssh "$$host" "mkdir -p \"$$remote_dir\"" || { \
		echo 'mkdir -p on remote failed'; exit 1; }; \
	remote_spec="$${REMOTE_PROJ_ROOT}$$rel"; \
	echo "Syncing  $$local  ↔  $$remote_spec"; \
	unison "$$local" "$$remote_spec" -auto -batch -perms 0 -prefer newer -repeat watch \
		-ignore 'Name .venv' \
		-ignore 'Name .ruff_cache' \
		-ignore 'Name .mypy_cache' \
		-ignore 'Name .pytest_cache' \
		-ignore 'Name *.egg-info' \
		-ignore 'Name *.pyc' \
		-ignore 'Name .DS_Store'

help: ## Show this help message.
	@## https://gist.github.com/prwhite/8168133#gistcomment-1716694
	@echo -e "$$(grep -hE '^\S+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##\s*/:/' -e 's/^\(.\+\):\(.*\)/\\x1b[36m\1\\x1b[m:\2/' | column -c2 -t -s :)" | sort