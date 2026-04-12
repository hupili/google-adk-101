# ==============================================================================
# Agent Starter Pack - Study Pack Makefile
# Centralized workflow for local development and remote cloud deployment.
# ==============================================================================

# Force uv to ignore global configurations that might point to restricted registries.
# This ensures we always use the public PyPI index defined in pyproject.toml.
export UV_NO_CONFIG=1

# Install/Update dependencies using uv.
install:
	@command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Installing uv..."; curl -LsSf https://astral.sh/uv/0.6.12/install.sh | sh; source $HOME/.local/bin/env; }
	uv sync --dev

sync:
	uv sync --dev

# Launch local dev playground (Streamlit-based UI).
playground:
	@echo "==============================================================================="
	@echo "| 🚀 Starting your agent playground...                                        |"
	@echo "==============================================================================="
	uv run adk web . --port 8501 --reload_agents

# Deploy the agent to Vertex AI Agent Engine.
# AUTOMATED UPGRADE: This target now handles "Platform Sanitization".
# It exports dependencies from uv.lock but STRIPS OUT Windows-specific packages
# (like pywin32) that traditionally cause Cloud Build failures on Linux.
backend:
	@echo "Sanitizing dependencies for Linux-based Cloud Build..."
	uv export --no-hashes --no-header --no-dev --no-emit-project \
	| grep -v "sys_platform == 'win32'" \
	| grep -v "platform_system == 'Windows'" > .requirements.txt
	@echo "Deploying to Vertex AI..."
	uv run app/agent_engine_app.py

# Set up development environment resources using Terraform
setup-dev-env:
	PROJECT_ID=$$(gcloud config get-value project) && \
	(cd deployment/terraform/dev && terraform init && terraform apply --var-file vars/env.tfvars --var dev_project_id=$$PROJECT_ID --auto-approve)

# Run unit and integration tests
test:
	uv run pytest tests/unit && uv run pytest tests/integration

# Run code quality checks (codespell, ruff, mypy)
lint:
	uv sync --dev --extra lint
	uv run codespell
	uv run ruff check . --diff
	uv run ruff format . --check --diff
	uv run mypy .