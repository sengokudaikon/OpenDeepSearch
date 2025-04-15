# =====================================================================
# OpenDeepSearch - Modern Python Development Justfile
# =====================================================================

# Default recipe to run when just is called without arguments
default:
    @just --list

# =====================================================================
# ENVIRONMENT SETUP
# =====================================================================

# Setup the virtual environment with uv
setup:
    @echo "Creating virtual environment with uv..."
    uv venv
    just install-dev

# Create a fresh virtual environment and install all dependencies
fresh-start:
    @echo "Creating fresh environment..."
    rm -rf .venv
    just setup
    just install-all

# Clean Python cache files and build artifacts
clean:
    @echo "Cleaning cache files and build artifacts..."
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name ".ruff_cache" -exec rm -rf {} +
    find . -type d -name ".mypy_cache" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    rm -rf build/ dist/ *.egg-info/

# =====================================================================
# DEPENDENCY MANAGEMENT
# =====================================================================

# Install main project dependencies
install:
    @echo "Installing main project dependencies..."
    uv pip install -e .

# Install development dependencies
install-dev:
    @echo "Installing development dependencies..."
    uv pip install -e ".[dev]"

# Update all dependencies with uv
update-deps:
    @echo "Updating dev dependencies..."
    uv pip install --upgrade -e ".[dev]"
    just compile-all

# Install from compiled requirements.txt (useful for CI/production)
install-req:
    @echo "Installing from requirements.txt..."
    uv pip install -r requirements.txt

# Install from compiled requirements-dev.txt
install-req-dev:
    @echo "Installing from requirements-dev.txt..."
    uv pip install -r requirements-dev.txt

# =====================================================================
# REQUIREMENTS COMPILATION
# =====================================================================

# Compile requirements.txt from pyproject.toml
compile-reqs:
    @echo "Compiling requirements.txt..."
    uv pip compile pyproject.toml -o requirements.txt

# Compile requirements with dev dependencies
compile-dev-reqs:
    @echo "Compiling requirements-dev.txt..."
    uv pip compile pyproject.toml --extra=dev -o requirements-dev.txt

# =====================================================================
# CODE QUALITY
# =====================================================================

# Format code using black
format:
    @echo "Formatting code with black..."
    black .

# Lint code using ruff
lint:
    @echo "Linting code with ruff..."
    ruff check .

# Fix linting issues with ruff
lint-fix:
    @echo "Fixing linting issues with ruff..."
    ruff check --fix .

# Run type checking with mypy
type-check:
    @echo "Type checking with mypy..."
    mypy .

# Run security checks with bandit
security-check:
    @echo "Running security checks with bandit..."
    bandit -r .

# Run all code quality checks
quality-check: format lint type-check security-check
    @echo "All code quality checks completed."

# =====================================================================
# TESTING
# =====================================================================

# Run tests using pytest
test:
    @echo "Running tests..."
    pytest

# Run tests with coverage report
test-cov:
    @echo "Running tests with coverage..."
    pytest --cov=. --cov-report=term --cov-report=html

# Run tests in verbose mode
test-verbose:
    @echo "Running tests in verbose mode..."
    pytest -v

# =====================================================================
# DEMO & EVALUATION
# =====================================================================

# Run the gradio demo (default demo for backward compatibility)
demo:
    @echo "Installing gradio demo dependencies and running gradio demo..."
    uv pip install -e ".[gradio-demo]"
    python gradio_demo.py

# Run the web demo
web-demo:
    @echo "Installing web demo dependencies and running web demo..."
    uv pip install -e ".[web-demo]"
    python simple_web_demo.py --pro-mode

# Run all demos (installs all demo dependencies)
all-demos:
    @echo "Installing all demo dependencies..."
    uv pip install -e ".[demo]"
    @echo "Run specific demos with 'just demo' or 'just web-demo'"

# =====================================================================
# BUILD & DISTRIBUTION
# =====================================================================

# Build the package
build:
    @echo "Building package..."
    python -m build

# Build and check the package with twine
build-check:
    @echo "Building and checking package..."
    python -m build
    twine check dist/*

# =====================================================================
# PRE-COMMIT HOOKS
# =====================================================================

# Install pre-commit hooks
install-hooks:
    @echo "Installing pre-commit hooks..."
    pre-commit install

# Run pre-commit hooks on all files
run-hooks:
    @echo "Running pre-commit hooks on all files..."
    pre-commit run --all-files

# =====================================================================
# UTILITY COMMANDS
# =====================================================================

# List all available commands
help:
    @just --list

# Show project information
info:
    @echo "Python version:" && python --version
    @echo "Virtual environment:" && which python
    @echo "Package version:" && python -c "import importlib.metadata; print(importlib.metadata.version('OpenDeepSearch'))"

# Create a new version tag
version tag='':
    #!/usr/bin/env python3
    import re
    import subprocess
    from pathlib import Path

    if not "{{tag}}":
        print("Please provide a version tag (e.g., just version 1.0.0)")
        exit(1)

    # Validate version format
    if not re.match(r'^\d+\.\d+\.\d+$', "{{tag}}"):
        print("Version must be in format X.Y.Z")
        exit(1)

    # Update version in pyproject.toml
    pyproject = Path("pyproject.toml")
    content = pyproject.read_text()
    content = re.sub(r'version = "\d+\.\d+\.\d+"', f'version = "{{tag}}"', content)
    pyproject.write_text(content)

    print(f"Updated version to {{tag}} in pyproject.toml")
