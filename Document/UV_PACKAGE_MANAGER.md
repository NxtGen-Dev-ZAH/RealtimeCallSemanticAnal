# UV Package Manager Guide

This project uses **uv** as the primary Python package manager. This guide explains how to use uv with this project.

---

## What is UV?

**uv** is an extremely fast Python package installer and resolver, written in Rust. It's designed as a drop-in replacement for pip and pip-tools, and is 10-100x faster than pip.

**Key Benefits:**
- ⚡ **10-100x faster** than pip
- 🔒 **Reproducible builds** with lock files
- 📦 **Modern standard** using pyproject.toml
- 🎯 **Simple commands** similar to npm/yarn

---

## Installation

### macOS/Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Alternative: via pip
```bash
pip install uv
```

### Verify Installation
```bash
uv --version
```

---

## Project Setup

### Initial Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Install all dependencies:**
   ```bash
   uv sync
   ```
   
   This command will:
   - Create a virtual environment (`.venv` by default)
   - Install all dependencies from `pyproject.toml`
   - Create/update `uv.lock` file for reproducible builds

3. **Activate virtual environment (if needed):**
   ```bash
   # macOS/Linux
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate
   ```

### Running the Application

After `uv sync`, you can run the application normally:

```bash
python run_web_app.py
```

---

## Common Commands

### Install Dependencies
```bash
# Install all dependencies from pyproject.toml
uv sync

# Install in editable mode (for development)
uv pip install -e .
```

### Add a New Dependency
```bash
# Add a dependency (updates pyproject.toml and installs)
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Add with version constraint
uv add "package-name>=1.0.0"
```

### Remove a Dependency
```bash
# Remove a package (updates pyproject.toml)
uv remove package-name
```

### Update Dependencies
```bash
# Update all dependencies to latest compatible versions
uv sync --upgrade

# Update a specific package
uv add package-name@latest
```

### Run Commands in Virtual Environment
```bash
# Run a command in the uv-managed environment
uv run python script.py

# Run with specific Python version
uv run --python 3.11 python script.py
```

### Lock File Management
```bash
# Update lock file without installing
uv lock

# Update lock file and sync
uv sync
```

---

## Project Structure

```
backend/
├── pyproject.toml      # Project config and dependencies (primary)
├── uv.lock            # Lock file for reproducible builds
├── requirements.txt   # Kept for compatibility (not primary)
└── .venv/            # Virtual environment (created by uv)
```

### Key Files

- **`pyproject.toml`**: Contains project metadata and dependencies
- **`uv.lock`**: Lock file ensuring reproducible installs
- **`requirements.txt`**: Kept for compatibility, but dependencies are managed in pyproject.toml

---

## Migration from pip/requirements.txt

If you're used to using `pip install -r requirements.txt`, here's the equivalent:

### Old Way (pip)
```bash
pip install -r requirements.txt
```

### New Way (uv)
```bash
uv sync
```

### Installing a Package

**Old Way:**
```bash
pip install package-name
# Then manually add to requirements.txt
```

**New Way:**
```bash
uv add package-name
# Automatically updates pyproject.toml and installs
```

---

## Virtual Environment

### Default Behavior

By default, `uv sync` creates a virtual environment in `.venv/` directory.

### Custom Virtual Environment Location

```bash
# Create venv in custom location
uv venv custom-venv

# Use custom venv
uv sync --python custom-venv/bin/python
```

### Activating Virtual Environment

After `uv sync`, activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

---

## Troubleshooting

### Issue: Command not found
```bash
# Ensure uv is in PATH
# Add to ~/.bashrc or ~/.zshrc:
export PATH="$HOME/.cargo/bin:$PATH"

# Or reinstall uv
```

### Issue: Lock file conflicts
```bash
# Regenerate lock file
uv lock

# Or sync to update
uv sync
```

### Issue: Python version mismatch
```bash
# Use specific Python version
uv sync --python 3.10

# Or specify in pyproject.toml
requires-python = ">=3.10"
```

### Issue: Need to use pip instead
```bash
# uv can use pip-compatible commands
uv pip install package-name

# Or activate venv and use pip normally
source .venv/bin/activate
pip install package-name
```

---

## Comparison: uv vs pip

| Feature | pip | uv |
|---------|-----|-----|
| Speed | Baseline | 10-100x faster |
| Lock files | pip-tools needed | Built-in |
| Project config | requirements.txt | pyproject.toml |
| Dependency resolution | Can be slow | Very fast |
| Virtual env management | Manual | Automatic |

---

## Best Practices

1. **Always use `uv sync`** instead of `pip install`
2. **Commit `uv.lock`** to version control for reproducible builds
3. **Use `uv add`** to add new dependencies (updates pyproject.toml automatically)
4. **Run `uv sync`** after pulling changes to update dependencies
5. **Use `uv run`** for one-off commands in the project environment

---

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Install uv
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Install dependencies
  run: uv sync

- name: Run tests
  run: uv run pytest
```

---

## Additional Resources

- **Official Documentation**: https://docs.astral.sh/uv/
- **GitHub Repository**: https://github.com/astral-sh/uv
- **Project pyproject.toml**: `backend/pyproject.toml`

---

## Quick Reference

```bash
# Setup
uv sync                    # Install all dependencies

# Add/Remove
uv add package-name        # Add dependency
uv remove package-name     # Remove dependency

# Update
uv sync --upgrade          # Update all packages

# Run
uv run python script.py    # Run in project environment

# Lock
uv lock                    # Update lock file
```

---

**Note:** While this project uses `uv` as the primary package manager, `requirements.txt` is kept for compatibility. However, all dependencies should be managed through `pyproject.toml` and `uv sync`.


