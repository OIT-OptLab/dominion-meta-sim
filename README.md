# dominion-meta-sim

AI-based simulation framework for analyzing and optimizing game balance through evolving meta dynamics in Dominion.

---

## Environment Setup (using uv)
This project uses [uv](https://github.com/astral-sh/uv), a fast and reproducible Python package manager, to manage dependencies and virtual environments.

### 1. Install uv

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```
Alternatively, using pip:
```bash
pip install uv
```

### 2. Initialize Project
```bash
mkdir dominion-meta-sim
cd dominion-meta-sim

uv init
```

This will generate a pyproject.toml file.

### 3. Create and Activate Virtual Environment
```bash
uv venv
source .venv/bin/activate  # macOS / Linux
```

---
