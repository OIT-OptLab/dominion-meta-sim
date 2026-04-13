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

### 2. Setup Environment
```bash
uv sync
```
This will:
- create a virtual environment (.venv)
- install all dependencies from uv.lock

### 3. Activate Environment
```bash
source .venv/bin/activate  # macOS / Linux
```

### 4. Run
```bash
uv run python main.py
```

---
