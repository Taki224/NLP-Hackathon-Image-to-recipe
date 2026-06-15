# Development

<!-- GSD-GENERATED -->

## Environment Setup

```bash
uv sync          # install all dependencies
```

Python 3.12 is required (see `.python-version`).

## Project Layout

| Path | Purpose |
|------|---------|
| `notebooks/phase1/` | Original data creation and training pipeline |
| `notebooks/phase2/` | Refined pipeline — keyword-based data creation + training |
| `helper_scripts/` | Standalone data download and preprocessing scripts |
| `ui/src/app.py` | FastAPI backend |
| `ui/static/` | HTML/JS frontend |
| `data/images/` | Training images (not in git) |
| `data/indexes/` | Precomputed recipe embeddings (not in git) |
| `models/checkpoints/` | Model checkpoint (not in git) |
| `specs/` | Phase specs and task breakdowns |

## Development Workflow

### Adding to the web app

The backend is `ui/src/app.py`. It's a plain FastAPI app — no special build step.

Run with hot reload:
```bash
uvicorn ui.src.app:app --reload
```

The frontend (`ui/static/index.html`) is served as a static file. Edit it directly.

### Working on the ML pipeline

Use Jupyter notebooks directly:
```bash
uv run jupyter notebook
```

The `notebooks/phase2/` pipeline is the current active one. Phase1 notebooks are kept for reference.

### Wiring ML into the API

The open task is connecting the inference pipeline from `notebooks/phase2/train.ipynb` to `ui/src/app.py`. The `/retrieve` endpoint currently returns hardcoded results. Key steps:
1. Extract model loading + embedding code from the training notebook into a callable module
2. Load the checkpoint and recipe index at app startup
3. In `/retrieve`: embed the uploaded image → cosine search against index → return top-N results

See `specs/001-task-2/plan.md` for the detailed plan.

### Helper scripts

Scripts in `helper_scripts/` have hardcoded absolute paths (pointing to the original HPC cluster). Update `BASE_DIR` / `output_dir` constants before running locally.

## Dependencies

Managed by uv. To add a dependency:
```bash
uv add <package>
```

To remove:
```bash
uv remove <package>
```

`uv.lock` is committed — ensures reproducible installs.

## Key Files

| File | Role |
|------|------|
| `pyproject.toml` | Dependency declarations |
| `uv.lock` | Pinned dependency tree |
| `.python-version` | Python version pin (3.12) |
| `.gitignore` | Excludes data/, models/checkpoints/, .venv/ |

## Data and Checkpoints

`data/` and `models/checkpoints/` are excluded from git. After cloning:
- Download checkpoint: see [GETTING-STARTED.md](GETTING-STARTED.md)
- Regenerate data: run helper scripts + notebooks (see [GETTING-STARTED.md](GETTING-STARTED.md))

## Code Style

No linter is configured. The project uses standard Python conventions.
