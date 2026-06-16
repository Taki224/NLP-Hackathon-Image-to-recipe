# Notebooks

<!-- GSD-GENERATED -->

The project uses Jupyter notebooks for data creation, model training, and evaluation. All notebooks live in `notebooks/` organized by phase.

## Running Notebooks

```bash
uv run jupyter notebook
```

Run all cells top-to-bottom unless a cell's comment says otherwise.

## Phase 1 (`notebooks/phase1/`)

Initial pipeline — kept for reference. The phase2 pipeline supersedes it.

| Notebook | Purpose |
|----------|---------|
| `data_creator.ipynb` | Creates image-recipe pairs from Food101 + Food.com. Matches Food101 categories to Food.com recipes by dish name keyword matching. Outputs training data. |
| `train.ipynb` | Trains the CLIP adapter model. Implements InfoNCE contrastive loss, saves best checkpoint to `models/checkpoints/best_model.pt`, precomputes 150k recipe embeddings. Expected runtime ~30 min on NVIDIA L40S. |

## Phase 2 (`notebooks/phase2/`)

Current active pipeline with improved data quality.

| Notebook | Purpose |
|----------|---------|
| `data_creator.ipynb` | Refined data creation pipeline. |
| `data_creator_keyword.ipynb` | Keyword-extraction variant of data creation — extracts key terms from recipe titles for better category matching. |
| `train.ipynb` | Training notebook for phase2 data. Same architecture as phase1 but trained on the improved dataset. |

## Execution Order

For a full run from scratch:

1. `notebooks/phase2/data_creator.ipynb` (or `data_creator_keyword.ipynb`)
2. `notebooks/phase2/train.ipynb`

Phase1 notebooks are standalone — they can be run independently but produce lower-quality data.

## Outputs

| Notebook output | Path |
|----------------|------|
| Model checkpoint | `models/checkpoints/best_model.pt` |
| Recipe embedding index | `data/indexes/` |
| Training curves | displayed inline |

Both output paths are excluded from git.
