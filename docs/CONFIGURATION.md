# Configuration

<!-- GSD-GENERATED -->

## Python Version

Requires Python 3.12 (see `.python-version`).

## Package Manager

This project uses [uv](https://docs.astral.sh/uv/). Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

```bash
uv sync
```

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| `open-clip-torch` | ≥3.3.0 | CLIP-ViT-L/14 backbone and tokenizer |
| `torch` | ≥2.11.0 | Deep learning framework |
| `torchvision` | ≥0.26.0 | Image transforms |
| `lightning` | ≥2.6.3 | Training loop utilities |
| `datasets` | ≥4.8.4 | HuggingFace Food101 download |
| `kagglehub` | ≥1.0.0 | Food.com dataset download |
| `numpy` | ≥2.4.4 | Embedding index operations |
| `pandas` | ≥3.0.2 | Recipe CSV processing |
| `pillow` | ≥12.2.0 | Image I/O |
| `transformers` | ≥5.7.0 | Tokenizer utilities |
| `sentencepiece` | ≥0.2.1 | Tokenizer backend |
| `matplotlib` | ≥3.10.8 | Training curve plots |
| `tqdm` | ≥4.67.3 | Progress bars |

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `KAGGLE_USERNAME` | Yes (for data download) | Kaggle API authentication |
| `KAGGLE_KEY` | Yes (for data download) | Kaggle API key |
| `HF_TOKEN` | No <!-- VERIFY: required for private repos --> | HuggingFace authentication |

Set Kaggle credentials:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

Or place them in `~/.kaggle/kaggle.json`:
```json
{"username": "your_username", "key": "your_api_key"}
```

## Data Paths

| Path | Contents | In git |
|------|---------|--------|
| `data/images/` | Training images (Food101) | No |
| `data/indexes/` | Precomputed recipe embeddings (`.npy`) | No |
| `models/checkpoints/` | Model checkpoint (`best_model.pt`) | No |

These directories must be populated before inference. See [GETTING-STARTED.md](GETTING-STARTED.md).

## Model Checkpoint

The trained checkpoint is hosted on HuggingFace at `llevi95/dish-to-recipe` <!-- VERIFY: repository is publicly accessible -->.

Download:
```bash
uv run hf download llevi95/dish-to-recipe best_model.pt --local-dir models/checkpoints
```

Place the file at `models/checkpoints/best_model.pt`.

## Datasets

### Food101 (images)
Downloaded automatically via HuggingFace `datasets` library in the data creation notebooks. No manual steps needed.

HuggingFace cache: `~/.cache/huggingface/` <!-- VERIFY: exact cache path on your platform -->

### Recipe corpus (text)
Downloaded via `helper_scripts/recipe_download.py`, which uses `kagglehub` to pull `wilmerarltstrmberg/recipe-dataset-over-2m` and writes `output/recipe_dataset_2m.csv`. The notebooks expect it at `data/datasets/recipe_dataset_2m.csv`:

```bash
uv run python helper_scripts/recipe_download.py
mkdir -p data/datasets && mv output/recipe_dataset_2m.csv data/datasets/
```

The CSV must have `title`, `ingredients`, and `directions` columns (the indexing notebook reads these).

## Model Hyperparameters

These are set in the training notebooks (`notebooks/phase2/train.ipynb`), not in a config file.

| Parameter | Default | Location |
|-----------|---------|---------|
| CLIP backbone | `ViT-L-14` | notebook cell |
| Adapter hidden dim (bottleneck) | 64 | notebook cell (`ADAPTER_BOTTLENECK`) |
| Embedding dim | 768 | model architecture |
| Temperature init | 0.07 | model architecture |
| Batch size | <!-- VERIFY: check notebook --> | notebook cell |
| Learning rate | <!-- VERIFY: check notebook --> | notebook cell |
| Training epochs | <!-- VERIFY: check notebook --> | notebook cell |

## Web App

The FastAPI app (`ui/src/app.py`) has no external config file. Host and port are set via `uvicorn` flags:

```bash
uvicorn ui.src.app:app --host 0.0.0.0 --port 8000
```
