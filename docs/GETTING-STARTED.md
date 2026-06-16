# Getting Started

<!-- GSD-GENERATED -->

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- Kaggle account (for dataset download)
- GPU recommended for training (CPU works for inference)

## Installation

```bash
git clone <repo-url>
cd Hackathon
uv sync
```

## Download Model Weights

The trained checkpoint is not stored in git. Download it from HuggingFace:

```bash
uv run hf download llevi95/dish-to-recipe best_model.pt --local-dir models/checkpoints
```

Expected path: `models/checkpoints/best_model.pt`

### (Optional) Download the full recipe index

The bundled seed recipes work with just the checkpoint. To retrieve against the full corpus instead, download the prebuilt index (~3.2 GB fp16 embeddings + ids + recipe text):

```bash
uv run hf download llevi95/dish-to-recipe \
  recipe_index_fp16.npy recipe_index_ids.npy recipe_index_metadata_full.json \
  --local-dir data/indexes
```

Once these sit in `data/indexes/`, the app uses them automatically (it prefers the full index over the seed set). The fp16 embeddings load to ~3.2 GB RAM on first query.

## Run the Web App (inference only)

With the checkpoint in place, run the FastAPI app **from inside `ui/`** (the app reads `static/index.html` relative to the working directory):

```bash
cd ui && uv run uvicorn src.app:app --reload
```

Open `http://localhost:8000` in a browser, upload a food photo, and see recipe results.

> **How `/retrieve` works:** with the checkpoint present it runs the real CLIP + adapter model and ranks a small **bundled recipe set** (`ui/src/recipes_seed.json`) by cosine similarity — real end-to-end inference on any photo. If a full prebuilt index exists at `data/indexes/recipe_index.npy` (+ ids + metadata), it uses that instead. If the checkpoint or the ML deps (`torch`, `open_clip`) are missing, it falls back to placeholder recipes so the UI still works. The app reads `static/index.html` by absolute path, so it runs from any directory.

## Full Setup (data + training)

### 1. Download Food101 images

```bash
uv run python helper_scripts/download_full_data.py
```

This downloads Food101 via HuggingFace datasets and saves images locally. Update `output_dir` in the script to your preferred path.

### 2. Download the recipe corpus

`recipe_download.py` uses `kagglehub` (reads `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`) and pulls `wilmerarltstrmberg/recipe-dataset-over-2m`:

```bash
uv run python helper_scripts/recipe_download.py
```

This saves `output/recipe_dataset_2m.csv` (2M+ recipes). The notebooks read it from `data/datasets/`, so move it there:

```bash
mkdir -p data/datasets && mv output/recipe_dataset_2m.csv data/datasets/
```

### 3. (Optional) Download ISIA Food-500

For additional training images (~43 GB):
```bash
uv run python helper_scripts/download_isia500.py
```

Supports resume on interrupted downloads.

### 4. Merge datasets

```bash
uv run python helper_scripts/merger.py          # merge Food101 + ISIA-500
uv run python helper_scripts/merge_train_val.py # merge train/val splits
```

### 5. Split dataset

```bash
uv run python helper_scripts/splitter.py   # 80/20 train/val split
```

### 6. Data creation and training

Open `notebooks/phase2/data_creator.ipynb` and run cells top-to-bottom. Then open `notebooks/phase2/train.ipynb` and run cells.

Training exports the adapter weights to `models/checkpoints/best_model.pt` and precomputes the recipe index to `data/indexes/`: embeddings (`recipe_index.npy`), ids (`recipe_index_ids.npy`), and recipe text metadata (`recipe_index_metadata_*.json`). All three are needed for retrieval — the metadata supplies the titles/ingredients shown for each result.

The app prefers a half-size fp16 index if present. Before uploading/sharing, convert the fp32 embeddings once (halves size, ranking unchanged):

```bash
uv run python -c "import numpy as np; np.save('data/indexes/recipe_index_fp16.npy', np.load('data/indexes/recipe_index.npy', mmap_mode='r').astype(np.float16))"
```

## Project Structure

```
Hackathon/
├── notebooks/      # Data creation + training notebooks
├── helper_scripts/ # Data download + processing scripts
├── ui/             # FastAPI backend + HTML frontend
├── data/           # Images and embedding indexes (not in git)
├── models/         # Checkpoints (not in git)
└── docs/           # Documentation
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design and [CONFIGURATION.md](CONFIGURATION.md) for all configuration options.
