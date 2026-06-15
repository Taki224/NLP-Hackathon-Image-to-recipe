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
hf download llevi95/dish-to-recipe best_model.pt --local-dir models/checkpoints
```

Expected path: `models/checkpoints/best_model.pt`

## Run the Web App (inference only)

With the checkpoint in place, run the FastAPI app:

```bash
uvicorn ui.src.app:app --reload
```

Open `http://localhost:8000` in a browser, upload a food photo, and see recipe results.

> **Note:** The current `/retrieve` endpoint returns placeholder recipes. Wiring the real model into the API is an open task (see `specs/001-task-2/`).

## Full Setup (data + training)

### 1. Download Food101 images

```bash
uv run python helper_scripts/download_full_data.py
```

This downloads Food101 via HuggingFace datasets and saves images locally. Update `output_dir` in the script to your preferred path.

### 2. Download Food.com recipes

Set Kaggle credentials first:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

Then:
```bash
uv run python helper_scripts/recipe_download.py
```

Saves `output/recipe_dataset_2m.csv` (2M+ recipes).

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

Training saves the best checkpoint to `models/checkpoints/best_model.pt` and precomputes recipe embeddings to `data/indexes/`.

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
