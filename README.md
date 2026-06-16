<!-- generated-by: gsd-doc-writer -->
# Food Photo to Recipe Retrieval

Cross-modal retrieval system that takes a food photograph and returns top-5 matching recipes using CLIP-ViT-L/14 with fine-tuned adapter layers and InfoNCE contrastive loss.

**Team:** Bálint Takács · Levente Lukács · Olivér Reményi — Academic NLP/ML Hackathon

---

## How it works

1. A food photograph is encoded by a frozen CLIP-ViT-L/14 image encoder followed by a learned 2-layer MLP adapter (768 → 64 → 768 with residual connection).
2. The recipe corpus is pre-encoded offline into a recipe embedding index using a symmetric CLIP text encoder with its own adapter.
3. At query time, the image embedding is compared against the recipe index via cosine similarity and the top matches are returned.

The web app works out of the box on a small **bundled recipe set** (`ui/src/recipes_seed.json`) so you can try it with just the checkpoint. Build the full index (below) to retrieve against the whole corpus instead.

---

## Prerequisites

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- A Kaggle account (for the recipe corpus download, full-setup only)
- NVIDIA GPU recommended — training runs ~30 min on an NVIDIA L40S

---

## Installation

```bash
git clone <repo-url>
cd Hackathon
uv sync
```

---

## Quick Start

### 1. Download model weights

Trained weights are not stored in git. Download the checkpoint and place it at `models/checkpoints/best_model.pt`:

```bash
uv run hf download llevi95/dish-to-recipe best_model.pt \
  --local-dir models/checkpoints
```

Optionally grab the full prebuilt recipe index (~3.2 GB fp16 + ids + recipe text) to retrieve against the whole corpus instead of the bundled seed set:

```bash
uv run hf download llevi95/dish-to-recipe \
  recipe_index_fp16.npy recipe_index_ids.npy recipe_index_metadata_full.json \
  --local-dir data/indexes
```

### 2. Run the API server

```bash
uv run uvicorn ui.src.app:app --reload
```

The API is available at `http://localhost:8000`. Open it in a browser and upload a food photo, or `POST /retrieve` directly. With the checkpoint present it runs the real model against the bundled recipe set; without it (or without the ML deps) it returns placeholder recipes so the UI still works.

---

## Usage

### Prepare the dataset

Use the helper scripts to download the data, then the data-prep notebook:

```bash
uv run python helper_scripts/download_full_data.py   # Food101 images (edit output_dir first)
uv run python helper_scripts/recipe_download.py      # recipe corpus -> output/recipe_dataset_2m.csv
```

`recipe_download.py` pulls the `wilmerarltstrmberg/recipe-dataset-over-2m` dataset via `kagglehub`. Move its CSV to where the notebooks read it:

```bash
mkdir -p data/datasets && mv output/recipe_dataset_2m.csv data/datasets/
```

Then run the data preparation notebook cell by cell:

```
notebooks/phase2/data_creator.ipynb
```

### Train the model

Run the training notebook cell by cell:

```
notebooks/phase2/train.ipynb
```

This trains CLIP-ViT-L/14 adapter layers using symmetric InfoNCE contrastive loss, exports the adapter weights to `models/checkpoints/best_model.pt`, and pre-computes the recipe index — embeddings (`recipe_index.npy`), ids (`recipe_index_ids.npy`), and recipe text metadata (`recipe_index_metadata_*.json`) — to `data/indexes/`. The web app reads all three; the metadata is what makes retrieved recipes show titles and ingredients rather than `"unknown"`.

### Run retrieval

Upload a food photo to the `/retrieve` endpoint (see API section below) or use the retrieval cells at the end of the training notebook for direct inference.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the frontend HTML interface |
| `POST` | `/retrieve` | Upload a food image, receive ranked recipe matches |

**Request (POST /retrieve):** `multipart/form-data` with field `image` containing the food photo.

**Response:**

```json
[
  {
    "rank": 1,
    "recipe_name": "Spaghetti Carbonara",
    "ingredients": ["pasta", "eggs", "bacon", "parmesan", "black pepper"],
    "instructions": "...",
    "score": 0.92
  }
]
```

---

## Model Architecture

| Component | Details |
|-----------|---------|
| Image encoder | CLIP-ViT-L/14 (frozen) + adapter layers |
| Text encoder | CLIP text transformer (frozen) + adapter layers |
| Adapter | 2-layer MLP with residual connection (768 → 64 → 768) |
| Training objective | Symmetric InfoNCE contrastive loss |
| Temperature | Learnable, initialised at 0.07 |
| Embedding dim | 768 |

---

## Dataset

| Source | Role |
|--------|------|
| Food101 (HuggingFace) | Training images |
| Recipe Dataset over 2M (Kaggle, `wilmerarltstrmberg/recipe-dataset-over-2m`) | Recipe text + ingredients; matched to images by dish-name keyword |

Categories are matched from Food101 to Food.com recipes by dish name keyword matching.

---

## Project Structure

```
Hackathon/
├── data/
│   ├── images/          # Food101 images
│   └── indexes/         # Pre-computed recipe embedding index
├── helper_scripts/      # Data download and processing utilities
├── models/
│   └── checkpoints/     # best_model.pt (download from HuggingFace)
├── notebooks/
│   ├── phase1/          # Initial data creation and training experiments
│   └── phase2/          # Final training pipeline with validation
├── ui/
│   ├── src/app.py       # FastAPI backend
│   └── static/          # Frontend HTML
└── pyproject.toml       # uv-managed dependencies
```

---

## Results

Evaluated on 10 manually curated golden-set queries (unseen test images).

| Metric | Score |
|--------|-------|
| Recall@1 | TBD |
| Recall@3 | TBD |
| Recall@5 | TBD |

---

## References

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — Radford et al., OpenAI
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Recipe Dataset over 2M (Kaggle)](https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m)
- [Food101 Dataset](https://huggingface.co/datasets/food101)
