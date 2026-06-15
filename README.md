<!-- generated-by: gsd-doc-writer -->
# Food Photo to Recipe Retrieval

Cross-modal retrieval system that takes a food photograph and returns top-5 matching recipes using CLIP-ViT-L/14 with fine-tuned adapter layers and InfoNCE contrastive loss.

**Team:** Bálint Takács · Levente Lukács · Olivér Reményi — Academic NLP/ML Hackathon

---

## How it works

1. A food photograph is encoded by a frozen CLIP-ViT-L/14 image encoder followed by a learned 2-layer MLP adapter (768 → 256 → 768 with residual connection).
2. 150k Food.com recipes are pre-encoded offline into a recipe embedding index using a symmetric CLIP text encoder with its own adapter.
3. At query time, the image embedding is compared against the recipe index via cosine similarity and the top matches are returned.

---

## Prerequisites

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- A Kaggle account (for `RAW_recipes.csv` download)
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
hf download llevi95/dish-to-recipe best_model.pt \
  --local-dir models/checkpoints
```

### 2. Run the API server

```bash
uv run uvicorn ui.src.app:app --reload
```

The API is available at `http://localhost:8000`. Upload a food image to `POST /retrieve` to get matching recipes.

---

## Usage

### Prepare the dataset

Food101 is downloaded automatically from HuggingFace. For the recipe corpus, download `RAW_recipes.csv` from [Food.com Recipes — Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) and place it in `data/`.

Run the data preparation notebook cell by cell:

```
notebooks/phase2/data_creator.ipynb
```

Alternatively, use the helper scripts to download and process data:

```bash
uv run python helper_scripts/download_full_data.py   # Food101 images
uv run python helper_scripts/recipe_download.py      # Food.com recipes
```

### Train the model

Run the training notebook cell by cell:

```
notebooks/phase2/train.ipynb
```

This trains CLIP-ViT-L/14 adapter layers using symmetric InfoNCE contrastive loss, saves the best checkpoint to `models/checkpoints/best_model.pt`, and pre-computes 150k recipe embeddings to `data/indexes/`.

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
| Adapter | 2-layer MLP with residual connection (768 → 256 → 768) |
| Training objective | Symmetric InfoNCE contrastive loss |
| Temperature | Learnable, initialised at 0.07 |
| Embedding dim | 768 |

---

## Dataset

| Source | Role | Size |
|--------|------|------|
| Food101 (HuggingFace) | Training images | 5,000 (50 categories × 100) |
| Food.com Recipes (Kaggle) | Recipe text + ingredients | 5,000 training pairs / 150k index |

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
- [Food.com Recipes Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- [Food101 Dataset](https://huggingface.co/datasets/food101)
