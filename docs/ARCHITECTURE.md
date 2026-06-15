# Architecture

<!-- GSD-GENERATED -->

## Overview

Food Photo → Recipe Retrieval is a cross-modal retrieval system. A food photograph is encoded into a shared embedding space alongside a 150k-recipe text index, then nearest-neighbor search returns the top-N closest recipes.

The core is a dual-encoder model built on CLIP-ViT-L/14 with lightweight adapter layers — both visual and text encoders share the same adapter architecture and are trained jointly with symmetric InfoNCE contrastive loss.

## System Diagram

```
Query time:
  [Photo] → CLIP Image Encoder (frozen) → Adapter MLP → embedding (768d)
                                                              │
                                               cosine similarity search
                                                              │
  [Recipe Index] ← precomputed text embeddings ─────────────┘
                                                              │
                                                    top-5 results → [UI]

Indexing (offline):
  [Recipe CSV] → CLIP Text Encoder (frozen) → Adapter MLP → embedding (768d)
                                                              │
                                                    stored in data/indexes/
```

## Components

### Model

| Component | Details |
|-----------|---------|
| Image encoder | CLIP-ViT-L/14 (frozen weights) + 2-layer MLP adapter |
| Text encoder | CLIP text transformer (frozen weights) + 2-layer MLP adapter |
| Adapter architecture | Linear(768→256) → ReLU → Linear(256→768) + residual connection |
| Training loss | Symmetric InfoNCE contrastive loss |
| Temperature | Learnable scalar, initialised at 0.07 |
| Embedding dimension | 768 |
| Checkpoint | `models/checkpoints/best_model.pt` |

Adapters are the only trainable parameters. The CLIP backbone is frozen, which keeps training fast (~30 min on NVIDIA L40S) and avoids catastrophic forgetting.

### Data Pipeline

| Stage | Tool | Output |
|-------|------|--------|
| Food101 image download | HuggingFace `datasets` | `data/images/` |
| Food.com recipe download | `kagglehub` + `recipe_download.py` | CSV |
| Category matching | Keyword matching (dish name → Food.com category) | Aligned image-recipe pairs |
| Dataset splitting | `splitter.py` | train/val splits |
| Train+val merge | `merge_train_val.py` | merged dataset |
| Recipe embedding precomputation | training notebook | `data/indexes/*.npy` |

Helper scripts in `helper_scripts/` handle raw data acquisition and restructuring. Jupyter notebooks in `notebooks/` handle model training and embedding generation.

### Web Application

| Layer | File | Role |
|-------|------|------|
| Backend | `ui/src/app.py` | FastAPI: `/` (HTML), `/retrieve` (POST, image → recipes) |
| Frontend | `ui/static/index.html` | Drag-and-drop upload UI |

**API shape:**

```
POST /retrieve
Content-Type: multipart/form-data
Field: image (file)

Response: JSON array
[
  {
    "rank": 1,
    "recipe_name": "...",
    "ingredients": ["..."],
    "instructions": "...",
    "score": 0.92
  },
  ...
]
```

**Current state:** `/retrieve` returns hardcoded placeholder recipes. The ML inference pipeline (model load → image embed → index search) is not yet wired into the API.

### Training Infrastructure

Training uses PyTorch Lightning (`notebooks/phase2/train.ipynb`). Key features:
- Checkpoint resume from `models/checkpoints/best_model.pt`
- Loss tracking and curve plotting via matplotlib
- Contrastive loss computed over the full batch

## Directory Structure

```
Hackathon/
├── notebooks/
│   ├── phase1/           # Initial data creation + training pipeline
│   │   ├── data_creator.ipynb
│   │   └── train.ipynb
│   └── phase2/           # Refined pipeline with keyword extraction
│       ├── data_creator.ipynb
│       ├── data_creator_keyword.ipynb
│       └── train.ipynb
├── helper_scripts/       # Raw data acquisition and restructuring
│   ├── download_full_data.py   # Food101 via HuggingFace
│   ├── download_isia500.py     # ISIA Food-500 downloader
│   ├── recipe_download.py      # Food.com via kagglehub
│   ├── splitter.py             # Train/val split
│   ├── merger.py               # Dataset merging
│   └── merge_train_val.py      # Merge train and validation sets
├── ui/
│   ├── src/app.py        # FastAPI backend
│   └── static/           # Frontend HTML/JS
├── data/
│   ├── images/           # Training images (not in git)
│   └── indexes/          # Precomputed recipe embeddings (not in git)
├── models/
│   └── checkpoints/      # best_model.pt (not in git)
├── specs/
│   └── 001-task-2/       # Phase plan and tasks
├── docs/                 # Project documentation
├── pyproject.toml        # uv-managed dependencies
└── README.md
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Freeze CLIP backbone | Fast training, avoids forgetting, CLIP features already strong |
| Adapter-only training | Minimal parameters, good transfer |
| Offline recipe indexing | Inference latency is O(1) query embed + ANN search, not O(N) forward passes |
| Title + score output only | Avoids recipe text serving complexity; sufficient for demo |
| uv for dependency management | Reproducible, fast installs |
