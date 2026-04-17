
---

## Usage

### 1. Prepare Dataset
Run `dataset_prep.ipynb` cell by cell.

Requires `RAW_recipes.csv` from [Food.com Recipes — Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions). Food101 is downloaded automatically via HuggingFace.

### 2. Train the Model
Run `training.ipynb` cell by cell.

Trains CLIP-ViT-L/14 adapter layers using InfoNCE contrastive loss. Saves best checkpoint to `checkpoints/best_model.pt` and pre-computes 150k recipe embeddings to `recipe_index.npy`. Expected training time: ~30 min on an NVIDIA L40S.

### 3. Run Retrieval
Drop food photos into `test_images/` and run Step 11 of `training.ipynb`.

---

## Model Architecture

| Component | Details |
|---|---|
| Image encoder | CLIP-ViT-L/14 (frozen) + adapter layers |
| Text encoder | CLIP text transformer (frozen) + adapter layers |
| Adapter | 2-layer MLP with residual connection (768 → 256 → 768) |
| Training objective | Symmetric InfoNCE contrastive loss |
| Temperature | Learnable, initialised at 0.07 |
| Embedding dim | 768 |

---

## Dataset

| Source | Role | Size |
|---|---|---|
| Food101 (HuggingFace) | Training images | 5,000 (50 categories × 100) |
| Food.com Recipes (Kaggle) | Recipe text + ingredients | 5,000 training pairs / 150k index |

Categories are matched from Food101 to Food.com recipes by dish name keyword matching.

---

## Results

Evaluated on 10 manually curated golden set queries (unseen test images).

| Metric | Score |
|---|---|
| Recall@1 | TBD |
| Recall@3 | TBD |
| Recall@5 | TBD |

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full post-hackathon improvement plan.

---

## Team

Bálint Takács · Levente Lukács · Olivér Reményi

---

## References

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — Radford et al., OpenAI
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Food.com Recipes Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- [Food101 Dataset](https://huggingface.co/datasets/food101)