# Dish to Recipe — Project Roadmap

## Hackathon MVP (Today)
- [x] Dataset preparation — Food101 + Food.com paired dataset (5k pairs)
- [x] CLIP-ViT-L/14 adapter training with InfoNCE contrastive loss
- [x] Pre-computed recipe index (150k recipes)
- [x] Retrieval pipeline — image in → top-3 recipes out
- [ ] Validate on golden set (10 test images)
- [ ] Gradio UI for live demo

---

## Post-Hackathon Improvements

### Data & Training
- [ ] Scale to full Food101 train split (75k images), remove 100-image-per-category cap
- [ ] Proper train/val split (90/10) before training
- [ ] Add validation loss during training, use it for early stopping and best checkpoint selection
- [ ] Experiment with full Food.com corpus in the index (180k recipes)
- [ ] Try larger adapter bottleneck (256 → 512) and compare results
- [ ] Experiment with unfreezing last few CLIP transformer blocks for deeper fine-tuning
- [ ] Broaden recipe matching to search ingredients column, not just recipe title
- [ ] Assign 3–5 recipes per image for 75k training to reduce duplication

### Evaluation
- [ ] Formal Recall@1, @3, @5 on Food101 test split (25k unseen images)
- [ ] Confusion matrix — which categories get mixed up most?
- [ ] Zero-shot CLIP baseline (no adapters) for comparison — quantify adapter improvement
- [ ] Report median rank as additional metric

### Model Architecture
- [ ] Try separate BERT/Sentence-Transformer text encoder — handles long ingredient lists better than CLIP's 77-token limit
- [ ] Ingredient confidence layer — multi-label classification head returning ranked ingredient list with per-ingredient match scores
- [ ] Inverse retrieval — given a list of ingredients as text input, retrieve most visually matching dish photo from an image index
- [ ] Hard negative mining — use visually similar but wrong recipes as negatives instead of random in-batch negatives
- [ ] Experiment with asymmetric adapter sizes for image vs text branch

### Web Interface
- [ ] Replace Gradio with proper frontend (React or plain HTML/JS)
- [ ] Recipe cards with dish image, full ingredient list, similarity score bar
- [ ] Ranked ingredient list with per-ingredient match scores displayed alongside recipe results
- [ ] Tunable similarity threshold slider
- [ ] Text-only / ingredient list query mode — type ingredients, get recipes
- [ ] Inverse mode UI — enter ingredients, get matching dish photos
- [ ] Multi-image upload
- [ ] Export / save results

### Infrastructure
- [ ] Serve model as REST API (FastAPI)
- [ ] Dockerize full stack (model + API + frontend)
- [ ] Cache frequent queries
- [ ] FAISS index with GPU acceleration for faster retrieval at scale
