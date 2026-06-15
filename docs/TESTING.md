# Testing

<!-- GSD-GENERATED -->

## Current State

No automated test suite exists. Verification is done manually via notebooks and the web UI.

## Manual Verification

### Model inference

1. Download checkpoint (`models/checkpoints/best_model.pt`) — see [GETTING-STARTED.md](GETTING-STARTED.md)
2. Run the FastAPI app: `uvicorn ui.src.app:app --reload`
3. Upload a food photo at `http://localhost:8000`
4. Verify recipe results are returned

### Training correctness

Run `notebooks/phase2/train.ipynb` end-to-end and verify:
- Loss decreases across epochs
- Checkpoint saved to `models/checkpoints/best_model.pt`
- Recipe embeddings saved to `data/indexes/`

### Data pipeline

Run `notebooks/phase2/data_creator.ipynb` end-to-end and verify image-recipe pairs are created without errors.

## Evaluation Metrics

The project uses retrieval metrics on a held-out test set:

| Metric | Description |
|--------|-------------|
| Recall@1 | Correct recipe in top-1 result |
| Recall@3 | Correct recipe in top-3 results |
| Recall@5 | Correct recipe in top-5 results |

A 10-query golden set (manually curated, unseen test images) is used for evaluation. Results are reported in the README.

Evaluation is performed in the training notebook — run the evaluation cells after training completes.

## Adding Tests

If a test suite is added, use `pytest`:
```bash
uv add --dev pytest
uv run pytest
```

Priority areas for tests:
- Model loading and forward pass shape
- Cosine similarity / index search correctness
- `/retrieve` API endpoint response schema
- Data pipeline output structure
