# Dish-to-Recipe System Improvement Specification

## 1. Overview
Improve the existing Dish-to-Recipe cross-modal retrieval system. The training pipeline currently relies on a limited dataset (5,000 images) and lacks validation-based early stopping. Furthermore, the retrieval step needs an enhancement to provide a weighted aggregation of ingredients from retrieved recipes. The system remains a notebook-based demonstration.

## 2. Requirements

### 2.1 Scale Training Data (`notebooks/data_creator.ipynb`)
- **Remove Cap:** Remove the 100-image-per-category cap during dataset pairing.
- **Full Dataset:** Match and use the full Food101 training split (approximately 75,000 images).
- **Data Splitting:** Implement a robust 90/10 train/validation split. Generate corresponding outputs (e.g., `paired_dataset_train.json` and `paired_dataset_val.json`) to distinguish the splits.

### 2.2 Validation and Early Stopping (`notebooks/train.ipynb`)
- **Validation Loop:** Incorporate the validation dataset split into the training pipeline. Add a validation loop to evaluate the InfoNCE contrastive loss on the validation set at the end of each epoch.
- **Early Stopping:** Modify the checkpointing and early stopping logic to monitor validation loss instead of the training loss.

### 2.3 Ingredient Confidence Scores (`notebooks/train.ipynb`)
- **Soft Voting Mechanism:** Enhance the `retrieve()` function (which currently returns top-3 recipes from a 150k Food.com index).
- **Aggregation:** Aggregate the ingredients from the top-`k` retrieved recipes.
- **Weighting:** Score the ingredients by how often they appear across the top results, weighted by the cosine similarity score of each respective recipe.
- **Output:** Output a sorted list of `(ingredient, confidence_score)` pairs alongside the standard retrieved recipe dictionary.

### 2.4 Future Scope
- The Gradio application in `ui/src/app.py` is excluded from this specification but will be connected after these training and retrieval improvements are validated.
