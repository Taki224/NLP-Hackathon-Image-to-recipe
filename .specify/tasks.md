# Tasks

## 1. Scale Training Data (`notebooks/data_creator.ipynb`)
- [ ] Remove the 100-image-per-category cap in the pairing logic.
- [ ] Ensure all ~75,000 images from the Food101 train split are matched and used.
- [ ] Implement a 90/10 train/validation split for the paired dataset.
- [ ] Save the splits to separate JSON files (`paired_dataset_train.json` and `paired_dataset_val.json`).
- [ ] Update dataset final saving logic to reflect the file output changes.

## 2. Validation and Early Stopping (`notebooks/train.ipynb`)
- [ ] Update `FoodRecipeDataset` instantiation to load the new `paired_dataset_train.json` and `paired_dataset_val.json`.
- [ ] Create separate DataLoaders for training and validation.
- [ ] Add a `model.eval()` validation loop at the end of each training epoch to compute the validation InfoNCE loss.
- [ ] Modify the checkpointing/early stopping logic to track and stop based on validation loss rather than average training loss.
- [ ] Log and plot validation loss alongside training loss.

## 3. Ingredient Confidence Scores (`notebooks/train.ipynb`)
- [ ] Modify the `retrieve()` function.
- [ ] Extract ingredients from the top-k retrieved recipes.
- [ ] Compute a weighted score for each ingredient based on the similarity scores of the recipes they belong to.
- [ ] Aggregate and return the sorted `(ingredient, confidence_score)` list along with the top-k recipe outputs.
- [ ] Test the updated retrieval function with sample images to verify the new outputs.