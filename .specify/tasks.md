# Tasks

## 1. Scale Training Data (`notebooks/data_creator.ipynb`)
- [ ] Remove the 100-image-per-category cap in the pairing logic.
- [ ] Ensure all ~75,000 images from the Food101 train split are matched and used.
- [ ] Broaden the recipe search beyond title-only matching to also search the ingredients column (recipes without keywords in the title but in ingredients should match).
- [ ] Assign 3–5 matching recipes per image instead of 1 (creates 3-5 positive pairs per image instead of 1, generating ~375k pairs).
- [ ] Deduplicate matched recipes per image using a `seen_recipe_ids` set, so the same recipe is not assigned to the same image twice even if it matches on both title and ingredients.
- [ ] Shuffle all pairs before performing the 90/10 split to prevent category-biased splits (e.g. all images of one category ending up only in train or only in val).
- [ ] Implement a 90/10 train/validation split for the paired dataset.
- [ ] Save the splits to separate JSON files (`paired_dataset_train.json` and `paired_dataset_val.json`).
- [ ] Update dataset final saving logic to reflect the file output changes.

## 2. Validation and Early Stopping (`notebooks/train.ipynb`)
- [ ] Update `FoodRecipeDataset` instantiation to load the new `paired_dataset_train.json` and `paired_dataset_val.json`.
- [ ] Create separate DataLoaders for training and validation.
- [ ] Add a `model.eval()` validation loop at the end of each training epoch to compute the validation InfoNCE loss.
- [ ] Modify the checkpointing/early stopping logic to track and stop based on validation loss rather than average training loss.
- [ ] Log and plot validation loss alongside training loss.
- [ ] After training completes, re-run the Step 10 `retrieve()` sanity check with a few sample images to confirm the new model and updated index produce sensible results before hackathon day.
- [ ] Confirm whether the 150k recipe retrieval index (`.npy`) needs to be rebuilt or can be reused as-is with the newly trained adapter weights.

## 3. Ingredient Confidence Scores (`notebooks/train.ipynb`)
- [ ] Modify the `retrieve()` function.
- [ ] Extract ingredients from the top-k retrieved recipes.
- [ ] Compute a weighted score for each ingredient based on the similarity scores of the recipes they belong to.
- [ ] Aggregate and return the sorted `(ingredient, confidence_score)` list along with the top-k recipe outputs.
- [ ] Test the updated retrieval function with sample images to verify the new outputs.

## 4. Formal Evaluation (Post-Hackathon)
- [ ] Run the trained model on the Food101 test split (never seen during training).
- [ ] Implement Recall@1, Recall@3, and Recall@5 metrics — for each test image, check if the correct category recipe appears in the top-1, top-3, and top-5 retrieved results.
- [ ] Log and report the results as the official benchmark for the model.