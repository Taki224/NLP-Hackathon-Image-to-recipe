# Tasks

## 1. Create new Training Data script (`notebooks/phase2/data_creator.ipynb`)

### 1.1 Data Loading & Setup
- [ ] Parse `data/datasets/final_dataset/train` and `data/datasets/final_dataset/val` folders instead of Food101 from Hugging Face since the dataset is already split ~80/20.
- [ ] Load the new 2M recipe dataset from `data/datasets/recipe_dataset_2m.csv` instead of RAW_recipes.csv.

### 1.2 Recipe Matching & Formatting
- [ ] Sanitize directory/category names (e.g., replacing underscores with spaces, like `spaghetti_carbonara` -> `spaghetti carbonara`) to create reliable keyword mappings for recipe matching.
- [ ] Broaden the recipe search beyond title-only matching to also search the ingredients column (recipes without keywords in the title but in ingredients should match).
- [ ] Assign 3–5 matching recipes per image instead of 1 (creates 3-5 positive pairs per image instead of 1).
- [ ] Deduplicate matched recipes per image using a `seen_recipe_ids` set, so the same recipe is not assigned to the same image twice even if it matches on both title and ingredients.
- [ ] Format the text output for each recipe to include *only* the Title and Ingredients, deliberately excluding the instructions to fit within CLIP's 77-token limit.

### 1.3 Exporting & Reporting
- [ ] Process `train` and `val` folders independently and save them to separate JSON files (`paired_dataset_train.json` and `paired_dataset_val.json`).
- [ ] Add a reporting step to display categories that returned 0 or very few (e.g., <20) recipe matches to allow for manual investigation and keyword adjustments.
- [ ] Update dataset final saving logic to reflect the file output changes.

## 2. Validation and Early Stopping (`notebooks/phase2/train.ipynb`)

### 2.1 Dataset & Loader Adjustments
- [ ] Update `FoodRecipeDataset` instantiation to load the new `paired_dataset_train.json` and `paired_dataset_val.json`.
- [ ] Create separate DataLoaders for training and validation.
- [ ] Update the `text_adapter` and tokenizer logic to enforce the 77-token limit cleanly, prioritizing the "Title + Ingredients" global/local features.

### 2.2 Model Architecture Experiments
- [ ] Investigate and try swapping standard CLIP for Long-CLIP to natively support a larger text token window (allowing full recipes including instructions).
- [ ] Experiment with unfreezing the last few layers of the base CLIP model (image and text encoders) alongside the adapter layers to improve domain-specific accuracy.

### 2.3 Training Loop & Metrics
- [ ] Add a `model.eval()` validation loop at the end of each training epoch to compute the validation InfoNCE loss.
- [ ] Modify the checkpointing/early stopping logic to track and stop based on validation loss rather than average training loss.
- [ ] Log and plot validation loss alongside training loss.

### 2.4 Post-Training Tasks
- [ ] After training completes, re-run the Step 10 `retrieve()` sanity check with a few sample images to confirm the new model and updated index produce sensible results before hackathon day.
- [ ] Generate the recipe retrieval `.npy` index for all 2 million recipes after training is complete (batch loading to avoid RAM crashes).

## 3. Ingredient Confidence Scores (`notebooks/phase3/train.ipynb`)

### 3.1 Ingredient Parsing
- [ ] Write a `parse_ingredient(raw: str) -> str` helper that strips quantities, units, and prep notes using regex, keeping only the ingredient noun.
- [ ] Filter out any parsed strings shorter than 3 characters (junk/empty results).

### 3.2 Confidence Score Calculation
- [ ] Modify the `retrieve()` function to handle the new ingredient aggregation.
- [ ] Extract ingredients from the top-k retrieved recipes.
- [ ] In `get_ingredient_confidence()`, iterate top-k results, parse each ingredient, and accumulate similarity-weighted scores per ingredient name.
- [ ] Compute a weighted score for each ingredient based on the similarity scores of the recipes they belong to.
- [ ] Normalize final scores to 0–1 range and return as a sorted list of (ingredient, confidence) tuples.
- [ ] Aggregate and return the sorted `(ingredient, confidence_score)` list along with the top-k recipe outputs.

### 3.3 Testing
- [ ] Test the updated retrieval function with sample images to verify the new outputs.

## 4. Formal Evaluation

### 4.1 Benchmark Setup
- [ ] Run the trained model on the `data/datasets/final_dataset/val` split.
- [ ] Implement Recall@1, Recall@3, and Recall@5 category-match metrics — since exact recipe matching is noisy, check if the retrieved recipe's raw text contains the category-specific keywords associated with the test image.
- [ ] Log and report the results as the official benchmark for the model.