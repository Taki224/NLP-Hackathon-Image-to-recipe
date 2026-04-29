# Tasks

## 1. Create new Training Data script (`notebooks/phase2/data_creator.ipynb`)

### 1.1 Data Loading & Setup
- [X] Parse `data/datasets/combined_dataset` folder (merged training and validation data).
- [X] Load the new 2M recipe dataset from `data/datasets/recipe_dataset_2m.csv` instead of RAW_recipes.csv.

### 1.2 Recipe Matching & Formatting
- [X] Sanitize directory/category names (e.g., replacing underscores with spaces, like `spaghetti_carbonara` -> `spaghetti carbonara`) to create reliable keyword mappings for recipe matching.
- [X] Broaden the recipe search beyond title-only matching to also search the ingredients column (recipes without keywords in the title but in ingredients should match).
- [X] Assign 3–5 matching recipes per image instead of 1 (creates 3-5 positive pairs per image instead of 1).
- [X] Deduplicate matched recipes per image using a `seen_recipe_ids` set, so the same recipe is not assigned to the same image twice even if it matches on both title and ingredients.
- [X] Format the text output for each recipe to include Title, Ingredients, and Instructions (full recipe text) - LongCLIP supports larger token windows.

### 1.2.1 SigLIP Image-Recipe Pair Scoring
- [X] Use SigLIP to score each image-recipe pair by encoding the image and recipe title + top ingridients that fit (not the text because it doesn't fit into SigLIP), then computing cosine similarity.
- [X] Determine per-category thresholds based on the distribution of similarity scores (e.g., 75th percentile or median).
- [X] Apply global minimum threshold (0.5) first, then per-category top-50% as a secondary rank filter.
- [X] Filter out pairs that fall below their category's threshold, keeping only high-confidence matches.
- [X] Log the threshold values and number of pairs retained per category.

### 1.2.2 Vision-Language LLM Validation
- [X] Run VLM validation only on pairs with SigLIP score in the "uncertain" range; auto-accept pairs scoring > 0.65 as they are already high confidence
- [X] The VL-LLM should assess whether the recipe is a plausible match for the food shown in the image, considering ingredients, cooking style, and presentation.
- [X] Keep pairs that pass the VL-LLM validation; discard those that fail.
- [X] Log validation results to understand which pairs the VL-LLM rejected and why.

### 1.3 Exporting & Reporting
- [X] Process the combined dataset and save high-quality pairs (after SigLIP and VL-LLM filtering) to a single JSON file (`paired_dataset.json`).
- [X] Add a reporting step to display categories that have too few pairs (e.g., <20) after filtering to allow for manual investigation and keyword adjustments.
- [X] Update dataset final saving logic to reflect the file output changes.

### 1.4 Metadata Lookup System (Full Recipe Retrieval)
- [X] Create a separate `recipe_metadata.json` file mapping `recipe_id → {title, ingredients, instructions}` (full recipe text).
- [X] During paired dataset creation, store only recipe IDs and short text (title + ingredients) in `paired_dataset.json` for model indexing.
- [X] Ensure `recipe_metadata.json` is loaded and available during retrieval without impacting model inference speed.
- [X] Validate that metadata lookup works correctly: given a recipe ID from retrieval results, full recipe text is correctly retrieved.

## 2. Validation and Early Stopping (`notebooks/phase2/train.ipynb`)

### 2.1 Dataset & Loader Adjustments
- [ ] Update `FoodRecipeDataset` instantiation to load the new `paired_dataset.json`.
- [ ] Create DataLoader for training on the combined dataset.
- [ ] Update the `text_adapter` and tokenizer logic to use LongCLIP's tokenizer for full recipe text (Title + Ingredients + Instructions).

### 2.2 Model Architecture Experiments
- [ ] Use LongCLIP as the base model to support full recipe text (title + ingredients + instructions).
- [ ] Verify full recipe text (title + ingredients + instructions) fits within LongCLIP's 248-token limit; truncate instructions if needed, prioritising title and ingredients
- [ ] Experiment with lightweight adapter architectures (e.g., a bottleneck MLP like 512 -> 64 -> 512) or a TIP-Adapter to prevent overfitting on the low-volume training data.
- [ ] Experiment with generating hard positive and negative pairs for training to improve model discrimination between similar dishes (e.g., Goulash vs Mediterranean soup) and maximize learning signal from the small dataset.

### 2.3 Training Loop & Metrics
- [ ] Track and log training loss throughout training.
- [ ] Save model checkpoints at regular intervals (e.g., every N epochs or when loss improves).
- [ ] Log and plot training loss progression.

### 2.4 Post-Training Tasks
- [ ] After training completes, re-run the Step 10 `retrieve()` sanity check with a few sample images to confirm the new model and updated index produce sensible results before hackathon day.
- [ ] Generate the recipe retrieval `.npy` index for all 2 million recipes after training is complete (batch loading to avoid RAM crashes).
- [ ] Update `retrieve()` function to accept recipe IDs from index and perform metadata lookup to return full recipe text (title, ingredients, instructions).
- [ ] Update `/retrieve` API endpoint in `ui/src/app.py` to include full recipe instructions in the JSON response alongside similarity scores.
- [ ] Update the UI (`ui/static/index.html`) to display full recipe instructions in recipe cards below the ingredients list.

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

## 4. Manual Testing & Validation

### 4.1 Manual Testing with Known Pairs
- [ ] Prepare a small set of image-recipe pairs with known correct matches.
- [ ] Index the recipes from these pairs into the model's retrieval index.
- [ ] For each test image, retrieve the top results and check if the model successfully finds the known paired recipe(s).
- [ ] Document which pairs were retrieved correctly and which were missed to identify any failure modes.