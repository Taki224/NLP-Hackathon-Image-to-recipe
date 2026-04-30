# Project Challenges and Issues

This document tracks the technical challenges, hurdles, and solutions encountered during the development of the recipe matching project.

## 1. The "77-Token" Wall (Standard CLIP Text Limit)

**The Problem:**
Standard CLIP models have a hard sequence limit of 77 tokens for text inputs. Most of our training data recipes (Title + Ingredients + Instructions) are significantly longer than 77 tokens. If the full recipe is piped directly into CLIP, it truncates early, often cutting off after the first few ingredients. This means the model completely ignores the rest of the recipe, dropping valuable semantic information.

**The Solution:**
Use **LongCLIP** instead of standard CLIP, which natively supports larger text token windows. This allows us to:

1. **Full Recipe Text:** Include the complete recipe (Title + Ingredients + Instructions) without truncation.
2. **Enhanced Semantic Information:** The model now has access to cooking instructions and full ingredient lists, enabling better understanding of the dish.
3. **Improved Text Encoder:** LongCLIP's text encoder can handle the full context, leading to richer embeddings for recipe-image matching.

## 2. Granularity of Food Categories

**The Problem:**
There are not enough specific granular food categories, which leads to poor representation of similar but distinct dishes. For example, testing an image of "Goulash" resulted in a prediction/retrieval of "Mediterranean Veggie Soup." Because the category definitions are too broad or missing regional/specific depths, the model fails to discriminate fine details and instead falls back to visually generic classes.

**Potential Solutions to Explore:**
- **Refining Category Definitions:** Investigate adding deeper, more specific categories for underrepresented Cuisines.
- **Hierarchical Matching:** Implement a coarse-to-fine approach where the model first identifies the broad category (e.g., Soup/Stew) and then a specific dish.
- **Ingredient-Level Analysis:** Put more weight on ingredient detection and matching to differentiate visually similar dishes that contain different core ingredients (e.g., Goulash uses beef/paprika vs Mediterranean Veggie Soup).

## 3. Information Loss from Post-Model Adapters

**The Problem:**
The primary issue with using a post-model adapter for recipe matching is information loss at the bottleneck. Since a recipe contains complex, long-form data (ingredients, quantities, and techniques), squeezing all that nuance through a simple MLP after the encoder forces the model to discard fine-grained details that are essential for distinguishing between similar dishes.

**Potential Solutions to Explore:**
- **Unfreezing Encoder Layers:** Unfreeze the last few layers of the base CLIP text and image encoders so domain-specific nuanced data alters the base representations, preventing the bottleneck.
- **Cross-Attention Mechanisms:** Instead of a simple MLP, utilize cross-attention layers that can attend to specific sequences (like individual ingredients) when matching against image regions.

## 4. Data & Dataset Challenges
- **Merged train/val split:** Instead of maintaining separate train/val sets, we merged them into a single `combined_dataset` for training, simplifying the pipeline and utilizing all available data.
- **Recipe duplication per category:** With only 10–50 matching recipes per category and 750 images, the same recipe was assigned to hundreds of images, making training pairs heavily repetitive. *Solution:* Assign 3–5 recipes per image to generate ~375k pairs from combined dataset.
- **Title-only recipe matching:** Searching only recipe titles missed valid matches (e.g., a recipe titled "Sunday Dinner" containing pasta). The search had to be extended to the ingredients column.
- **Only 1 recipe per image:** Single positive pairs limited dataset diversity. The solution was assigning 3–5 recipes per image to generate more diverse training pairs.
- **No deduplication:** The same recipe could match an image multiple times if it hit on both the title and ingredients simultaneously. *Solution:* Use `seen_recipe_ids` set to deduplicate.
- **Noisy recipe candidates:** Bad keyword search matches (e.g., frosting for cakes) corrupted training pairs. *Solution:* Multi-stage filtering with LongCLIP + SigLIP + Vision-Language LLM validation.

## 4.1 Dataset Scale & Duplication Challenges (Early Planning Issue)

**The Problem (During Initial Planning):**
The initial 100-image-per-category artificial cap limited the dataset to ~5,000 images total. With only 10–50 matching recipes per category and ~750 images in the full dataset initially, the same recipe was being assigned to hundreds of images, creating extremely repetitive training pairs. This repetition meant the model saw the same recipe-image associations repeatedly, limiting diversity and generalization.

**The Solution:**
1. **Expanded Data Source:** Moved from the limited initial dataset to using the full Food101-based combined dataset (150k+ images after merging train/val splits).
2. **Multi-Recipe Positive Pairs:** Instead of assigning 1 recipe per image, now assign 3–5 recipes per image, creating diverse training examples even when the recipe pool is limited.
3. **Quality-Scale Balance:** Apply aggressive multi-stage filtering (LongCLIP + SigLIP + VL-LLM) to ensure that even at larger scale, we're training on verified high-quality pairs, not just more volume.
4. **Efficient Deduplication:** Track seen recipes per image with `seen_recipe_ids` to eliminate redundant assignments within each image's positive pair set.

## 4.2 Category-Biased Data Split Challenge (Early Planning Issue)

**The Problem (During Initial Planning):**
When splitting the dataset into train and val, doing so without proper shuffling first would result in all images of certain categories ending up exclusively in one split. This would leave the validation set without coverage for those categories, making validation unreliable and training biased.

**The Solution:**
By merging the entire dataset into `combined_dataset` without maintaining train/val splits, we eliminate this bias issue entirely. The model now trains on all available images across all categories, and manual testing with known image-recipe pairs provides a simpler, more flexible validation approach that doesn't require careful split management.

## 5. Training Challenges
- **Single combined dataset:** Training now uses the merged dataset without separate train/val splits, simplifying the workflow.
- **Loss tracking:** The model tracks and logs training loss throughout training with periodic checkpoints.
- **Loss not converging:** The model may need 50+ epochs to converge; training is configured accordingly.
- **Manual validation approach:** Instead of automated validation metrics, we use manual testing with known image-recipe pairs to assess model quality after training.

## 5.1 Validation Split Challenges (Early Planning Issue)

**The Problem (During Initial Planning):**
Early training only tracked training loss, providing no signal on model generalization. Stopping based on training loss is unreliable, and without a validation split, we couldn't detect overfitting. Additionally, the model needed 50+ epochs to converge, but early runs were stopped far too early without clear stopping criteria.

**The Solution:**
Instead of maintaining separate train/val splits with automated early stopping, we now:

1. **Merged Dataset Approach:** Combined train and val into a single `combined_dataset`, maximizing training data availability.
2. **Extended Training:** Configure training for 50+ epochs from the start, allowing the model sufficient time to converge.
3. **Manual Testing:** After training, use manual validation with known image-recipe pairs to assess model quality, avoiding the complexity of per-epoch validation loops.
4. **Checkpoint Intervals:** Save model checkpoints at regular intervals so we can analyze training progression and select the best performing checkpoint.

## 5.2 Dataset Scale Challenges (Early Planning Issue)

**The Problem (During Initial Planning):**
The initial dataset was artificially limited to 5,000 images total (100 per category cap); we needed to scale to the full combined Food101 dataset (~150k+ images when merged). With only 10–50 matching recipes per category and the smaller dataset, the same recipe was assigned to hundreds of images, making training pairs heavily repetitive.

**The Solution:**
1. **Full Dataset Integration:** Merged `final_dataset/train` and `final_dataset/val` into `combined_dataset`, utilizing all available images.
2. **Multi-Recipe Assignment:** Assign 3–5 matching recipes per image instead of just 1, generating diverse positive pairs (~375k+ pairs from the combined dataset).
3. **Quality Filtering:** Use LongCLIP + SigLIP + Vision-Language LLM multi-stage filtering to ensure that even with a larger dataset, the pairs are high-quality (not just numerically scaled bad data).
4. **Deduplication:** Use `seen_recipe_ids` to prevent the same recipe from being assigned to the same image multiple times, reducing repetition.

## 6. Retrieval & Model Challenges
- **Out-of-distribution images:** Uploading a dish not in Food101 (e.g., goulash) causes the model to silently return wrong-category results instead of signaling uncertainty. *Solution:* Implement a similarity confidence threshold to trigger a warning.
- **Ingredient text is noisy:** Food.com ingredients include quantities and units (e.g., "2 cups flour"), making raw ingredient confidence scores unreadable. *Solution:* Needs regex parsing to extract just the ingredient name.
- **No formal evaluation metric:** No Recall@1/3/5 metrics were implemented yet, making model quality objectively hard to measure.

## 7. Dataset Acquisition Challenges
- **ISIA Food-500 server is extremely slow:** Download speeds maxed out at ~400 KB/s on a massive 43GB dataset, accompanied by frequent connection drops mid-download.

## 8. Noisy Keyword Search Results

**The Problem:**
The keyword search returns a noisy pool of recipe candidates per category — some don't actually represent the dish. For example, frosting recipes match to "chocolate cake", and sauce recipes match to "pasta". Since your training pairs come from this keyword search pool, bad matches mean the contrastive loss is trained on false positives, which corrupts the embedding space and degrades model discrimination.

**The Solution:**
Implement a **multi-stage filtering pipeline** to progressively refine pair quality:

### Stage 1: LongCLIP Text-to-Text Ranking
After keyword search, run a LongCLIP text-to-text ranking step as a post-processing filter:

1. **Canonical Category Descriptions:** For each category, create a brief description (e.g., "a dish of chocolate cake, showing its typical appearance and ingredients").
2. **LongCLIP Text Ranking:** Score each candidate recipe against the canonical description using LongCLIP's text encoder (via cosine similarity).
3. **Hard Filter:** Immediately reject recipes whose titles contain exclusionary words (e.g., "frosting", "glaze", "sauce") when those ingredients don't belong to the category.
4. **Top-K Selection:** Keep only the top 5 candidates by text-to-text cosine similarity and discard the rest.

### Stage 2: SigLIP Image-Recipe Pair Scoring
Score each image-recipe pair using SigLIP (a more efficient image-text model):

1. **Compute Pair Scores:** Encode images and recipe text, then compute cosine similarity for each pair.
2. **Per-Category Thresholds:** Determine thresholds based on score distributions (e.g., 75th percentile).
3. **Filter Below Threshold:** Keep only high-confidence pairs that exceed their category's threshold.

### Stage 3: Vision-Language LLM Validation
For all pairs that pass SigLIP filtering, use a Vision-Language LLM for final validation:

1. **Plausibility Assessment:** The VL-LLM evaluates whether the recipe genuinely matches the food shown in the image.
2. **Quality Confirmation:** Keep only pairs that pass VL-LLM validation; discard those that fail.
3. **Final Training Data:** The resulting pairs are high-quality and ready for model training.

This three-stage approach ensures that only reliable, multi-model-validated pairs make it into the final training dataset.

## 9. Low Yield of High-Quality Training Pairs

**The Problem:**
After applying the strict multi-stage filtering (SigLIP threshold > 0.5 + VLM validation), the final dataset yield is fundamentally low. We ended up with exactly 8,160 high-quality pairs across 403 categories, resulting in an average of ~20 pairs per category and a median of just 10.

**The Impact & Solutions:**
- **Overfitting Risk:** With such a limited amount of training data, unfreezing the base layers of a massive model like LongCLIP will inevitably lead to severe overfitting. 
- **Lightweight Adapters:** We must abandon unfreezing the base encoders. Instead, we should rely on highly parameter-efficient fine-tuning methods. Exploring a bottleneck MLP adapter (e.g., 512 -> 64 -> 512) or a TIP-Adapter will allow domain adaptation without destroying the base model's existing generalization.
- **Hard Mining:** Since the overall volume is low, maximizing the signal-to-noise ratio in every batch is paramount. Training with hard negative and hard positive pairs will challenge the model to learn fine-grained discriminative features (like determining Goulash vs. typical Beef Stew) despite the smaller data pool.

## 10. Fixing the Data Yield & Checkpointing Flaws

**The Problem:**
When trying to run the long-running SigLIP + VLM stages, the notebook occasionally crashed. The initial checkpointing implementation failed to resume seamlessly because the prior step generating the candidate `paired_dataset` used non-deterministic random sampling. Upon a kernel restart, it generated a completely different set of random pairs, causing the saved checkpoint dictionary keys to mismatch. Furthermore, the strict 500-recipe cap and strict zero-reuse deduplication were artificially choking the yield.

**The Solution:**
By removing the 500-recipe pool cap in the regex search and allowing high-quality recipes to be reused up to 5 times across different images, we prevented early pool exhaustion. After running the stabilized pipeline, we successfully generated **~74,000 high-quality image-recipe pairs**.

**The Impact:**
With 74k high-quality pairs, the severe overfitting risk encountered when we only had 8k pairs (Issue #9) is mitigated. We now have sufficient data volume to safely resume experiments with unfreezing the last few layers of the base LongCLIP text and image encoders, giving us the flexibility to compare full parameter fine-tuning against parameter-efficient adapter methods.