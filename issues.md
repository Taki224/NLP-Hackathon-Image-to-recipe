# Project Challenges and Issues

This document tracks the technical challenges, hurdles, and solutions encountered during the development of the recipe matching project.

## 1. The "77-Token" Wall (Standard CLIP Text Limit)

**The Problem:**
Standard CLIP models have a hard sequence limit of 77 tokens for text inputs. Most of our training data recipes (Title + Ingredients + Instructions) are significantly longer than 77 tokens. If the full recipe is piped directly into CLIP, it truncates early, often cutting off after the first few ingredients. This means the model completely ignores the rest of the recipe, dropping valuable semantic information.

**The Solution:**
Instead of blindly passing the full recipe text and letting the tokenizer truncate it arbitrarily, we implemented a workaround focusing on "Global" vs. "Local" features (Strategy A):

1. **Data Formatting:** Modified the data creation pipeline to only include the *Title and Ingredients* for each recipe, deliberately excluding the instructions. These components represent the most visually descriptive parts of the recipe that map well to the images.
2. **Text Adapter Update:** Updated the `text_adapter` and DataLoader tokenizer logic to enforce the 77-token limit cleanly on this prioritized text subset.

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
- **100-image-per-category cap:** The initial dataset was artificially limited to 5,000 images total; needed to scale to the full Food101 dataset (~75k images).
- **Recipe duplication per category:** With only 10–50 matching recipes per category and 750 images, the same recipe was assigned to hundreds of images, making training pairs heavily repetitive.
- **Title-only recipe matching:** Searching only recipe titles missed valid matches (e.g., a recipe titled "Sunday Dinner" containing pasta). The search had to be extended to the ingredients column.
- **Only 1 recipe per image:** Single positive pairs limited dataset diversity. The solution was assigning 3–5 recipes per image to generate ~375k pairs from 75k images.
- **No deduplication:** The same recipe could match an image multiple times if it hit on both the title and ingredients simultaneously.
- **Category-biased train/val split:** Splitting without shuffling first would put all images of certain categories exclusively into the train set.

## 5. Training Challenges
- **No validation split:** Early training only tracked training loss, providing no signal on model generalization.
- **Early stopping on train loss:** Stopping based on training loss is unreliable; the stopping criterion needed to be switched to validation loss.
- **Loss not converging:** The model needed 50+ epochs to converge, meaning initial 20-epoch runs were stopped far too early.

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
After keyword search, run a CLIP text-to-text ranking step as a post-processing filter in the data creation pipeline:

1. **Canonical Category Descriptions:** For each category, create a brief description (e.g., "a dish of chocolate cake, showing its typical appearance and ingredients").
2. **CLIP Text Ranking:** Score each candidate recipe against the canonical description using CLIP's text encoder only (via cosine similarity).
3. **Hard Filter:** Immediately reject recipes whose titles contain exclusionary words (e.g., "frosting", "glaze", "sauce") when those ingredients don't belong to the category.
4. **Top-K Selection:** Keep only the top 5 candidates by text-to-text cosine similarity and discard the rest.

This requires no images, runs fast on the L40S, and fits directly into `data_creator.ipynb` as a post-processing step after existing keyword search.