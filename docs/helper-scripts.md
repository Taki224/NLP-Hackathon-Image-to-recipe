# Helper Scripts

<!-- GSD-GENERATED -->

Standalone Python scripts for raw data acquisition and dataset preparation. Located in `helper_scripts/`.

Run any script with:
```bash
uv run python helper_scripts/<script>.py
```

## Scripts

### `download_full_data.py`

Downloads Food101 from HuggingFace `datasets` and saves all images to disk as JPEG files organized by category.

**Output:** `{output_dir}/{split}/{category}/{category}_{idx}.jpg`

**Note:** `output_dir` is hardcoded to an HPC path. Update it before running locally:
```python
output_dir = "/your/local/path"
```

---

### `recipe_download.py`

Downloads the 2M+ recipe dataset (`wilmerarltstrmberg/recipe-dataset-over-2m`) from Kaggle via `kagglehub` and saves it as `output/recipe_dataset_2m.csv`.

**Requires:** Kaggle API credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`)

**Output:** `output/recipe_dataset_2m.csv`

---

### `download_isia500.py`

Downloads ISIA Food-500 (~43 GB, 11 split zip files) from the ICT server with resume support and automatic retry on failure.

**Output:** `{DEST_DIR}/ISIA_Food500.z01` … `ISIA_Food500.zip` + extracted `images/`

**Note:** `DEST_DIR` is hardcoded. Update before running:
```python
DEST_DIR = Path("/your/local/path")
```

Prompts to extract after download completes. To extract manually:
```bash
cd <DEST_DIR> && unzip ISIA_Food500.zip -d images/
```

---

### `merger.py`

Merges Food101 and ISIA Food-500 into a single unified dataset. Maps ISIA-500 class names to Food101 equivalents using a hardcoded 40-class dictionary. Images are prefixed (`food101_` or `isia500_`) to avoid filename collisions.

**Input:** `{BASE_DIR}/full_data_food101/`, `{BASE_DIR}/full_data_isia-food-500/images/`
**Output:** `{BASE_DIR}/merged_dataset/`

**Note:** `BASE_DIR` is hardcoded. Update before running.

The training notebook reads `combined_dataset` and handles its own train/val split, so just copy the merged output: `cp -r <base>/merged_dataset <base>/combined_dataset`.

## Typical Execution Order

```
download_full_data.py      # Food101 images
download_isia500.py        # ISIA Food-500 images (optional, large)
recipe_download.py         # Recipe text data
merger.py                  # Merge image datasets -> merged_dataset
# then: cp -r <base>/merged_dataset <base>/combined_dataset
```

After this, run the data creation notebooks (`notebooks/phase2/data_creator.ipynb`) to build image-recipe pairs.
