import os
import shutil
from pathlib import Path

# --- Configuration Paths ---
BASE_DIR = Path("/home/s22imc10262/data/NLP/hackathon/data/datasets/")
FINAL_DIR = BASE_DIR / "final_dataset"
TRAIN_DIR = FINAL_DIR / "train"
VAL_DIR = FINAL_DIR / "val"
OUTPUT_DIR = BASE_DIR / "combined_dataset"

def setup_directories():
    """Creates the output directory and removes it if it exists."""
    if OUTPUT_DIR.exists():
        print(f"Removing existing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {OUTPUT_DIR}\n")

def merge_datasets():
    """Merges train and val datasets into a single combined dataset."""
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print(f"Error: Train ({TRAIN_DIR}) or Val ({VAL_DIR}) directory not found.")
        return
    
    # Get all categories from train directory
    categories = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    total_categories = len(categories)
    
    print(f"Found {total_categories} categories. Starting merge...\n")
    
    total_files_copied = 0
    
    for idx, category_dir in enumerate(categories, 1):
        category_name = category_dir.name
        output_category = OUTPUT_DIR / category_name
        
        # Create category directory in output
        output_category.mkdir(parents=True, exist_ok=True)
        
        # Copy files from train
        train_category = TRAIN_DIR / category_name
        if train_category.exists():
            for image_file in train_category.iterdir():
                if image_file.is_file():
                    dest_file = output_category / image_file.name
                    shutil.copy2(image_file, dest_file)
                    total_files_copied += 1
        
        # Copy files from val
        val_category = VAL_DIR / category_name
        if val_category.exists():
            for image_file in val_category.iterdir():
                if image_file.is_file():
                    dest_file = output_category / image_file.name
                    shutil.copy2(image_file, dest_file)
                    total_files_copied += 1
        
        num_files = len(list(output_category.iterdir()))
        print(f"[{idx}/{total_categories}] {category_name}: {num_files} images")
    
    print(f"\n✓ Merge completed successfully!")
    print(f"✓ Total files copied: {total_files_copied}")
    print(f"✓ Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    setup_directories()
    merge_datasets()
