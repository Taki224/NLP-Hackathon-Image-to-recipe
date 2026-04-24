import os
import shutil
import random
from pathlib import Path

# --- Configuration Paths ---
BASE_DIR = Path("/data/s22imc10262/NLP/hackathon/data/datasets/")
INPUT_DIR = BASE_DIR / "merged_dataset"
OUTPUT_DIR = BASE_DIR / "final_dataset"
TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"

# Split ratio (80% train, 20% validation)
TRAIN_RATIO = 0.8
# Random seed for reproducible shuffling
RANDOM_SEED = 42 

def setup_directories():
    """Creates the necessary output directories."""
    if OUTPUT_DIR.exists():
        print(f"Warning: Output directory {OUTPUT_DIR} already exists. Writing into it.")
    
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

def split_dataset():
    """Iterates through the merged dataset and splits each class 80/20."""
    if not INPUT_DIR.exists():
        print(f"Error: Input directory {INPUT_DIR} not found. Please run the merge script first.")
        return

    # Set seed so you get the exact same random split if you run this again
    random.seed(RANDOM_SEED)

    # Get all category folders
    categories = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    total_categories = len(categories)
    
    print(f"Found {total_categories} categories. Starting the 80/20 split...\n")

    for idx, category_dir in enumerate(categories, 1):
        category_name = category_dir.name
        
        # Get all image files in this category
        images = [f for f in category_dir.iterdir() if f.is_file()]
        
        # Shuffle the list of images randomly
        random.shuffle(images)
        
        # Calculate the split index
        split_idx = int(len(images) * TRAIN_RATIO)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create class subdirectories in both train and val folders
        train_class_dir = TRAIN_DIR / category_name
        val_class_dir = VAL_DIR / category_name
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        
        # Copy the files
        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
            
        for img in val_images:
            shutil.copy2(img, val_class_dir / img.name)
            
        # Optional: Print progress every 20 categories to avoid console spam
        if idx % 20 == 0 or idx == total_categories:
            print(f"[{idx}/{total_categories}] Split '{category_name}': {len(train_images)} train, {len(val_images)} val")

def main():
    setup_directories()
    split_dataset()
    print(f"\n✅ Split complete! Your model-ready data is in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()