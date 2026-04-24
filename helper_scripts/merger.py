import os
import shutil
from pathlib import Path

# --- Configuration Paths ---
# Adjust these paths if you run the script from a different directory
BASE_DIR = Path("/home/s22imc10262/data/NLP/hackathon/data/datasets")
FOOD101_DIR = BASE_DIR / "full_data_food101"
ISIA500_DIR = BASE_DIR / "full_data_isia-food-500" / "images"
OUTPUT_DIR = BASE_DIR / "merged_dataset"

# --- Mapping Dictionary ---
# Maps Dataset 1 (Food-101) folder names to Dataset 2 (ISIA-500) folder names
MATCHED_CLASSES = {
    "apple_pie": "Apple_pie",
    "beignets": "Beignet",
    "bibimbap": "Bibimbap",
    "bruschetta": "Bruschetta",
    "caesar_salad": "Caesar_salad",
    "caprese_salad": "Caprese_salad",
    "carrot_cake": "Carrot_cake",
    "ceviche": "Ceviche",
    "cheesecake": "Cheesecake",
    "chicken_curry": "Chicken_curry",
    "churros": "Churro",
    "clam_chowder": "Clam_chowder",
    "club_sandwich": "Club_sandwich",
    "crab_cakes": "Crab_cake",
    "donuts": "Doughnut",
    "edamame": "Edamame",
    "filet_mignon": "Filet_mignon",
    "french_onion_soup": "French_onion_soup",
    "french_toast": "French_toast",
    "fried_rice": "Fried_rice",
    "garlic_bread": "Garlic_bread",
    "greek_salad": "Greek_salad",
    "hamburger": "hamburgers",
    "hot_and_sour_soup": "Hot_and_sour_soup",
    "huevos_rancheros": "Huevos_rancheros",
    "hummus": "Hummus",
    "lasagna": "Lasagna",
    "lobster_bisque": "Lobster_bisque",
    "macarons": "Macaron",
    "nachos": "Nachos",
    "onion_rings": "Onion_ring",
    "paella": "Paella",
    "pho": "Pho",
    "poutine": "Poutine",
    "ramen": "Ramen",
    "samosa": "Samosa",
    "sashimi": "Sashimi",
    "tacos": "Tacos",
    "takoyaki": "Takoyaki",
    "tiramisu": "Tiramisu"
}

# Create a reverse mapping for easy ISIA-500 lookup (ISIA_Name -> Food101_Name)
REVERSE_MAPPING = {v: k for k, v in MATCHED_CLASSES.items()}

def setup_output_dir():
    if OUTPUT_DIR.exists():
        print(f"Warning: Output directory {OUTPUT_DIR} already exists. Merging into it.")
    else:
        OUTPUT_DIR.mkdir(parents=True)

def merge_food101():
    print("--- Merging Food-101 (Train and Validation) ---")
    splits = ["train", "validation"]
    
    for split in splits:
        split_dir = FOOD101_DIR / split
        if not split_dir.exists():
            print(f"Skipping {split_dir}, does not exist.")
            continue
            
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            target_dir = OUTPUT_DIR / class_name
            target_dir.mkdir(exist_ok=True)
            
            # Copy images and prefix them to avoid train/val name collisions
            for img_path in class_dir.iterdir():
                if img_path.is_file():
                    new_img_name = f"food101_{split}_{img_path.name}"
                    shutil.copy2(img_path, target_dir / new_img_name)
                    
        print(f"Processed Food-101: {split}")

def merge_isia500():
    print("--- Merging ISIA-Food-500 ---")
    if not ISIA500_DIR.exists():
        print(f"Directory {ISIA500_DIR} not found. Exiting.")
        return

    for class_dir in ISIA500_DIR.iterdir():
        if not class_dir.is_dir():
            continue
            
        isia_class_name = class_dir.name
        
        # Check if this ISIA-500 class has a mapped Food-101 equivalent
        if isia_class_name in REVERSE_MAPPING:
            target_class_name = REVERSE_MAPPING[isia_class_name]
        else:
            # If no strict match, keep the ISIA-500 original name
            target_class_name = isia_class_name
            
        target_dir = OUTPUT_DIR / target_class_name
        target_dir.mkdir(exist_ok=True)
        
        # Copy images and prefix them to avoid collisions with Food-101 files
        for img_path in class_dir.iterdir():
            if img_path.is_file():
                new_img_name = f"isia500_{img_path.name}"
                shutil.copy2(img_path, target_dir / new_img_name)
                
    print("Processed ISIA-Food-500")

def main():
    setup_output_dir()
    merge_food101()
    merge_isia500()
    print(f"\nMerge complete! All data is located in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()