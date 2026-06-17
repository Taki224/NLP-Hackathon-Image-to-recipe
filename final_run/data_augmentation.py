#!/usr/bin/env python
# coding: utf-8

# # DAR Data Augmentation Pipeline
# 
# This notebook implements the Data Augmentation for Recipe retrieval (DAR) framework.
# It uses:
# 1. **Ollama (Llama 3)** to synthesize 30-word visual imaginations of recipes.
# 2. **Segment Anything Model (SAM)** to crop food items out of the background.
# 
# We use the official Meta `segment_anything` package.



pass # pass # get_ipython().system('uv add segment_anything opencv-python requests tqdm pillow matplotlib')

import os
import urllib.request
from pathlib import Path

# Download SAM weights if they don't exist
weights_dir = Path("sam_weights")
weights_dir.mkdir(exist_ok=True)
sam_checkpoint = weights_dir / "sam_vit_h_4b8939.pth"
if not sam_checkpoint.exists():
    print("Downloading SAM ViT-H checkpoint (this will take a few minutes)...")
    urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", str(sam_checkpoint))
    print("Download complete.")


import json
import torch
import cv2
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from segment_anything import sam_model_registry, SamPredictor

# --- CONFIGURATION ---
DEBUG = False  # Set to False to process the entire dataset (Overnight run)
DEBUG_LIMIT = 5

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

PROJECT_ROOT = Path.cwd() if (Path.cwd() / 'pyproject.toml').exists() else Path.cwd().parent
DATASET_PATH = PROJECT_ROOT / 'final_run/data/datasets/paired_dataset_70k_train.json'

# Output matching what train_dar_74k.py expects
OUTPUT_JSON_PATH = PROJECT_ROOT / 'final_run/data/datasets/paired_dataset_dar_70k_train.json'
AUGMENTED_IMAGES_DIR = PROJECT_ROOT / 'final_run/data/images/sam_cropped'
AUGMENTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ## 1. Load Dataset & Models



# Load SAM
print("Loading Segment Anything Model...")
sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
sam.to(device=device)
predictor = SamPredictor(sam)

# Load Recipe Dataset
print(f"Loading dataset from {DATASET_PATH}...")
with open(DATASET_PATH, 'r') as f:
    dataset = json.load(f)

if DEBUG:
    print(f"[DEBUG MODE] Taking only the first {DEBUG_LIMIT} items.")
    dataset = dataset[:DEBUG_LIMIT]
else:
    print(f"Loaded {len(dataset)} pairs for full processing.")


# ## 2. Augmentation Functions



def augment_text_with_ollama(title, ingredients, instructions):
    """
    Uses Ollama to generate a ~30-word visual description of the final dish.
    """
    prompt = (
        f"Recipe Name: {title}\n"
        f"Ingredients: {ingredients}\n"
        f"Instructions: {instructions[:500]}...\n\n"
        "Based on the recipe above, write a brief, highly visual description (around 30 words) "
        "of what the final prepared dish looks like on a plate. "
        "Focus only on its visual appearance, colors, textures, and plating. "
        "Do not include the recipe instructions or greeting.\n"
        "Description:"
    )
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Ollama request failed: {e}")
        return f"{title} with {str(ingredients)[:100]}" # Fallback

def extract_food_with_sam(image_path, output_path):
    """
    Uses SAM to segment the central food object and crop the background.
    Returns True if successful.
    """
    if not Path(image_path).is_absolute():
        clean_path = str(image_path).replace('../../', '')
        image_path = str(PROJECT_ROOT / clean_path)
        
    if not Path(image_path).exists():
        return False

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Predict using the center point of the image as the positive prompt
    predictor.set_image(image_rgb)
    h, w = image.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1]) # Positive point
    
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    # Pick the mask with the highest score
    best_mask = masks[np.argmax(scores)]
    
    # Create an image with a white background where the mask is False
    result_img = image_rgb.copy()
    result_img[~best_mask] = [255, 255, 255] # White out background
    
    # Save the augmented image
    result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), result_bgr)
    return True


import concurrent.futures
from threading import Lock

sam_lock = Lock()

# Load existing progress if available
existing_data = []
processed_ids = set()
if OUTPUT_JSON_PATH.exists():
    try:
        with open(OUTPUT_JSON_PATH, 'r') as f:
            existing_data = json.load(f)
            processed_ids = {str(item.get('recipe_id', '')) for item in existing_data}
        print(f"Resuming: Loaded {len(processed_ids)} already processed items.")
    except Exception as e:
        print(f"Could not load existing checkpoint: {e}. Starting fresh.")
        existing_data = []
        processed_ids = set()

augmented_dataset = list(existing_data)

# Filter dataset to skip processed items
filtered_dataset = [item for item in dataset if str(item.get('recipe_id', '')) not in processed_ids]
print(f"Total dataset: {len(dataset)} | Remaining to process: {len(filtered_dataset)}")

def process_item(item):
    orig_img_path = item['image_path']
    recipe_id = str(item.get('recipe_id', ''))
    title = item.get('recipe_title') or item.get('recipe_name') or ''
    ingredients = item.get('ingredients', '')
    instructions = item.get('directions', '') # Try getting instructions if available in item
    
    # Define output path for SAM image
    img_filename = Path(orig_img_path).name
    aug_img_path = AUGMENTED_IMAGES_DIR / f"sam_{img_filename}"
    
    # 1. Image Augmentation (Thread-safe lock on GPU predictor)
    # Check if SAM cropped image already exists on disk to save time
    if not aug_img_path.exists():
        with sam_lock:
            success = extract_food_with_sam(orig_img_path, aug_img_path)
        if not success:
            return None
        
    # 2. Text Augmentation (Concurrent local LLM requests)
    aug_text = augment_text_with_ollama(title, ingredients, instructions)
    
    # Compile
    return {
        "recipe_id": recipe_id,
        "image_path": orig_img_path,
        "aug_image_path": str(aug_img_path.absolute()),
        "recipe_title": title,
        "ingredients": ingredients,
        "aug_visual_text": aug_text
    }

print("Starting parallel augmentation pipeline...")
max_workers = 4  # Scale to balance CPU/GPU and Ollama concurrency

if len(filtered_dataset) > 0:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, item) for item in filtered_dataset]
        
        save_counter = 0
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Augmenting pairs"):
            result = future.result()
            if result is not None:
                augmented_dataset.append(result)
                save_counter += 1
                
                # Checkpoint every 100 items
                if save_counter % 100 == 0:
                    with open(OUTPUT_JSON_PATH, 'w') as f:
                        json.dump(augmented_dataset, f, indent=2)

# Save final dataset
with open(OUTPUT_JSON_PATH, 'w') as f:
    json.dump(augmented_dataset, f, indent=2)

print(f"\nPipeline complete. Augmented dataset saved to {OUTPUT_JSON_PATH}")

