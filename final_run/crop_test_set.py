import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent
FINAL_RUN_DIR = PROJECT_ROOT / 'final_run'
TEST_DATA_PATH = FINAL_RUN_DIR / 'data/datasets/test_dataset.json'
OUTPUT_JSON_PATH = FINAL_RUN_DIR / 'data/datasets/test_dataset_cropped.json'
CROPPED_IMAGES_DIR = FINAL_RUN_DIR / 'data/images/sam_cropped_test'
CROPPED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Load SAM
sam_checkpoint = FINAL_RUN_DIR / "sam_weights" / "sam_vit_h_4b8939.pth"
print("Loading SAM model...")
sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
sam.to(device=device)
predictor = SamPredictor(sam)

def extract_food_with_sam(image_path, output_path):
    if not Path(image_path).is_absolute():
        clean_path = str(image_path).replace('../../', '')
        image_path = str(PROJECT_ROOT / clean_path)
        
    if not Path(image_path).exists():
        return False

    image = cv2.imread(image_path)
    if image is None:
        return False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    predictor.set_image(image_rgb)
    h, w = image.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])
    
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    best_mask = masks[np.argmax(scores)]
    result_img = image_rgb.copy()
    result_img[~best_mask] = [255, 255, 255] # White background
    
    result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), result_bgr)
    return True

print("Loading test dataset...")
with open(TEST_DATA_PATH, 'r') as f:
    test_data = json.load(f)

cropped_test_data = []

print("Generating SAM crops for test set...")
for item in tqdm(test_data):
    orig_path = item['image_path']
    img_filename = Path(orig_path).name
    output_img_path = CROPPED_IMAGES_DIR / f"sam_test_{img_filename}"
    
    success = True
    if not output_img_path.exists():
        success = extract_food_with_sam(orig_path, output_img_path)
        
    if success:
        # Build relative path for compatibility
        rel_path = f"final_run/data/images/sam_cropped_test/{output_img_path.name}"
        item_cropped = item.copy()
        item_cropped['image_path'] = rel_path
        cropped_test_data.append(item_cropped)
    else:
        # Fallback to original if SAM fails
        cropped_test_data.append(item)

with open(OUTPUT_JSON_PATH, 'w') as f:
    json.dump(cropped_test_data, f, indent=2)

print(f"Cropped test dataset saved to {OUTPUT_JSON_PATH}")
