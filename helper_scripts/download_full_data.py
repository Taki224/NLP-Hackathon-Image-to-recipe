import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# DATA_ROOT env var, default = <repo>/data/datasets
DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[1] / "data" / "datasets"))

def save_food101_images():
    print("Downloading/Loading Food101 dataset from HuggingFace...")
    dataset = load_dataset('food101')

    # merger.py expects food101 images under DATA_ROOT/full_data_food101/<split>/<cat>/
    output_dir = str(DATA_ROOT / "full_data_food101")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Exporting raw image files to {output_dir}...")
    
    for split in ['train', 'validation']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Iterate through the dataset and save each image
        for idx, item in enumerate(tqdm(dataset[split], desc=f"Exporting {split} split")):
            image = item['image']
            label = item['label']
            
            # Convert integer label to actual category name (e.g., 'apple_pie')
            category_name = dataset[split].features['label'].int2str(label)
            
            # Create a folder for the specific category inside the split directory
            cat_dir = os.path.join(split_dir, category_name)
            os.makedirs(cat_dir, exist_ok=True)
            
            # Ensure the image is in RGB before saving as JPEG
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Save the image to disk permanently
            img_path = os.path.join(cat_dir, f"{category_name}_{idx}.jpg")
            image.save(img_path)
            
    print(f"\nExport complete! All 101,000 images are permanently saved in '{output_dir}'.")

if __name__ == "__main__":
    save_food101_images()