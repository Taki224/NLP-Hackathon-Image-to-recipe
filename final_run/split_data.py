import json
import random
from pathlib import Path
import os

def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    FINAL_RUN_DIR = PROJECT_ROOT / 'final_run'
    DATA_DIR = FINAL_RUN_DIR / 'data' / 'datasets'
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the strict 8k dataset to sample from
    dataset_8k_path = PROJECT_ROOT / 'data/datasets/paired_dataset_8k.json'
    with open(dataset_8k_path, 'r') as f:
        data_8k = json.load(f)
        
    print(f"Loaded {len(data_8k)} pairs from {dataset_8k_path}")
    
    # We want 500 test pairs
    TEST_SIZE = 500
    random.seed(42)
    test_data = random.sample(data_8k, TEST_SIZE)
    test_image_paths = set(item['image_path'] for item in test_data)
    test_recipe_ids = set(str(item['recipe_id']) for item in test_data)
    
    # Save the test split
    test_path = DATA_DIR / 'test_dataset.json'
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"Saved {len(test_data)} test pairs to {test_path}")
    
    # 2. Filter datasets to create train splits (excluding test items)
    datasets_to_split = [
        'paired_dataset.json',
        'paired_dataset_8k.json',
        'paired_dataset_70k.json'
    ]
    
    for filename in datasets_to_split:
        src_path = PROJECT_ROOT / 'data/datasets' / filename
        if not src_path.exists():
            print(f"Warning: {src_path} not found. Skipping.")
            continue
            
        with open(src_path, 'r') as f:
            full_data = json.load(f)
            
        train_data = []
        excluded = 0
        for item in full_data:
            if item['image_path'] in test_image_paths or str(item['recipe_id']) in test_recipe_ids:
                excluded += 1
            else:
                train_data.append(item)
                
        train_filename = filename.replace('.json', '_train.json')
        out_path = DATA_DIR / train_filename
        with open(out_path, 'w') as f:
            json.dump(train_data, f)
            
        print(f"Saved {len(train_data)} train pairs to {out_path} (Excluded {excluded} test items from {filename})")

if __name__ == '__main__':
    main()
