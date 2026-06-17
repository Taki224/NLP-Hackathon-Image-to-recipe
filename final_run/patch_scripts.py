import re
import os

def patch_script(filename, dataset_name, checkpoint_suffix, is_dar=False, is_aug=False):
    with open(filename, 'r') as f:
        content = f.read()

    # Disable jupyter magics
    content = re.sub(r'get_ipython\(\)', 'pass # get_ipython()', content)
    
    # Force PROJECT_ROOT to be the parent directory of final_run, but all outputs go to final_run
    
    if is_aug:
        # Patch data_augmentation.py
        content = re.sub(r'DEBUG\s*=\s*True', 'DEBUG = False', content)
        content = re.sub(r'DATASET_PATH\s*=\s*PROJECT_ROOT / .*?\.json\'', f"DATASET_PATH = PROJECT_ROOT / 'final_run/data/datasets/{dataset_name}'", content)
        content = re.sub(r'OUTPUT_JSON_PATH\s*=\s*Path\(.*?\)', f"OUTPUT_JSON_PATH = PROJECT_ROOT / 'final_run/data/datasets/paired_dataset_dar_70k_train.json'", content)
    else:
        # Fix dataloader deadlocks
        content = re.sub(r'NUM_WORKERS\s*=\s*.*', 'NUM_WORKERS = 0', content)
        
        # Replace dataset paths to use the train splits inside final_run
        content = re.sub(r'(?m)^DATASET_PATH\s*=.*$', f"DATASET_PATH = PROJECT_ROOT / 'final_run/data/datasets/{dataset_name}'", content)
        content = re.sub(r'(?m)^PAIRED_DATASET_PATH\s*=.*$', f"PAIRED_DATASET_PATH = PROJECT_ROOT / 'final_run/data/datasets/{dataset_name}'", content)
        content = re.sub(r"str\(PROJECT_ROOT / 'data/datasets/paired_dataset.json'\)", f"str(PROJECT_ROOT / 'final_run/data/datasets/{dataset_name}')", content)
        
        # Make all output directories point inside final_run
        
        # Phase 1 uses CHECKPOINT_PATH directly instead of CHECKPOINT_DIR
        content = re.sub(r"CHECKPOINT_PATH\s*=\s*PROJECT_ROOT / 'models/checkpoints/best_model\.pt'", f"CHECKPOINT_PATH = PROJECT_ROOT / 'final_run/models/checkpoints/{checkpoint_suffix}/best_model.pt'", content)
        
        # Phase 2 uses CHECKPOINT_DIR
        content = re.sub(r"CHECKPOINT_DIR\s*=\s*PROJECT_ROOT / 'models/checkpoints/phase2_8k'", f"CHECKPOINT_DIR = PROJECT_ROOT / 'final_run/models/checkpoints/{checkpoint_suffix}'", content)
        content = re.sub(r"CHECKPOINT_DIR\s*=\s*PROJECT_ROOT / 'models/checkpoints'", f"CHECKPOINT_DIR = PROJECT_ROOT / 'final_run/models/checkpoints/{checkpoint_suffix}'", content)
        content = re.sub(r"CHECKPOINT_DIR\s*=\s*Path\.cwd\(\) / 'checkpoints'", f"CHECKPOINT_DIR = PROJECT_ROOT / 'final_run/models/checkpoints/{checkpoint_suffix}'", content)
        
        # Reports
        content = re.sub(r"REPORTS_DIR\s*=\s*Path\.cwd\(\) / 'reports'", f"REPORTS_DIR = PROJECT_ROOT / 'final_run/reports'", content)
        content = re.sub(r"save_dir=str\(PROJECT_ROOT / 'reports'\)", f"save_dir=str(PROJECT_ROOT / 'final_run/reports')", content)
        
        # Figures
        content = re.sub(r"FIGURES_DIR\s*=\s*PROJECT_ROOT / 'reports/figures'", f"FIGURES_DIR = PROJECT_ROOT / 'final_run/reports/figures'", content)
        
        # Indexes (Prevent them from overwriting each other)
        content = re.sub(r"INDEX_PATH\s*=\s*PROJECT_ROOT / 'data/indexes/recipe_index\.npy'", f"INDEX_PATH = PROJECT_ROOT / 'final_run/data/indexes/recipe_index_{checkpoint_suffix}.npy'", content)
        content = re.sub(r"INDEX_META_PATH\s*=\s*PROJECT_ROOT / 'data/indexes/recipe_index_metadata\.csv'", f"INDEX_META_PATH = PROJECT_ROOT / 'final_run/data/indexes/recipe_index_metadata.csv'", content)
        content = re.sub(r"INDEX_IDS_PATH\s*=\s*PROJECT_ROOT / 'data/indexes/recipe_index_ids\.npy'", f"INDEX_IDS_PATH = PROJECT_ROOT / 'final_run/data/indexes/recipe_index_ids_{checkpoint_suffix}.npy'", content)
        
        # Ensure it trains for 30 epochs
        content = re.sub(r'(?m)^NUM_EPOCHS\s*=.*$', 'NUM_EPOCHS = 30', content)
        
    with open(filename, 'w') as f:
        f.write(content)

# 1. Baseline
patch_script('train_phase1.py', 'paired_dataset_train.json', 'phase1')
# 2. Phase 2 8k
patch_script('train_phase2_8k.py', 'paired_dataset_8k_train.json', 'phase2_8k')
# 3. Phase 2 74k
patch_script('train_phase2_74k.py', 'paired_dataset_70k_train.json', 'phase2_74k')
# 4. DAR Augmentation (run on 70k)
patch_script('data_augmentation.py', 'paired_dataset_70k_train.json', '', is_aug=True)
# 5. DAR Train 74k (run on output of augmentation)
patch_script('train_dar_74k.py', 'paired_dataset_dar_70k_train.json', 'dar_74k', is_dar=True)

print("Patching complete.")
