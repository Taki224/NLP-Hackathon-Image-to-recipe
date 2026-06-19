import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import open_clip
import numpy as np
from pathlib import Path
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PROJECT_ROOT = Path(__file__).parent.parent
FINAL_RUN_DIR = PROJECT_ROOT / 'final_run'
MANUAL_DATA_PATH = FINAL_RUN_DIR / 'data/manual_test_pairs.json'
REPORTS_DIR = FINAL_RUN_DIR / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RECIPE_METADATA_PATHS = [
    PROJECT_ROOT / 'data/indexes/recipe_metadata.json',
    PROJECT_ROOT / 'data/indexes/recipe_index_metadata_newdata3.json',
    PROJECT_ROOT / 'data/indexes/recipe_index_metadata_newdata2.json',
    PROJECT_ROOT / 'data/indexes/recipe_index_metadata_newdata.json',
]

def find_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None

RECIPE_METADATA_PATH = find_first_existing(RECIPE_METADATA_PATHS)

def token_count(text, context_length=512):
    try:
        tokens = open_clip.tokenize([text], context_length=context_length)
    except TypeError:
        tokens = open_clip.tokenize([text])
    return int((tokens[0] != 0).sum().item())

def normalize_ingredients(ingredients, max_ingredients=20):
    if isinstance(ingredients, list):
        return ', '.join([str(i) for i in ingredients[:max_ingredients]])
    if isinstance(ingredients, str):
        return ingredients
    return str(ingredients)

def build_full_text(title, ingredients, instructions, max_tokens=248):
    title = str(title or '')
    ingredients = normalize_ingredients(ingredients)
    instructions = str(instructions or '')
    base = f"Title: {title}\nIngredients: {ingredients}\nInstructions: "
    if not instructions.strip():
        return base.strip()
    if token_count(base + instructions) <= max_tokens:
        return base + instructions
    words = instructions.split()
    if not words:
        return base.strip()
    low, high = 0, len(words)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = base + ' '.join(words[:mid])
        if token_count(candidate) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return base + ' '.join(words[:low])

class Adapter(nn.Module):
    def __init__(self, dim=768, bottleneck=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, dim)
        )
    def forward(self, x):
        return x + self.net(x)

def evaluate_model(name, model_name, pretrained, context_length, bottleneck, checkpoint_path=None, test_data=None):
    print(f"\n--- Evaluating {name} on Manual Pairs ---")
    
    # 1. Load base model
    from longclip_loader import load_longclip, tokenize
    
    model, preprocess, _ = load_longclip(model_name, pretrained, context_length)
    model = model.to(device)
    model.eval()
            
    # 2. Load adapters if path is provided
    use_adapters = checkpoint_path is not None
    if use_adapters:
        image_adapter = Adapter(dim=768, bottleneck=bottleneck).to(device)
        text_adapter = Adapter(dim=768, bottleneck=bottleneck).to(device)
        if Path(checkpoint_path).exists():
            print(f"Loading checkpoint {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            image_adapter.load_state_dict(ckpt['image_adapter'])
            text_adapter.load_state_dict(ckpt['text_adapter'])
        else:
            print(f"Warning: {checkpoint_path} not found. Using untrained adapters!")
        image_adapter.eval()
        text_adapter.eval()
        
    # Load recipe metadata instructions lookup
    metadata = {}
    if RECIPE_METADATA_PATH and RECIPE_METADATA_PATH.exists():
        print(f"Loading metadata instructions from {RECIPE_METADATA_PATH}")
        with open(RECIPE_METADATA_PATH, 'r') as f:
            raw_metadata = json.load(f)
        metadata = {str(k): v for k, v in raw_metadata.items()}

    print(f"Embedding {len(test_data)} manual items...")
    img_embeds = []
    txt_embeds = []
    
    with torch.no_grad():
        for item in tqdm(test_data, desc=f"{name} Embedding"):
            # Image path parsing (making it robust to relative paths)
            img_path_raw = item['image_path']
            if img_path_raw.startswith("final_run/"):
                img_path = PROJECT_ROOT / img_path_raw
            else:
                img_path = Path(img_path_raw)
                if not img_path.is_absolute():
                    img_path = PROJECT_ROOT / img_path_raw

            if not img_path.exists():
                print(f"\nWarning: Image not found at {img_path}. Attempting to locate relative to PROJECT_ROOT...")
                img_path = PROJECT_ROOT / Path(img_path_raw).name
                if not img_path.exists():
                    raise FileNotFoundError(f"Could not find image at {img_path_raw}")

            img = Image.open(img_path).convert('RGB')
            img_t = preprocess(img).unsqueeze(0).to(device)
            img_f = model.encode_image(img_t).float()
            if use_adapters: img_f = image_adapter(img_f)
            img_embeds.append(F.normalize(img_f, dim=-1).cpu().numpy())
            
            # Text
            rid = str(item.get('recipe_id', ''))
            title = item.get('recipe_title') or item.get('recipe_name') or ''
            ingredients = item.get('ingredients', '')
            instructions = ''
            if rid in metadata:
                meta = metadata[rid]
                instructions = meta.get('instructions') or meta.get('directions', '')
            
            text = build_full_text(title, ingredients, instructions, max_tokens=context_length)
            txt_t = tokenize([text], context_length=context_length).to(device)
            txt_f = model.encode_text(txt_t).float()
            if use_adapters: txt_f = text_adapter(txt_f)
            txt_embeds.append(F.normalize(txt_f, dim=-1).cpu().numpy())
            
    img_embeds = np.vstack(img_embeds)
    txt_embeds = np.vstack(txt_embeds)
    
    # 4. Calculate Similarity and Metrics
    print(f"Calculating metrics...")
    sim_matrix = img_embeds @ txt_embeds.T
    
    # Standard Recall@K
    recalls = {1: 0, 5: 0, 10: 0}
    custom_points = 0
    total = len(test_data)
    
    for i in range(total):
        scores = sim_matrix[i]
        rank = (scores >= scores[i]).sum()
        
        for k in recalls.keys():
            if rank <= k:
                recalls[k] += 1
                
        # Custom Metric: Rank 1 -> 3pts, Rank 2 -> 2pts, Rank 3 -> 1pt
        if rank == 1: custom_points += 3
        elif rank == 2: custom_points += 2
        elif rank == 3: custom_points += 1
        
    metrics = {f"Recall@{k}": v/total for k, v in recalls.items()}
    metrics["Custom Score"] = custom_points
    metrics["Max Possible Score"] = total * 3
    
    print("--- Results ---")
    for k, v in metrics.items():
        if "Recall" in k:
            print(f"{k}: {v*100:.1f}%")
        else:
            print(f"{k}: {v}")
            
    return metrics

def main():
    if not MANUAL_DATA_PATH.exists():
        print(f"\n[!] Manual test dataset not found at: {MANUAL_DATA_PATH}")
        print("Please create this file with your custom pairs when they are ready.")
        print("Example structure of 'data/manual_test_pairs.json':")
        print(json.dumps([
            {
                "image_path": "final_run/data/manual_test_images/image_1.jpg",
                "recipe_title": "Recipe Name",
                "ingredients": ["ingred1", "ingred2"],
                "recipe_id": 90001
            }
        ], indent=2))
        return
        
    with open(MANUAL_DATA_PATH, 'r') as f:
        test_data = json.load(f)
        
    if not test_data:
        print("Manual test dataset is empty!")
        return

    models_to_eval = [
        {"name": "Zero-Shot CLIP", "model": "ViT-L-14", "pre": "openai", "ctx": 77, "btn": 0, "ckpt": None},
        {"name": "Zero-Shot LongCLIP", "model": "ViT-L-14", "pre": "longclip", "ctx": 248, "btn": 0, "ckpt": None},
        {"name": "Phase 1 Baseline (5k)", "model": "ViT-L-14", "pre": "openai", "ctx": 77, "btn": 256, "ckpt": FINAL_RUN_DIR / 'models/checkpoints/phase1/best_model.pt'},
        {"name": "Phase 2 Precision (8k)", "model": "ViT-L-14", "pre": "longclip", "ctx": 248, "btn": 64, "ckpt": FINAL_RUN_DIR / 'models/checkpoints/phase2_8k/best_model.pt'},
        {"name": "Phase 2 Scale (74k)", "model": "ViT-L-14", "pre": "longclip", "ctx": 248, "btn": 64, "ckpt": FINAL_RUN_DIR / 'models/checkpoints/phase2_74k/best_model.pt'},
        {"name": "Phase 3 DAR (74k)", "model": "ViT-L-14", "pre": "longclip", "ctx": 248, "btn": 64, "ckpt": FINAL_RUN_DIR / 'models/checkpoints/dar_74k/best_model.pt'},
    ]
    
    results = {}
    for cfg in models_to_eval:
        try:
            res = evaluate_model(cfg["name"], cfg["model"], cfg["pre"], cfg["ctx"], cfg["btn"], cfg["ckpt"], test_data)
            results[cfg["name"]] = res
        except Exception as e:
            print(f"Error evaluating {cfg['name']}: {e}")
        
    # Generate Markdown Table
    report_path = REPORTS_DIR / "manual_evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("# Manual Custom Evaluation Results\n\n")
        f.write("| Model | Recall@1 | Recall@5 | Recall@10 | Custom Score (Max 3xPairs) |\n")
        f.write("|-------|----------|----------|-----------|----------------------------|\n")
        for name, res in results.items():
            r1 = f"{res['Recall@1']*100:.1f}%"
            r5 = f"{res['Recall@5']*100:.1f}%"
            r10 = f"{res['Recall@10']*100:.1f}%"
            cs = f"{res['Custom Score']}"
            f.write(f"| {name} | {r1} | {r5} | {r10} | {cs} |\n")
            
    print(f"\nManual evaluation complete! Report saved to {report_path}")

if __name__ == '__main__':
    main()
