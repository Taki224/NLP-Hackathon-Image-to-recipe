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
TEST_DATA_PATH = FINAL_RUN_DIR / 'data/datasets/test_dataset.json'
REPORTS_DIR = FINAL_RUN_DIR / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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

def evaluate_model(name, model_name, pretrained, context_length, bottleneck, checkpoint_path=None):
    print(f"\n--- Evaluating {name} ---")
    
    # 1. Load base model
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    except:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
        
    if hasattr(model, 'context_length'):
        model.context_length = context_length
        
    if hasattr(model, 'positional_embedding') and model.positional_embedding.shape[0] != context_length:
        pe = model.positional_embedding
        new_pe = torch.zeros(context_length, pe.shape[1], device=pe.device, dtype=pe.dtype)
        new_pe[:pe.shape[0]] = pe
        new_pe[pe.shape[0]:] = pe[-1]
        model.positional_embedding = torch.nn.Parameter(new_pe)
        
    if hasattr(model, 'attn_mask') and model.attn_mask is not None and model.attn_mask.shape[0] != context_length:
        mask = torch.empty(context_length, context_length, device=model.attn_mask.device)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        model.attn_mask = mask
        
    model = model.to(device)
    model.eval()

    def tokenize(texts):
        try:
            return open_clip.tokenize(texts, context_length=context_length)
        except TypeError:
            tokenizer = open_clip.get_tokenizer(model_name)
            return tokenizer(texts)
            
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
        
    # 3. Load Test Data
    with open(TEST_DATA_PATH, 'r') as f:
        test_data = json.load(f)
        
    print(f"Embedding {len(test_data)} test items...")
    img_embeds = []
    txt_embeds = []
    
    with torch.no_grad():
        for item in tqdm(test_data, desc=f"{name} Embedding"):
            # Image
            img_path = PROJECT_ROOT / item['image_path'].replace('../../', '')
            img = Image.open(img_path).convert('RGB')
            img_t = preprocess(img).unsqueeze(0).to(device)
            img_f = model.encode_image(img_t).float()
            if use_adapters: img_f = image_adapter(img_f)
            img_embeds.append(F.normalize(img_f, dim=-1).cpu().numpy())
            
            # Text
            title = item.get('recipe_title', '')
            ingredients = ', '.join(item.get('ingredients', [])[:15]) if isinstance(item.get('ingredients'), list) else item.get('ingredients', '')
            text = f"Title: {title}\nIngredients: {ingredients}"
            txt_t = tokenize([text]).to(device)
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
        # The correct recipe for image i is recipe i (since we ordered them together)
        scores = sim_matrix[i]
        rank = (scores >= scores[i]).sum() # rank is number of scores greater or equal to the correct one
        
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
    if not TEST_DATA_PATH.exists():
        print(f"Error: {TEST_DATA_PATH} not found. Run split_data.py first.")
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
        res = evaluate_model(cfg["name"], cfg["model"], cfg["pre"], cfg["ctx"], cfg["btn"], cfg["ckpt"])
        results[cfg["name"]] = res
        
    # Generate Markdown Table
    report_path = REPORTS_DIR / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("# Final Quantitative Evaluation Results\n\n")
        f.write("| Model | Recall@1 | Recall@5 | Recall@10 | Custom Score (Max 1500) |\n")
        f.write("|-------|----------|----------|-----------|------------------------|\n")
        for name, res in results.items():
            r1 = f"{res['Recall@1']*100:.1f}%"
            r5 = f"{res['Recall@5']*100:.1f}%"
            r10 = f"{res['Recall@10']*100:.1f}%"
            cs = f"{res['Custom Score']}"
            f.write(f"| {name} | {r1} | {r5} | {r10} | {cs} |\n")
            
    print(f"\nEvaluation complete! Report saved to {report_path}")

if __name__ == '__main__':
    main()
