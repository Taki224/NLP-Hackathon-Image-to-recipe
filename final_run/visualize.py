import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import open_clip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PROJECT_ROOT = Path(__file__).parent.parent
FINAL_RUN_DIR = PROJECT_ROOT / 'final_run'
TEST_DATA_PATH = FINAL_RUN_DIR / 'data/datasets/test_dataset.json'
FIGURES_DIR = FINAL_RUN_DIR / 'reports/figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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

def get_embeddings(model_name, pretrained, context_length, bottleneck, checkpoint_path, test_data, subset_size=100):
    print(f"Loading model {model_name}...")
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
            
    image_adapter = Adapter(dim=768, bottleneck=bottleneck).to(device)
    text_adapter = Adapter(dim=768, bottleneck=bottleneck).to(device)
    
    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        image_adapter.load_state_dict(ckpt['image_adapter'])
        text_adapter.load_state_dict(ckpt['text_adapter'])
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained adapters.")
    image_adapter.eval()
    text_adapter.eval()
    
    img_embeds = []
    txt_embeds = []
    
    # Just take a subset for clear visualization
    np.random.seed(42)
    subset = np.random.choice(test_data, min(subset_size, len(test_data)), replace=False)
    
    print(f"Extracting embeddings for {len(subset)} pairs...")
    with torch.no_grad():
        for item in tqdm(subset, desc="Embedding"):
            # Image
            img_path = PROJECT_ROOT / item['image_path'].replace('../../', '')
            img = Image.open(img_path).convert('RGB')
            img_t = preprocess(img).unsqueeze(0).to(device)
            img_f = model.encode_image(img_t).float()
            img_f = image_adapter(img_f)
            img_embeds.append(F.normalize(img_f, dim=-1).cpu().numpy())
            
            # Text
            title = item.get('recipe_title', '')
            ingredients = ', '.join(item.get('ingredients', [])[:15]) if isinstance(item.get('ingredients'), list) else item.get('ingredients', '')
            text = f"Title: {title}\nIngredients: {ingredients}"
            txt_t = tokenize([text]).to(device)
            txt_f = model.encode_text(txt_t).float()
            txt_f = text_adapter(txt_f)
            txt_embeds.append(F.normalize(txt_f, dim=-1).cpu().numpy())
            
    return np.vstack(img_embeds), np.vstack(txt_embeds)

def main():
    if not TEST_DATA_PATH.exists():
        print(f"Error: {TEST_DATA_PATH} not found. Run split_data.py first.")
        return
        
    with open(TEST_DATA_PATH, 'r') as f:
        test_data = json.load(f)
        
    # We will visualize the DAR 74k model
    ckpt_path = FINAL_RUN_DIR / 'models/checkpoints/dar_74k/best_model.pt'
    img_embeds, txt_embeds = get_embeddings(
        model_name="ViT-L-14", 
        pretrained="longclip", 
        context_length=248, 
        bottleneck=64, 
        checkpoint_path=ckpt_path, 
        test_data=test_data, 
        subset_size=60 # 60 pairs is a good amount without being too cluttered
    )
    
    print("Running t-SNE...")
    # Stack them together for a unified projection space
    combined = np.vstack([img_embeds, txt_embeds])
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='pca', learning_rate='auto')
    projected = tsne.fit_transform(combined)
    
    n = len(img_embeds)
    img_proj = projected[:n]
    txt_proj = projected[n:]
    
    print("Plotting...")
    plt.figure(figsize=(12, 10))
    
    # Plot points
    plt.scatter(img_proj[:, 0], img_proj[:, 1], c='blue', alpha=0.6, label='Images', s=50)
    plt.scatter(txt_proj[:, 0], txt_proj[:, 1], c='orange', alpha=0.6, label='Recipes', s=50, marker='^')
    
    # Draw connecting lines between pairs
    for i in range(n):
        plt.plot([img_proj[i, 0], txt_proj[i, 0]], 
                 [img_proj[i, 1], txt_proj[i, 1]], 
                 'k-', alpha=0.15)
                 
    plt.title('t-SNE Visualization of Joint Image-Recipe Embedding Space (DAR Framework)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    save_path = FIGURES_DIR / 'tsne_alignment.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE plot saved to {save_path}")

if __name__ == '__main__':
    main()
