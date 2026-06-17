#!/usr/bin/env python
# coding: utf-8

# # Model Training
# Dish to Recipe — Cross-Modal Retrieval
# 
# Fine-tunes CLIP-ViT-L/14 adapter layers using InfoNCE contrastive loss on food-recipe pairs.
# Expected training time: ~1 hour on a good GPU.

# ## Step 1 — Install Dependencies



pass # pass # get_ipython().system('uv add open_clip_torch torch torchvision pillow tqdm -q')


# ## Step 2 — Imports & Device Setup



import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PROJECT_ROOT = Path.cwd() if (Path.cwd() / 'pyproject.toml').exists() else Path.cwd().parent
CHECKPOINT_PATH = PROJECT_ROOT / 'final_run/models/checkpoints/phase1/best_model.pt'
INDEX_PATH = PROJECT_ROOT / 'final_run/data/indexes/recipe_index_phase1.npy'
INDEX_META_PATH = PROJECT_ROOT / 'final_run/data/indexes/recipe_index_metadata.csv'
TEST_IMAGES_DIR = PROJECT_ROOT / 'data/images/test'
FIGURES_DIR = PROJECT_ROOT / 'final_run/reports/figures'

print(f'Using device: {device}')
if device == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')


# ## Step 3 — Load CLIP Model



from longclip_loader import load_longclip, tokenize as tokenizer
model, preprocess, _ = load_longclip('ViT-L-14', context_length=248)
model = model.to(device)

# Freeze all base CLIP weights
for param in model.parameters():
    param.requires_grad = False

print('CLIP-ViT-L/14 loaded and frozen')


# ## Step 4 — Define Adapter Layers
# Small MLP adapters added on top of CLIP's frozen image and text encoders.
# Only these layers are trained — keeps training fast and stable.



class Adapter(nn.Module):
    def __init__(self, dim=768, bottleneck=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, dim)
        )
        # Init as near-identity so training starts stable
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return x + self.net(x)  # residual connection

image_adapter = Adapter(dim=768).to(device)
text_adapter  = Adapter(dim=768).to(device)

# Learnable temperature
log_temperature = nn.Parameter(torch.tensor(0.07).log())

trainable_params = (
    list(image_adapter.parameters()) +
    list(text_adapter.parameters()) +
    [log_temperature]
)

total = sum(p.numel() for p in trainable_params)
print(f'Trainable parameters: {total:,}')


# ## Step 5 — Dataset Class



class FoodRecipeDataset(Dataset):
    def __init__(self, json_path, preprocess, tokenizer, max_ingredients=10):
        with open(json_path) as f:
            self.data = json.load(f)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.max_ingredients = max_ingredients

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load and preprocess image (resolving relative paths robustly)
        img_path_str = item['image_path'].replace('../../', '')
        img_path = PROJECT_ROOT / img_path_str
        if not img_path.exists():
            img_path = Path(item['image_path'])

        img = Image.open(img_path).convert('RGB')
        img_tensor = self.preprocess(img)

        # Build text: "<recipe name>: ingredient1, ingredient2, ..."
        ingredients = item['ingredients'][:self.max_ingredients]
        title = item.get('recipe_title') or item.get('recipe_name') or ''
        text = f"{title}: {', '.join(ingredients)}"

        # Tokenize (CLIP truncates at 77 tokens automatically)
        text_tensor = self.tokenizer([text])[0]

        return img_tensor, text_tensor


dataset = FoodRecipeDataset(
    str(PROJECT_ROOT / 'final_run/data/datasets/paired_dataset_train.json'),
    preprocess, tokenizer
)
print(f'Dataset size: {len(dataset)} pairs')

# Sample check
img_t, txt_t = dataset[0]
print(f'Image tensor shape: {img_t.shape}')
print(f'Text tensor shape:  {txt_t.shape}')


# ## Step 6 — DataLoader



BATCH_SIZE = 2048
NUM_EPOCHS = 30
LR = 1e-4

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

print(f'Batches per epoch: {len(loader)}')
print(f'Total steps: {len(loader) * NUM_EPOCHS}')


# ## Step 7 — InfoNCE Loss Function



def infonce_loss(image_embeds, text_embeds, log_temp):
    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds  = F.normalize(text_embeds,  dim=-1)

    # Similarity matrix: [batch x batch]
    temperature = log_temp.exp().clamp(min=0.01, max=100)
    logits = (image_embeds @ text_embeds.T) / temperature

    # Diagonal = correct pairs
    labels = torch.arange(len(logits), device=logits.device)

    # Symmetric loss: image→text and text→image
    loss_i2t = F.cross_entropy(logits,   labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2


# ## Step 8 — Training Loop



CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
best_loss = float('inf')
loss_history = []

model.eval()  # CLIP stays in eval (frozen)
image_adapter.train()
text_adapter.train()

patience = 5
min_delta = 0.05
no_improve_count = 0

# Resume from checkpoint
start_epoch = 0
if CHECKPOINT_PATH.exists():
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        image_adapter.load_state_dict(checkpoint['image_adapter'])
        text_adapter.load_state_dict(checkpoint['text_adapter'])
        log_temperature.data = checkpoint['log_temperature']
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming from epoch {start_epoch}')
    except Exception as e:
        print(f'Failed to load checkpoint: {e}. Starting from scratch.')
else:
    print('No checkpoint found. Starting from scratch.')

for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_losses = []

    for images, texts in tqdm(loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
        images = images.to(device)
        texts  = texts.to(device)

        with torch.no_grad():
            image_feats = model.encode_image(images).float()
            text_feats  = model.encode_text(texts).float()

        # Pass through adapters
        image_embeds = image_adapter(image_feats)
        text_embeds  = text_adapter(text_feats)

        loss = infonce_loss(image_embeds, text_embeds, log_temperature)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        epoch_losses.append(loss.item())

    scheduler.step()
    avg_loss = np.mean(epoch_losses)
    loss_history.append(avg_loss)
    print(f'Epoch {epoch+1} — avg loss: {avg_loss:.4f} | temp: {log_temperature.exp().item():.4f}')

    # Save best checkpoint
    if avg_loss < best_loss - min_delta:
        best_loss = avg_loss
        no_improve_count = 0
        torch.save({
            'image_adapter': image_adapter.state_dict(),
            'text_adapter':  text_adapter.state_dict(),
            'log_temperature': log_temperature.data,
            'epoch': epoch
        }, CHECKPOINT_PATH)
        print(f'  ✓ Saved best checkpoint')
    else:
        no_improve_count += 1
        print(f'  No improvement ({no_improve_count}/{patience})')
        if no_improve_count >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

if start_epoch >= NUM_EPOCHS:
    print(f'\nTraining already completed (resumed at epoch {start_epoch}).')
else:
    print(f'\nTraining complete. Best loss: {best_loss:.4f}')


# ## Step 9 — Plot Training Loss



import matplotlib.pyplot as plt

if len(loss_history) > 0:
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('InfoNCE Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'training_loss2.png', dpi=100)
    plt.show()
else:
    print("Skipping plot generation because no new epochs were trained.")


# ## Step 10 — Pre-compute Recipe Index
# Embeds all recipes from Food.com and saves to disk.
# This only needs to run once — the index is reused at inference time.



import pandas as pd
import ast

# Load best checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
image_adapter.load_state_dict(checkpoint['image_adapter'])
text_adapter.load_state_dict(checkpoint['text_adapter'])
log_temperature.data = checkpoint['log_temperature']
image_adapter.eval()
text_adapter.eval()

# Load Food.com recipes
recipes_df = pd.read_csv(PROJECT_ROOT / 'data/datasets/recipe_dataset_2m.csv')
if 'ingredients' in recipes_df.columns and isinstance(recipes_df['ingredients'].iloc[0], str):
    try:
        recipes_df['ingredients'] = recipes_df['ingredients'].apply(ast.literal_eval)
    except Exception:
        pass

# Rename for compatibility with downstream code
if 'title' in recipes_df.columns:
    recipes_df = recipes_df.rename(columns={'title': 'name'})
    
# The new dataset lacks an explicit ID column, so we generate it from the index
recipes_df = recipes_df.reset_index(names=['id'])
    
recipes_df = recipes_df[['name', 'ingredients', 'id']].dropna()

print(f'Indexing {len(recipes_df)} recipes...')

if not INDEX_PATH.exists():
    all_embeddings = []
    EMBED_BATCH = 2048

    with torch.no_grad():
        for i in tqdm(range(0, len(recipes_df), EMBED_BATCH)):
            batch = recipes_df.iloc[i:i+EMBED_BATCH]
            texts = [
                f"{row['name']}: {', '.join(row['ingredients'][:10])}"
                for _, row in batch.iterrows()
            ]
            tokens = tokenizer(texts).to(device)
            feats  = model.encode_text(tokens).float()
            embeds = text_adapter(feats)
            embeds = F.normalize(embeds, dim=-1)
            all_embeddings.append(embeds.cpu().numpy())

    recipe_index = np.vstack(all_embeddings)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_PATH, recipe_index)
    recipes_df.to_csv(INDEX_META_PATH, index=False)

    print(f'Index saved: {recipe_index.shape}')
else:
    print(f'Index already exists at {INDEX_PATH}, skipping generation.')
    recipe_index = np.load(INDEX_PATH)

# ## Step 11 — Quick Retrieval Test
# Sanity check: query with a Food101 image and see if the top-3 results make sense.



# Step 11 — Retrieval Test on New Images
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

NEW_IMAGES_FOLDER = TEST_IMAGES_DIR  # drop your new images here
NEW_IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)

def retrieve(image_path, k=3):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats  = model.encode_image(img_tensor).float()
        embeds = image_adapter(feats)
        embeds = F.normalize(embeds, dim=-1).cpu().numpy()

    scores = (recipe_index @ embeds.T).squeeze()
    top_k  = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_k:
        row = recipes_df.iloc[idx]
        results.append({
            'recipe':      row['name'],
            'ingredients': row['ingredients'][:8],
            'score':       float(scores[idx])
        })
    return results


# Find all images in the folder
image_files = [
    f for f in os.listdir(str(NEW_IMAGES_FOLDER))
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
]

if len(image_files) == 0:
    print(f'No images found in {NEW_IMAGES_FOLDER}. Add some JPG/PNG images and re-run.')
else:
    for image_file in image_files:
        image_path = os.path.join(str(NEW_IMAGES_FOLDER), image_file)
        results = retrieve(image_path, k=3)

        # Plot image + results side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: query image
        axes[0].imshow(mpimg.imread(image_path))
        axes[0].set_title(f'Query: {image_file}', fontsize=10)
        axes[0].axis('off')

        # Right: top-3 results as text
        result_text = ''
        for i, r in enumerate(results):
            ingredients_str = ', '.join(r['ingredients'][:5])
            result_text += f"#{i+1} {r['recipe'].title()} (score: {r['score']:.3f})\n"
            result_text += f"     {ingredients_str}\n\n"

        axes[1].text(0.05, 0.95, result_text, transform=axes[1].transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace',
                     wrap=True)
        axes[1].axis('off')

        plt.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURES_DIR / f'results_{os.path.splitext(image_file)[0]}.png', dpi=100)
        plt.show()
        print('-' * 60)


# ## Done!
# Output files:
# ```
# models/checkpoints/best_model.pt       ← trained adapter weights
# data/indexes/recipe_index.npy          ← pre-computed recipe embeddings (50k)
# data/indexes/recipe_index_metadata.csv ← recipe names and ingredients
# reports/figures/training_loss2.png     ← loss curve
# ```
