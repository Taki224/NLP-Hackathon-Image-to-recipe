#!/usr/bin/env python
# coding: utf-8

# # Model Training
# Dish to Recipe — Cross-Modal Retrieval
# 
# Fine-tunes CLIP-ViT-L/14 adapter layers using InfoNCE contrastive loss on food-recipe pairs.
# Expected training time: ~1 hour on a good GPU.

# ## Step 1 — Install Dependencies
# Install the Python packages needed for training and indexing.

# In[1]:


pass # pass # get_ipython().system('uv add lightning open_clip_torch torch torchvision pillow tqdm pandas numpy -q')


# ## Step 2 — Imports & Device Setup
# Import libraries, set paths, and choose dataset and metadata sources.

# In[ ]:


import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from PIL import Image
import open_clip
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
from lightning import Trainer, seed_everything
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PROJECT_ROOT = Path.cwd() if (Path.cwd() / 'pyproject.toml').exists() else Path.cwd().parent

CHECKPOINT_DIR = PROJECT_ROOT / 'final_run/models/checkpoints/phase2_8k'
CHECKPOINT_PATH = CHECKPOINT_DIR / 'best_model.ckpt'
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / 'last.ckpt'

INDEX_PATH = PROJECT_ROOT / 'final_run/data/indexes/recipe_index_phase2_8k.npy'
INDEX_IDS_PATH = PROJECT_ROOT / 'final_run/data/indexes/recipe_index_ids_phase2_8k.npy'
INDEX_META_PATH = PROJECT_ROOT / 'data/indexes/recipe_index_metadata_newdata3.json'

PAIRED_DATASET_PATH = PROJECT_ROOT / 'final_run/data/datasets/paired_dataset_8k_train.json'
ALT_PAIRED_DATASET_PATH = PROJECT_ROOT / 'data/datasets/paired_dataset3.json'
RECIPE_CSV_PATH = PROJECT_ROOT / 'data/datasets/recipe_dataset_2m.csv'

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

DATASET_VARIANT = 'filtered'  # 'filtered' or 'raw'
DATASET_PATH = PROJECT_ROOT / 'final_run/data/datasets/paired_dataset_8k_train.json'
# data_creator.ipynb may write paired_dataset3.json instead of paired_dataset.json;
# fall back to whichever paired dataset actually exists on disk.
if not DATASET_PATH.exists():
    DATASET_PATH = find_first_existing([PAIRED_DATASET_PATH, ALT_PAIRED_DATASET_PATH]) or DATASET_PATH

TEST_IMAGES_DIR = PROJECT_ROOT / 'data/images/test'
FIGURES_DIR = PROJECT_ROOT / 'final_run/reports/figures'

print(f'Using device: {device}')
if device == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')


# ## Step 3 — Load CLIP Model
# Load a LongCLIP-capable model and tokenizer, set context length, and freeze base weights.

# In[ ]:


MODEL_NAME = 'ViT-L-14'
PRETRAINED = 'openai'
CONTEXT_LENGTH = 248

from longclip_loader import load_longclip, tokenize

model, preprocess, model_source = load_longclip(MODEL_NAME, PRETRAINED, CONTEXT_LENGTH)
model = model.to(device)

# Freeze all base CLIP weights by default
for param in model.parameters():
    param.requires_grad = False

print(f'{MODEL_NAME} loaded ({model_source}) with context length {CONTEXT_LENGTH}')


# ## Step 4 — Define Adapter Layers
# Small MLP adapters added on top of CLIP's frozen image and text encoders.
# Only these layers are trained — keeps training fast and stable.
# Configure adapters and optional unfreezing of the last encoder blocks.

# In[ ]:


if hasattr(model, 'text_projection') and model.text_projection is not None:
    EMBED_DIM = model.text_projection.shape[1]
else:
    EMBED_DIM = 768

USE_ADAPTERS = True
ADAPTER_BOTTLENECK = 64
UNFREEZE_LAST_LAYERS = 0

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

def build_adapter(dim):
    if not USE_ADAPTERS:
        return nn.Identity()
    return Adapter(dim=dim, bottleneck=ADAPTER_BOTTLENECK)

def unfreeze_last_layers(model, num_layers):
    trainable = []
    if num_layers <= 0:
        return trainable

    text_blocks = getattr(model, 'transformer', None)
    if text_blocks is not None and hasattr(text_blocks, 'resblocks'):
        for block in text_blocks.resblocks[-num_layers:]:
            for param in block.parameters():
                param.requires_grad = True
                trainable.append(param)

    visual = getattr(model, 'visual', None)
    if visual is not None and hasattr(visual, 'transformer') and hasattr(visual.transformer, 'resblocks'):
        for block in visual.transformer.resblocks[-num_layers:]:
            for param in block.parameters():
                param.requires_grad = True
                trainable.append(param)

    return trainable

image_adapter = build_adapter(EMBED_DIM).to(device)
text_adapter  = build_adapter(EMBED_DIM).to(device)

# Learnable temperature
log_temperature = nn.Parameter(torch.tensor(0.07).log())

trainable_base_params = unfreeze_last_layers(model, UNFREEZE_LAST_LAYERS)
TRAIN_BASE = len(trainable_base_params) > 0

trainable_params = (
    list(image_adapter.parameters()) +
    list(text_adapter.parameters()) +
    [log_temperature] +
    trainable_base_params
 )

def apply_adapter(adapter, feats):
    return adapter(feats) if USE_ADAPTERS else feats

total = sum(p.numel() for p in trainable_params)
print(f'Trainable parameters: {total:,}')
print(f'Base layers unfrozen: {UNFREEZE_LAST_LAYERS if TRAIN_BASE else 0}')


# ## Step 5 — Dataset Class
# Build the dataset with full recipe text, token truncation, and optional hard-positive text.

# In[ ]:


MAX_TOKENS = CONTEXT_LENGTH
TOKEN_COUNT_CONTEXT = 512
MAX_INGREDIENTS = 20
USE_HARD_POSITIVE_AUG = False
USE_HARD_NEGATIVE_LOSS = False
HARD_NEGATIVE_K = 3
HARD_POSITIVE_WEIGHT = 0.5

def token_count(text, context_length=TOKEN_COUNT_CONTEXT):
    try:
        tokens = open_clip.tokenize([text], context_length=context_length)
    except TypeError:
        tokens = open_clip.tokenize([text])
    return int((tokens[0] != 0).sum().item())

def normalize_ingredients(ingredients, max_ingredients=MAX_INGREDIENTS):
    if isinstance(ingredients, list):
        return ', '.join([str(i) for i in ingredients[:max_ingredients]])
    if isinstance(ingredients, str):
        return ingredients
    return str(ingredients)

def build_full_text(title, ingredients, instructions, max_tokens=MAX_TOKENS):
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

class FoodRecipeDataset(Dataset):
    def __init__(self, json_path, preprocess, tokenize, metadata_path=None, project_root=None,
                 max_tokens=MAX_TOKENS, max_ingredients=MAX_INGREDIENTS, return_short_text=False):
        with open(json_path) as f:
            self.data = json.load(f)
        self.preprocess = preprocess
        self.tokenize = tokenize
        self.max_tokens = max_tokens
        self.max_ingredients = max_ingredients
        self.return_short_text = return_short_text
        self.project_root = Path(project_root) if project_root else Path.cwd()

        self.metadata = {}
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                raw_metadata = json.load(f)
            self.metadata = {str(k): v for k, v in raw_metadata.items()}

        self.recipe_texts = self._build_text_cache()

    def _build_text_cache(self):
        fallback = {}
        for item in self.data:
            rid = str(item.get('recipe_id', ''))
            if not rid or rid in fallback:
                continue
            fallback[rid] = {
                'title': item.get('recipe_title') or item.get('recipe_name') or '',
                'ingredients': item.get('ingredients', '')
            }

        recipe_texts = {}
        for rid, fb in fallback.items():
            meta = self.metadata.get(rid, {})
            title = meta.get('title') or fb['title']
            ingredients = meta.get('ingredients') or fb['ingredients']
            instructions = meta.get('instructions') or meta.get('directions', '')
            recipe_texts[rid] = build_full_text(title, ingredients, instructions, self.max_tokens)
        return recipe_texts

    def _resolve_image_path(self, image_path):
        p_str = str(image_path).replace('../../', '')
        path = Path(p_str)
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        return path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = self._resolve_image_path(item['image_path'])
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.preprocess(img)

        rid = str(item.get('recipe_id', ''))
        text = self.recipe_texts.get(rid)
        if text is None:
            title = item.get('recipe_title') or item.get('recipe_name') or ''
            ingredients = item.get('ingredients', '')
            text = build_full_text(title, ingredients, '', self.max_tokens)

        text_tensor = self.tokenize([text])[0]
        if not self.return_short_text:
            return img_tensor, text_tensor

        short_text = f"Title: {item.get('recipe_title') or item.get('recipe_name') or ''}\n"
        short_text += f"Ingredients: {normalize_ingredients(item.get('ingredients', ''))}"
        short_text_tensor = self.tokenize([short_text])[0]
        return img_tensor, text_tensor, short_text_tensor

dataset = FoodRecipeDataset(
    DATASET_PATH,
    preprocess,
    tokenize,
    metadata_path=RECIPE_METADATA_PATH,
    project_root=PROJECT_ROOT,
    max_tokens=CONTEXT_LENGTH,
    max_ingredients=MAX_INGREDIENTS,
    return_short_text=USE_HARD_POSITIVE_AUG,
 )
print(f'Dataset size: {len(dataset)} pairs')

# Sample check
if USE_HARD_POSITIVE_AUG:
    img_t, txt_t, txt_short_t = dataset[0]
    print(f'Image tensor shape: {img_t.shape}')
    print(f'Text tensor shape:  {txt_t.shape}')
    print(f'Short text shape:  {txt_short_t.shape}')
else:
    img_t, txt_t = dataset[0]
    print(f'Image tensor shape: {img_t.shape}')
    print(f'Text tensor shape:  {txt_t.shape}')


# ## Step 6 — DataLoader
# Set hyperparameters and create the DataLoader, optimizer, and scheduler.

# In[ ]:


BATCH_SIZE = 2048
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 8
SAVE_EVERY = 2
LOG_EVERY = 50
ACCUMULATE_GRAD_BATCHES = 1

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device == 'cuda')
 )

print(f'Batches per epoch: {len(train_loader)}')
print(f'Total steps: {len(train_loader) * NUM_EPOCHS}')


# ## Step 7 — InfoNCE Loss Function
# Define the symmetric contrastive loss for image-text matching.

# In[ ]:


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
# Run training with checkpoint resume, early stopping, and optional hard positive/negative losses.

# In[ ]:


seed_everything(42, workers=True)

class LitRecipeRetrieval(LightningModule):
    def __init__(self, model, image_adapter, text_adapter, log_temperature, train_base, lr, weight_decay):
        super().__init__()
        self.model = model
        self.image_adapter = image_adapter
        self.text_adapter = text_adapter
        self.log_temperature = log_temperature
        self.train_base = train_base
        self.lr = lr
        self.weight_decay = weight_decay

    def on_train_start(self):
        if self.train_base:
            self.model.train()
        else:
            self.model.eval()

    def training_step(self, batch, batch_idx):
        if USE_HARD_POSITIVE_AUG:
            images, texts, texts_short = batch
        else:
            images, texts = batch
        images = images.to(self.device)
        texts = texts.to(self.device)
        if USE_HARD_POSITIVE_AUG:
            texts_short = texts_short.to(self.device)

        if self.train_base:
            image_feats = self.model.encode_image(images).float()
            text_feats = self.model.encode_text(texts).float()
        else:
            with torch.no_grad():
                image_feats = self.model.encode_image(images).float()
                text_feats = self.model.encode_text(texts).float()

        image_embeds = apply_adapter(self.image_adapter, image_feats)
        text_embeds = apply_adapter(self.text_adapter, text_feats)

        loss = infonce_loss(image_embeds, text_embeds, self.log_temperature)
        if USE_HARD_POSITIVE_AUG:
            if self.train_base:
                short_feats = self.model.encode_text(texts_short).float()
            else:
                with torch.no_grad():
                    short_feats = self.model.encode_text(texts_short).float()
            short_embeds = apply_adapter(self.text_adapter, short_feats)
            pos_loss = infonce_loss(image_embeds, short_embeds, self.log_temperature)
            loss = loss + HARD_POSITIVE_WEIGHT * pos_loss
        if USE_HARD_NEGATIVE_LOSS:
            image_norm = F.normalize(image_embeds, dim=-1)
            text_norm = F.normalize(text_embeds, dim=-1)
            logits = image_norm @ text_norm.T
            mask = torch.eye(logits.size(0), device=logits.device).bool()
            logits = logits.masked_fill(mask, -1e9)
            hard_k = min(HARD_NEGATIVE_K, max(1, logits.size(1) - 1))
            if hard_k > 0:
                hard_neg_vals, _ = logits.topk(hard_k, dim=1)
                hard_neg_loss = F.relu(hard_neg_vals - HARD_NEGATIVE_MARGIN).mean()
                loss = loss + HARD_NEGATIVE_WEIGHT * hard_neg_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }


lit_model = LitRecipeRetrieval(
    model=model,
    image_adapter=image_adapter,
    text_adapter=text_adapter,
    log_temperature=log_temperature,
    train_base=TRAIN_BASE,
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    filename='best_model',
    monitor='train_loss_epoch',
    mode='min',
    save_top_k=1,
    save_last=True,
    save_weights_only=False
)
early_stopping = EarlyStopping(
    monitor='train_loss_epoch',
    mode='min',
    patience=5,
    min_delta=0.05
)
lr_monitor = LearningRateMonitor(logging_interval='step')
logger = CSVLogger(save_dir=str(PROJECT_ROOT / 'final_run/reports'), name='lightning_logs')

trainer = Trainer(
    max_epochs=NUM_EPOCHS,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    precision='16-mixed' if torch.cuda.is_available() else 32,
    gradient_clip_val=1.0,
    log_every_n_steps=LOG_EVERY,
    accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
    callbacks=[checkpoint_callback, early_stopping, lr_monitor],
    logger=logger,
    default_root_dir=str(CHECKPOINT_DIR),
    enable_checkpointing=True
)

ckpt_path = str(LAST_CHECKPOINT_PATH) if LAST_CHECKPOINT_PATH.exists() else None
if ckpt_path:
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'optimizer_states' not in checkpoint or not checkpoint['optimizer_states']:
            print("Checkpoint does not contain optimizer states. Loading weights only and starting from epoch 0.")
            lit_model.load_state_dict(checkpoint['state_dict'])
            ckpt_path = None
    except Exception as e:
        print(f"Failed to load checkpoint: {e}. Starting from scratch.")
        ckpt_path = None

APP_CHECKPOINT_PATH = CHECKPOINT_DIR / 'best_model.pt'
if APP_CHECKPOINT_PATH.exists():
    print(f"Trained model {APP_CHECKPOINT_PATH} already exists. Skipping training and going straight to indexing.")
    # Find the latest best_model*.ckpt file to load
    ckpt_files = list(CHECKPOINT_DIR.glob("best_model*.ckpt"))
    if ckpt_files:
        CHECKPOINT_PATH = max(ckpt_files, key=os.path.getmtime)
        print(f"Using existing checkpoint: {CHECKPOINT_PATH}")
else:
    trainer.fit(lit_model, train_dataloaders=train_loader, ckpt_path=ckpt_path)
    if checkpoint_callback.best_model_path:
        CHECKPOINT_PATH = Path(checkpoint_callback.best_model_path)

print(f'Best checkpoint: {CHECKPOINT_PATH}')


# ## Step 9 — Plot Training Loss
# Plot and save the loss curve for review.

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

metrics_path = Path(logger.log_dir) / 'metrics.csv'
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    if 'train_loss_epoch' in metrics.columns:
        loss_df = metrics.dropna(subset=['train_loss_epoch'])
        loss_values = loss_df.groupby('epoch')['train_loss_epoch'].last().tolist()
    elif 'train_loss_step' in metrics.columns:
        loss_df = metrics.dropna(subset=['train_loss_step'])
        loss_values = loss_df['train_loss_step'].tolist()
    else:
        loss_values = []
        
    if len(loss_values) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('InfoNCE Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURES_DIR / 'training_loss_phase2_8k.png', dpi=100)
        plt.show()
    else:
        print("Skipping plot generation because no new epochs were trained.")
else:
    print(f"Skipping plot: no metrics file found at {metrics_path}.")


# ## Step 10 — Pre-compute Recipe Index
# Embeds all recipes from Food.com and saves to disk.
# This only needs to run once — the index is reused at inference time.
# Generate embeddings and save the index plus recipe id mapping.

# In[ ]:


import pandas as pd
import ast

if not RECIPE_CSV_PATH.exists():
    raise FileNotFoundError(f'Missing recipe CSV: {RECIPE_CSV_PATH}')
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f'Missing checkpoint: {CHECKPOINT_PATH}')

def load_lightning_weights(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    image_state = {k.replace('image_adapter.', ''): v for k, v in state_dict.items() if k.startswith('image_adapter.')}
    text_state = {k.replace('text_adapter.', ''): v for k, v in state_dict.items() if k.startswith('text_adapter.')}
    model_state = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
    if image_state:
        image_adapter.load_state_dict(image_state)
    if text_state:
        text_adapter.load_state_dict(text_state)
    if model_state:
        model.load_state_dict(model_state, strict=False)
    if 'log_temperature' in state_dict:
        log_temperature.data = state_dict['log_temperature']

load_lightning_weights(CHECKPOINT_PATH)
image_adapter.to(device)
text_adapter.to(device)
model.to(device)
if hasattr(model, 'attn_mask') and model.attn_mask is not None:
    model.attn_mask = model.attn_mask.to(device)
if hasattr(model, 'positional_embedding') and model.positional_embedding is not None:
    model.positional_embedding = torch.nn.Parameter(model.positional_embedding.to(device))
image_adapter.eval()
text_adapter.eval()
model.eval()

# Export adapter weights in the flat dict format the web app (ui/src/retriever.py)
# and the published HuggingFace checkpoint expect: best_model.pt, not the Lightning .ckpt.
APP_CHECKPOINT_PATH = CHECKPOINT_DIR / 'best_model.pt'
torch.save({
    'image_adapter': image_adapter.state_dict(),
    'text_adapter': text_adapter.state_dict(),
    'log_temperature': log_temperature.detach().cpu(),
    'epoch': -1,
}, APP_CHECKPOINT_PATH)
print(f'App checkpoint saved: {APP_CHECKPOINT_PATH}')

recipes_df = pd.read_csv(RECIPE_CSV_PATH)
if 'ingredients' in recipes_df.columns and isinstance(recipes_df['ingredients'].iloc[0], str):
    try:
        recipes_df['ingredients'] = recipes_df['ingredients'].apply(ast.literal_eval)
    except Exception:
        pass
if 'directions' not in recipes_df.columns:
    recipes_df['directions'] = ''

recipes_df = recipes_df.dropna(subset=['title', 'ingredients', 'directions']).reset_index(names=['recipe_id'])
recipes_df['title'] = recipes_df['title'].astype(str)

if not INDEX_PATH.exists():
    recipe_ids = recipes_df['recipe_id'].astype(int).to_numpy()
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_IDS_PATH, recipe_ids)

    print(f'Indexing {len(recipes_df)} recipes...')
    all_embeddings = []
    EMBED_BATCH = 512

    with torch.no_grad():
        for i in tqdm(range(0, len(recipes_df), EMBED_BATCH)):
            batch = recipes_df.iloc[i:i+EMBED_BATCH]
            
            texts = []
            for _, row in batch.iterrows():
                title = row.get('title', '')
                ingredients = row.get('ingredients', '')
                instructions = row.get('directions', '')
                text = build_full_text(title, ingredients, instructions, max_tokens=248)
                texts.append(text)
                
            tokens = tokenize(texts).to(device)
            feats = model.encode_text(tokens).float()
            embeds = text_adapter(feats)
            embeds = F.normalize(embeds, dim=-1)
            all_embeddings.append(embeds.cpu().numpy())

    recipe_index = np.vstack(all_embeddings)
    np.save(INDEX_PATH, recipe_index)
    print(f'Index saved: {recipe_index.shape}')
else:
    print(f'Index already exists at {INDEX_PATH}, skipping generation.')
    recipe_index = np.load(INDEX_PATH)

# Must stay True: retrieval and the web app need the recipe text (title/ingredients/
# instructions), keyed by the same recipe_id used in the index above. Embeddings alone
# only give scores, not displayable recipes.
WRITE_METADATA = True
if WRITE_METADATA:
    recipe_metadata = {
        str(row['recipe_id']): {
            'title': row['title'],
            'ingredients': row['ingredients'],
            'instructions': str(row.get('directions', ''))
        }
        for _, row in recipes_df.iterrows()
    }
    with open(INDEX_META_PATH, 'w') as f:
        json.dump(recipe_metadata, f)
    print(f'Recipe metadata saved: {INDEX_META_PATH}')

print(f'Index saved: {recipe_index.shape}')
print(f'Recipe ids saved: {INDEX_IDS_PATH}')


# ## Step 11 — Quick Retrieval Test
# Sanity check: query with a Food101 image and see if the top-3 results make sense.
# Run retrieval and render results with ingredients and instructions.

# In[ ]:


# Step 11 — Retrieval Test on New Images
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

NEW_IMAGES_FOLDER = TEST_IMAGES_DIR  # drop your new images here
NEW_IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)

if 'recipe_index' not in globals():
    recipe_index = np.load(INDEX_PATH)
if 'recipe_ids' not in globals():
    if INDEX_IDS_PATH.exists():
        recipe_ids = np.load(INDEX_IDS_PATH)
    else:
        recipe_ids = np.arange(recipe_index.shape[0])

if 'recipe_metadata' not in globals():
    recipe_metadata = {}
    if RECIPE_METADATA_PATH and Path(RECIPE_METADATA_PATH).exists():
        with open(RECIPE_METADATA_PATH, 'r') as f:
            recipe_metadata = json.load(f)

def normalize_ingredient_list(ingredients):
    if isinstance(ingredients, list):
        return [str(i) for i in ingredients]
    if isinstance(ingredients, str):
        return [i.strip() for i in ingredients.split(',') if i.strip()]
    return [str(ingredients)]

def retrieve(image_path, k=3):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats  = model.encode_image(img_tensor).float()
        embeds = apply_adapter(image_adapter, feats)
        embeds = F.normalize(embeds, dim=-1).cpu().numpy()

    scores = (recipe_index @ embeds.T).squeeze()
    top_k  = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_k:
        recipe_id = int(recipe_ids[idx])
        meta = recipe_metadata.get(str(recipe_id), {})
        title = meta.get('title') or meta.get('name') or 'unknown'
        ingredients = normalize_ingredient_list(meta.get('ingredients', []))
        instructions = meta.get('instructions') or meta.get('directions', '')
        results.append({
            'recipe_id': recipe_id,
            'title': title,
            'ingredients': ingredients,
            'instructions': instructions,
            'score': float(scores[idx])
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
            instructions = r['instructions']
            if len(instructions) > 160:
                instructions = instructions[:160] + '...'
            result_text += f"#{i+1} {r['title'].title()} (score: {r['score']:.3f})\n"
            result_text += f"     {ingredients_str}\n"
            if instructions:
                result_text += f"     {instructions}\n\n"
            else:
                result_text += "\n"

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
