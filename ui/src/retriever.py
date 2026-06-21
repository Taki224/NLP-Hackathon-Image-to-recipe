"""Real CLIP+adapter retrieval behind /retrieve.

Imported lazily by app.py: the heavy ML deps (torch, open_clip, numpy) load only
when a checkpoint exists, so CI -- which installs neither -- falls back to
placeholders. Mirrors the model + retrieve logic in notebooks/phase2/train.ipynb.

Index source, in order of preference:
  1. data/indexes/recipe_index.npy (+ *_ids.npy + *metadata*.json) -- full build
  2. ui/src/recipes_seed.json -- small bundled corpus, embedded once and cached

ponytail: single module, model + index built once via lru_cache.
"""

import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = Path(__file__).resolve().parent / "best_model.pt"
INDEX_DIR = Path(__file__).resolve().parent
SEED_RECIPES = Path(__file__).resolve().parent / "recipes_seed.json"
SEED_CACHE = INDEX_DIR / "seed_index.npy"

MODEL_NAME = "ViT-L-14"
# Training used LongCLIP (248-token context). If those weights aren't available
# we fall back to stock OpenAI CLIP, whose native context is 77 -- still ample
# for the short recipe texts here, and the image query path ignores it entirely.
MAX_INGREDIENTS = 20


class _Adapter(torch.nn.Module):
    # Same shape as train.ipynb: Linear(dim,bottleneck) -> ReLU -> Linear + residual.
    def __init__(self, dim, bottleneck):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, bottleneck),
            torch.nn.ReLU(),
            torch.nn.Linear(bottleneck, dim),
        )

    def forward(self, x):
        return x + self.net(x)


def _build_text(title, ingredients, instructions):
    if isinstance(ingredients, list):
        ingredients = ", ".join(str(i) for i in ingredients[:MAX_INGREDIENTS])
    return (
        f"Title: {title}\n"
        f"Ingredients: {ingredients}\n"
        f"Instructions: {instructions or ''}"
    ).strip()


def _tokenize(texts, ctx):
    try:
        return open_clip.tokenize(texts, context_length=ctx)
    except TypeError:
        return open_clip.get_tokenizer(MODEL_NAME)(texts)


@lru_cache(maxsize=1)
def _load():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Mirror train.ipynb: prefer LongCLIP weights, fall back to stock OpenAI.
    pretrained = "longclip" if "longclip" in open_clip.list_pretrained_tags_by_model(MODEL_NAME) else "openai"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=pretrained)
    ctx = getattr(model, "context_length", 77)  # use the base model's real context
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    ckpt = torch.load(CHECKPOINT, map_location=device)
    # Infer dims from the checkpoint so we never mismatch the saved adapter.
    w = ckpt["image_adapter"]["net.0.weight"]
    bottleneck, dim = w.shape
    img_ad = _Adapter(dim, bottleneck).to(device).eval()
    txt_ad = _Adapter(dim, bottleneck).to(device).eval()
    img_ad.load_state_dict(ckpt["image_adapter"])
    txt_ad.load_state_dict(ckpt["text_adapter"])
    return device, model, preprocess, img_ad, txt_ad, ctx


def _full_index():
    # Prefer the half-size fp16 index; fall back to the fp32 one. Kept in its
    # stored dtype to halve RAM -- the query vector is cast to match at search time.
    emb_path = INDEX_DIR / "recipe_index_fp16.npy"
    if not emb_path.exists():
        emb_path = INDEX_DIR / "recipe_index.npy"
    if not emb_path.exists():
        return None
    emb = np.load(emb_path)
    ids_path = INDEX_DIR / "recipe_index_ids.npy"
    ids = np.load(ids_path) if ids_path.exists() else np.arange(emb.shape[0])
    meta = {}
    for p in sorted(INDEX_DIR.glob("*metadata*.json")):
        meta = json.loads(p.read_text())
        break
    return emb, ids, meta


@lru_cache(maxsize=1)
def _index():
    full = _full_index()
    if full is not None:
        return full

    recipes = json.loads(SEED_RECIPES.read_text())
    if SEED_CACHE.exists() and SEED_CACHE.stat().st_mtime >= SEED_RECIPES.stat().st_mtime:
        return np.load(SEED_CACHE), None, recipes

    device, model, preprocess, _, txt_ad, ctx = _load()
    texts = [_build_text(r["title"], r["ingredients"], r.get("instructions", "")) for r in recipes]
    with torch.no_grad():
        feats = model.encode_text(_tokenize(texts, ctx).to(device)).float()
        emb = F.normalize(txt_ad(feats), dim=-1).cpu().numpy().astype("float32")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SEED_CACHE, emb)
    return emb, None, recipes


def _norm_ingredients(ing):
    if isinstance(ing, list):
        return [str(i) for i in ing]
    if isinstance(ing, str):
        return [s.strip() for s in ing.split(",") if s.strip()]
    return [str(ing)] if ing else []


def retrieve(image_bytes, k=3):
    """Top-k recipes for an image. Raises if checkpoint/deps missing (caller falls back)."""
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"no checkpoint at {CHECKPOINT}")
    device, model, preprocess, img_ad, _, _ = _load()
    emb, ids, meta = _index()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with torch.no_grad():
        feats = model.encode_image(preprocess(img).unsqueeze(0).to(device)).float()
        q = F.normalize(img_ad(feats), dim=-1).cpu().numpy()

    # Match the index dtype (fp16 or fp32) so we don't upcast the whole index in RAM.
    scores = (emb @ q.astype(emb.dtype).T).squeeze(-1).astype("float32")
    top = np.argsort(scores)[::-1][:k]
    out = []
    for idx in top:
        if ids is None:
            r = meta[int(idx)]
        else:
            recipe_id = ids[int(idx)]
            r = meta.get(str(int(recipe_id)), {})
        out.append({
            "recipe_name": r.get("title") or r.get("name") or "unknown",
            "ingredients": _norm_ingredients(r.get("ingredients", [])),
            "instructions": r.get("instructions") or r.get("directions") or "",
            "score": float(scores[int(idx)]),
        })
    return out


if __name__ == "__main__":
    # Self-check: runs only when deps + checkpoint + a test image are present.
    test_img = ROOT / "data/images/test/peperoni-pizza-blogbeitrag.jpg"
    res = retrieve(test_img.read_bytes(), k=3)
    assert len(res) == 3, res
    assert all(0.0 <= r["score"] <= 1.0 for r in res), res
    assert res == sorted(res, key=lambda r: r["score"], reverse=True), "not ranked"
    print("OK:", [(r["recipe_name"], round(r["score"], 3)) for r in res])
