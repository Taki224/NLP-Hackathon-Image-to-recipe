# UI Implementation Plan: Dish to Recipe

A simple, step-by-step guide to building the web interface for the food image → recipe retrieval system.

---

## What We're Building

A single web page where a user can:
1. Upload (or drag-and-drop) a photo of a dish
2. See a preview of their photo
3. Click **"Find Recipes"**
4. See the top 3 matching recipes returned by the model

---

## Tech Stack (kept simple)

| Layer | Tool | Why |
|---|---|---|
| Backend | **FastAPI** (Python) | lightweight, async, easy file upload |
| Frontend | **Plain HTML + CSS + JS** | no framework needed, easy to understand |
| Communication | **Fetch API** (browser built-in) | sends image to backend, gets recipes back |

No React, no Node, no build step. Just one `.html` file and one `.py` file.

---

## Step 1 — Install FastAPI

Run this once to add FastAPI to the project:

```bash
uv add fastapi uvicorn python-multipart
```

- `fastapi` — the web framework
- `uvicorn` — the server that runs FastAPI
- `python-multipart` — needed to receive uploaded files

---

## Step 2 — Create the Backend (`src/app.py`)

This file does two things:
1. Serves the HTML page when someone visits the site
2. Accepts an uploaded image and returns 3 recipes

**File: `src/app.py`**

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import io

app = FastAPI()

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
def index():
    return Path("static/index.html").read_text()

# Accept image upload, return top-3 recipes
@app.post("/retrieve")
async def retrieve(image: UploadFile = File(...)):
    image_bytes = await image.read()

    # --- Replace this block with your real model later ---
    # For now it returns fake recipes so the UI works immediately
    results = [
        {
            "rank": 1,
            "recipe_name": "Spaghetti Carbonara",
            "ingredients": ["pasta", "eggs", "bacon", "parmesan", "black pepper"],
            "score": 0.92
        },
        {
            "rank": 2,
            "recipe_name": "Truffle Mac and Cheese",
            "ingredients": ["macaroni", "heavy cream", "truffle", "parmesan", "breadcrumbs"],
            "score": 0.85
        },
        {
            "rank": 3,
            "recipe_name": "Creamy Pasta Primavera",
            "ingredients": ["penne", "zucchini", "cherry tomatoes", "cream", "garlic"],
            "score": 0.78
        },
    ]
    # --- End placeholder block ---

    return JSONResponse(results)
```

**What this does:**
- `GET /` → returns the HTML page to the browser
- `POST /retrieve` → receives the uploaded image, runs the model (placeholder for now), returns JSON with 3 recipes

---

## Step 3 — Create the Frontend (`static/index.html`)

Create a folder called `static/` in the project root, then create this file inside it.

**File: `static/index.html`**

The page has 3 sections:
1. **Upload area** — drag-and-drop box or click-to-browse
2. **Image preview** — shows the photo the user picked
3. **Results area** — shows 3 recipe cards after the server responds

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Dish to Recipe</title>
  <style>
    /* ── Layout ── */
    body { font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #fafafa; }
    h1   { text-align: center; color: #333; }

    /* ── Upload box ── */
    #drop-zone {
      border: 2px dashed #aaa; border-radius: 12px;
      padding: 40px; text-align: center; cursor: pointer;
      color: #888; transition: background 0.2s;
    }
    #drop-zone.hover { background: #eef; border-color: #66f; }
    #file-input { display: none; }

    /* ── Preview ── */
    #preview { display: none; text-align: center; margin: 20px 0; }
    #preview img { max-height: 280px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }

    /* ── Button ── */
    #submit-btn {
      display: none; width: 100%; padding: 14px;
      background: #e85d04; color: white; border: none;
      border-radius: 8px; font-size: 1.1rem; cursor: pointer;
    }
    #submit-btn:hover { background: #c44d00; }

    /* ── Loading spinner ── */
    #loading { display: none; text-align: center; margin: 20px 0; color: #888; }

    /* ── Recipe cards ── */
    #results { margin-top: 30px; }
    .card {
      background: white; border-radius: 10px; padding: 18px 22px;
      margin-bottom: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);
      border-left: 5px solid #e85d04;
    }
    .card h3 { margin: 0 0 6px; color: #222; }
    .card .score { font-size: 0.85rem; color: #999; margin-bottom: 10px; }
    .card ul { margin: 0; padding-left: 18px; color: #555; }
    .card ul li { margin-bottom: 3px; }
  </style>
</head>
<body>

  <h1>🍽️ Dish to Recipe</h1>
  <p style="text-align:center; color:#777">Upload a photo of a dish — get the top 3 matching recipes instantly.</p>

  <!-- Step 1: Upload -->
  <div id="drop-zone">
    <p>Drag & drop a food photo here<br/>or <strong>click to browse</strong></p>
    <input type="file" id="file-input" accept="image/*" />
  </div>

  <!-- Step 2: Preview -->
  <div id="preview">
    <img id="preview-img" src="" alt="preview" />
  </div>

  <!-- Step 3: Submit -->
  <button id="submit-btn">Find Recipes</button>

  <!-- Loading indicator -->
  <div id="loading">Searching recipes…</div>

  <!-- Step 4: Results -->
  <div id="results"></div>

  <script>
    const dropZone   = document.getElementById('drop-zone');
    const fileInput  = document.getElementById('file-input');
    const preview    = document.getElementById('preview');
    const previewImg = document.getElementById('preview-img');
    const submitBtn  = document.getElementById('submit-btn');
    const loading    = document.getElementById('loading');
    const results    = document.getElementById('results');

    let selectedFile = null;

    // Open file picker on click
    dropZone.addEventListener('click', () => fileInput.click());

    // Highlight on drag-over
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('hover'); });
    dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('hover'));

    // Handle dropped file
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('hover');
      handleFile(e.dataTransfer.files[0]);
    });

    // Handle file-picker selection
    fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));

    function handleFile(file) {
      if (!file || !file.type.startsWith('image/')) return;
      selectedFile = file;

      // Show preview
      previewImg.src = URL.createObjectURL(file);
      preview.style.display  = 'block';
      submitBtn.style.display = 'block';
      results.innerHTML = '';
    }

    // Send image to backend
    submitBtn.addEventListener('click', async () => {
      if (!selectedFile) return;

      submitBtn.style.display = 'none';
      loading.style.display   = 'block';
      results.innerHTML       = '';

      const formData = new FormData();
      formData.append('image', selectedFile);

      try {
        const response = await fetch('/retrieve', { method: 'POST', body: formData });
        const recipes  = await response.json();
        displayResults(recipes);
      } catch (err) {
        results.innerHTML = '<p style="color:red">Something went wrong. Please try again.</p>';
      } finally {
        loading.style.display   = 'none';
        submitBtn.style.display = 'block';
      }
    });

    function displayResults(recipes) {
      results.innerHTML = '<h2 style="color:#333">Top Matches</h2>';
      recipes.forEach(r => {
        const ingredientItems = r.ingredients.map(i => `<li>${i}</li>`).join('');
        results.innerHTML += `
          <div class="card">
            <h3>#${r.rank} — ${r.recipe_name}</h3>
            <div class="score">Similarity score: ${(r.score * 100).toFixed(0)}%</div>
            <strong>Ingredients:</strong>
            <ul>${ingredientItems}</ul>
          </div>`;
      });
    }
  </script>
</body>
</html>
```

---

## Step 4 — Run the App

```bash
uvicorn src.app:app --reload
```

Then open your browser at: **http://localhost:8000**

You should see the upload page. Upload any food photo — it will return the 3 placeholder recipes. The UI is fully working before the model is even connected.

---

## Step 5 — Connect the Real Model

Once the model is trained, replace the placeholder block in `src/app.py` with the real inference code:

```python
# At the top of app.py, load model and index once at startup
from PIL import Image
from src.model import CLIPWithAdapters
from src.config import Config
import numpy as np, json, torch

cfg = Config()
model = CLIPWithAdapters.load(cfg.checkpoint_path)
model.eval()

recipe_embeddings = np.load(cfg.embeddings_path)          # shape (N, 256)
recipe_index      = json.load(open(cfg.recipe_index_path)) # list of {id, name, ingredients}

# Inside the /retrieve endpoint, replace the placeholder block with:
image_pil  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
image_emb  = model.encode_image(image_pil)                 # shape (1, 256)

scores     = recipe_embeddings @ image_emb.T               # cosine sim (N, 1)
top3_idx   = scores[:, 0].argsort()[::-1][:3]

results = [
    {
        "rank":        i + 1,
        "recipe_name": recipe_index[idx]["name"],
        "ingredients": recipe_index[idx]["ingredients"],
        "score":       float(scores[idx, 0]),
    }
    for i, idx in enumerate(top3_idx)
]
```

No changes to the HTML/JS are needed — the frontend already handles this response format.

---

## File Structure After This Plan

```
NLP-Hackathon-Image-to-recipe/
├── src/
│   └── app.py          ← Step 2
├── static/
│   └── index.html      ← Step 3
├── pyproject.toml      ← updated in Step 1
└── UI_PLAN.md
```

---

## Summary

| Step | What you do | Result |
|---|---|---|
| 1 | `uv add fastapi uvicorn python-multipart` | Dependencies installed |
| 2 | Create `src/app.py` | Backend with placeholder responses |
| 3 | Create `static/index.html` | Full drag-and-drop UI |
| 4 | `uvicorn src.app:app --reload` | Working website at localhost:8000 |
| 5 | Swap placeholder with real model code | Live recipe retrieval |
