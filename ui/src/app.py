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
            "instructions": "Boil pasta. Crisp bacon. Whisk eggs and cheese. Toss pasta with bacon and egg mixture off heat.",
            "score": 0.92
        },
        {
            "rank": 2,
            "recipe_name": "Truffle Mac and Cheese",
            "ingredients": ["macaroni", "heavy cream", "truffle", "parmesan", "breadcrumbs"],
            "instructions": "Cook macaroni. Make cheese sauce with cream and parmesan. Fold in truffle. Top with breadcrumbs and bake.",
            "score": 0.85
        },
        {
            "rank": 3,
            "recipe_name": "Creamy Pasta Primavera",
            "ingredients": ["penne", "zucchini", "cherry tomatoes", "cream", "garlic"],
            "instructions": "Saute vegetables. Simmer with cream and garlic. Toss with penne and season to taste.",
            "score": 0.78
        },
    ]
    # --- End placeholder block ---

    return JSONResponse(results)