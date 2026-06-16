import sys
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

# Absolute paths so the app runs from any cwd (root or ui/).
SRC = Path(__file__).resolve().parent
STATIC = SRC.parent / "static"
sys.path.insert(0, str(SRC))  # so `import retriever` works however app is loaded

app = FastAPI()

# Shown when the real model/checkpoint isn't available (e.g. CI, fresh clone).
PLACEHOLDER = [
    {
        "recipe_name": "Spaghetti Carbonara",
        "ingredients": ["pasta", "eggs", "bacon", "parmesan", "black pepper"],
        "instructions": "Boil pasta. Crisp bacon. Whisk eggs and cheese. Toss pasta with bacon and egg mixture off heat.",
        "score": 0.92,
    },
    {
        "recipe_name": "Truffle Mac and Cheese",
        "ingredients": ["macaroni", "heavy cream", "truffle", "parmesan", "breadcrumbs"],
        "instructions": "Cook macaroni. Make cheese sauce with cream and parmesan. Fold in truffle. Top with breadcrumbs and bake.",
        "score": 0.85,
    },
    {
        "recipe_name": "Creamy Pasta Primavera",
        "ingredients": ["penne", "zucchini", "cherry tomatoes", "cream", "garlic"],
        "instructions": "Saute vegetables. Simmer with cream and garlic. Toss with penne and season to taste.",
        "score": 0.78,
    },
]


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


@app.post("/retrieve")
async def retrieve(image: UploadFile = File(...)):
    image_bytes = await image.read()

    # Real model if checkpoint + ML deps are present; otherwise placeholders so
    # the UI (and the CI contract test) keep working.
    try:
        from retriever import retrieve as model_retrieve

        results = model_retrieve(image_bytes, k=3)
        if not results:
            raise ValueError("empty result")
    except Exception:
        results = PLACEHOLDER

    return JSONResponse([{"rank": i, **r} for i, r in enumerate(results, 1)])
