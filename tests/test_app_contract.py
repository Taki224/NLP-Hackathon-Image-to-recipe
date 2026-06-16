"""Contract tests for the FastAPI retrieval app.

These are the CI gate: if pushing/merging to main breaks the app's boot or the
/retrieve response shape the UI depends on, these fail. No model/checkpoint/data
needed -- the endpoint returns placeholder recipes, but the *contract* is what
the frontend consumes and must not break.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

REQUIRED_KEYS = {"rank", "recipe_name", "ingredients", "instructions", "score"}


def _fake_image():
    # Minimal JPEG magic bytes; handler reads but ignores content.
    return {"image": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")}


def test_app_imports():
    assert isinstance(app, FastAPI)


def test_retrieve_endpoint_exists():
    r = client.post("/retrieve", files=_fake_image())
    assert r.status_code == 200, f"expected 200, got {r.status_code}"


def test_retrieve_requires_image():
    # No file -> FastAPI validation rejects it. Guards the upload contract.
    r = client.post("/retrieve")
    assert r.status_code == 422


def test_retrieve_returns_ranked_list():
    data = client.post("/retrieve", files=_fake_image()).json()
    assert isinstance(data, list) and len(data) >= 1
    for i, item in enumerate(data, start=1):
        assert REQUIRED_KEYS <= item.keys(), f"missing keys: {REQUIRED_KEYS - item.keys()}"
        assert item["rank"] == i, f"ranks must be sequential from 1, got {item['rank']} at position {i}"
        assert isinstance(item["score"], (int, float))
