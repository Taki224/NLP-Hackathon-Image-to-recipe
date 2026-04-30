# Implementation Plan

## Goal
Implement Task 2 (Validation and Early Stopping) in the phase 2 training notebook so the pipeline uses the new paired dataset and LongCLIP-compatible text handling.

## Tech Stack
- Python 3.12
- Jupyter Notebooks
- PyTorch, torchvision
- open-clip-torch
- transformers (tokenizer)
- pandas, numpy, pillow, tqdm, matplotlib

## Data Inputs and Outputs
- Input dataset: data/datasets/paired_dataset.json
- Images: data/datasets/combined_dataset/*
- Recipe metadata: data/datasets/recipe_metadata.json
- Model checkpoints: models/checkpoints/
- Retrieval index: indexes/recipe_index.npy

## Architecture and Flow
1. Dataset/DataLoader in notebooks/phase2/train.ipynb loads the combined paired dataset.
2. Tokenizer and text adapter use LongCLIP so full recipe text can be represented.
3. Model uses LongCLIP base; optional unfreezing of last layers and/or lightweight adapters.
4. Training loop logs loss, saves checkpoints, and supports later retrieval sanity checks.

## Files to Modify
- notebooks/phase2/train.ipynb
- ui/src/app.py (later task for API response)
- ui/static/index.html (later task for UI display)
