# Slideshow Presentation Outline: Cross-Modal Recipe Retrieval

This outline is designed for a **10 to 15-minute presentation** covering the recipe retrieval project. It aligns with the experimental structure and findings of the final project report.

---

### Slide 1: Title Slide & Problem Context (1.5 mins)
* **Slide Title:** Cross-Modal Recipe Retrieval: Aligning Food Images and Long-Form Recipes
* **Subtitle:** Fine-Tuning Lightweight Adapters, Scaling Datasets, and Multi-Modal Alignment
* **Core Visuals:**
  * High-level architectural flowchart: `Query Image` $\rightarrow$ `Encoder` $\rightarrow$ `Shared Space` $\leftarrow$ `Encoder` $\leftarrow$ `2M Recipe database`.
* **Talking Points:**
  * **The Goal:** Match a photo of a prepared dish to its exact full recipe (ingredients, instructions) in a database of 2 million recipes.
  * **Traditional Limitation:** Standard classification maps images to static labels (e.g., "pizza"). We want to retrieve cooking instructions.
  * **The Sequence Constraint:** Standard CLIP has a hard **77-token limit**. Recipes are much longer. This project investigates using **LongCLIP** (248-token context) to bypass the "sequence wall."

---

### Slide 2: The Dataset Creation & Multi-Stage Filtration (2 mins)
* **Slide Title:** Curating the Dataset: Multi-Stage Filtration
* **Core Visuals:**
  * Flowchart diagram: `Raw Matches` $\rightarrow$ `Stage 1: LongCLIP Text Filter` $\rightarrow$ `Stage 2: SigLIP Cosine Filter` $\rightarrow$ `Stage 3: VLM Validation` $\rightarrow$ `74,000 High-Quality Pairs`.
* **Talking Points:**
  * **Raw Input:** Food101 images combined with Food.com 2M recipe database.
  * **The Noise Challenge:** Simple keyword matching is highly noisy (e.g. matching cake frosting recipes to "chocolate cake").
  * **The Solution:** A three-stage filtration pipeline:
    1. **Stage 1 (LongCLIP Text-to-Text):** Filter out bad matches (frosting, glaze, sauce) against canonical descriptions.
    2. **Stage 2 (SigLIP Image-Text):** Compare image to title + ingredients. Apply category-specific thresholds (75th percentile) and global minimum (0.5).
    3. **Stage 3 (VL-LLM/Ollama):** Local Llama 3 validation on borderline, uncertain cases.
  * **Scaling Yield:** Permits controlled recipe reuse (up to 5x), expanding the yield from an overfitting-prone 8k to a robust **74k pairs**.

---

### Slide 3: DAR: Data Augmentation for Recipe Retrieval (1.5 mins)
* **Slide Title:** DAR: SAM Segmentation & Visual Imaginations
* **Core Visuals:**
  * Side-by-side comparison:
    * Left: Original image with table background.
    * Right: SAM-segmented cropped food image with white background.
    * Text callout showing the LLM-generated ~30-word visual description.
* **Talking Points:**
  * **Segment Anything Model (SAM):** Visual query images contain plates, tables, and restaurant settings. SAM detects and crops the central food item to suppress background noise.
  * **LLM Visual Imaginations:** Ollama/Llama 3 translates structured instructions and ingredients into concrete visual descriptions (colors, plating, textures).
  * **Why DAR?** Prepares the model to learn fine-grained boundaries (e.g., calzone vs pizza) by matching original and segmented modalities.

---

### Slide 4: Model Architectures & Training Philosophy (1.5 mins)
* **Slide Title:** Parameter-Efficient Adapters & Loss Functions
* **Core Visuals:**
  * MLP Bottleneck Adapter Diagram: `Frozen Encoder Embedding (768)` $\rightarrow$ `MLP Bottleneck (64 or 256)` $\rightarrow$ `Zero-initialized bypass` $\rightarrow$ `Adapted Vector`.
* **Talking Points:**
  * **Parameter Efficiency:** We freeze the massive base CLIP/LongCLIP weights. We only train lightweight post-model MLP layers to avoid overfitting and massive compute requirements.
  * **Stable Initialization:** MLP adapter is initialized with zero weights on the output layer so training starts as a safe identity mapping.
  * **Contrastive Loss Functions:**
    * **InfoNCE:** Standard symmetric image-to-text / text-to-image similarity matrix diagonal maximization.
    * **Multi-Level Circle Loss (Phase 3):** Optimizes 4 alignment combinations (original/original, augmented/augmented, original/augmented, augmented/original) scaled by $\gamma=80$ to establish rigid positive boundaries and push away hard negatives.

---

### Slide 5: Slide-by-Slide Model Configuration Comparison (1.5 mins)
* **Slide Title:** Adapter Setup & Training Parameters
* **Core Visuals:**
  * Clean markdown comparison table:
    * **Phase 1 Baseline (5k):** 256-dim Bottleneck | 5k pairs | InfoNCE
    * **Phase 2 Precision (8k):** 64-dim Bottleneck | 8k pairs | InfoNCE
    * **Phase 2 Scale (74k):** 64-dim Bottleneck | 74k pairs | InfoNCE
    * **Phase 3 DAR (74k):** 64-dim Bottleneck | 74k pairs (augmented) | Multi-Level Circle Loss
* **Talking Points:**
  * **Phase 1 (5k dataset, 256 bottleneck):** An exploratory test on a small dataset to explore the upper limits of representation capacity.
  * **Phase 2 (Precision 8k):** A comparison model using a constricted 64-dimension bottleneck.
  * **Phase 2 Scale (74k):** Evaluates the impact of 10x dataset scale on the narrow 64-dim bottleneck.
  * **Phase 3 DAR (74k):** Tests multi-modal alignment (SAM crops, LLM visual descriptions) with Multi-Level Circle Loss on the 74k set.

---

### Slide 6: Results & Ablation Analysis (2 mins)
* **Slide Title:** Quantitative Performance Comparison
* **Core Visuals:**
  * Evaluation Results Table (Zero-Shot LongCLIP vs. 4 Adapters): Recall@1, Recall@5, Recall@10, Custom Score.
  * Bar charts showing Recall comparison and Custom retrieval score.
* **Talking Points:**
  * **Zero-Shot Base:** Zero-shot LongCLIP scores **25.6% Recall@1**.
  * **Bottleneck Capacity Ablation:** Phase 1 Baseline scores **33.6% Recall@1** (highest overall), proving that bottleneck dimension capacity (256 vs 64) is a dominant factor in preserving representations.
  * **Data Scale Ablation:** Scaling data from 8k to 74k (at 64 bottleneck) yields a massive **+10% Recall@1 gain** (22.0% $\rightarrow$ 32.0%), showing data volume recovers bottleneck losses.
  * **Augmentation Ablation:** Phase 3 DAR yields **32.6% Recall@1** (+7.0% gain over base), confirming Segment Anything crops and Circle Loss enforce fine-grained retrieval.

---

### Slide 7: Training Saturation & Computational Complexity (1.5 mins)
* **Slide Title:** Saturation Dynamics & Training Durations
* **Core Visuals:**
  * Comparative loss curves: InfoNCE (Phase 1 / Phase 2 Scale) side-by-side with Multi-Level Circle Loss (Phase 3 DAR, showing the gamma=80 starting scale of 80.2 converging to 64.7).
* **Talking Points:**
  * **Training Saturation:** Loss curves converge and plateau early (within 15 epochs) because the base encoders are frozen. Adapters cannot learn features beyond the baseline model's intrinsic resolution.
  * **Computational Speed:**
    * Training on 8k dataset takes **~6 minutes**; training on 74k dataset takes **1h 26m**.
    * **DAR training takes 3h 41m (2.5x longer):** Due to processing 4 forward/backward passes per batch and computing the 4-level Circle Loss.
    * *Note:* Indexing the 2M recipe dataset takes ~2 hours and 18 minutes on GPU.

---

### Slide 8: Retrieval Visualization (1 min)
* **Slide Title:** Output Space Retrieval Example
* **Core Visuals:**
  * The Pepperoni Pizza query image with top 3 matched calzone/pizza recipes side-by-side.
* **Talking Points:**
  * Show query example: A photo of a pepperoni pizza correctly retrieves pizza and calzone recipes (e.g. *Three-Meat Calzone*, *Pizza Margherita*).
  * Demonstrates that the joint image-text space is aligned well enough to locate culinary matches from a pool of 2 million recipes.

---

### Slide 9: Methodological Challenges Faced (1 min)
* **Slide Title:** Key Challenges and Solutions
* **Core Visuals:**
  * 2x2 grid representing:
    * **The 77-Token Wall:** Solved by migrating to LongCLIP.
    * **MLP Bottleneck Information Loss:** Solved by balancing 64-dim (speed/VRAM) with 256-dim baseline lessons.
    * **Low Data Yield:** Solved by removing recipe caps and allowing up to 5x controlled reuse.
    * **Out-of-Distribution Noise:** Solved by similarity confidence warnings in the API.
* **Talking Points:**
  * Summarize the core scientific hurdles of the project and how they were systematically bypassed.

---

### Slide 10: Overall Learning & Conclusions (1.5 mins)
* **Slide Title:** Overall Learnings & Next Steps
* **Core Visuals:**
  * Summary bullet points: `Capacity Over Scale` $\rightarrow$ `DAR Effectiveness` $\rightarrow$ `Unfreeze Base Layers`.
* **Talking Points:**
  * **Learning 1:** Adapter width (bottleneck dimension) is critical. Constricting embeddings down to 64 dimensions limits retrieval accuracy. We recommend a 256/512 bottleneck.
  * **Learning 2:** SAM background suppression and Circle Loss align fine-grained modalities (Visual/Imagined) effectively.
  * **Learning 3:** To beat standard CLIP (35.6% Recall@1), future runs must **unfreeze the last 1–2 layers of the LongCLIP base model** during adapter training, enabling attention weights to adapt directly to cooking instruction vocabulary.
