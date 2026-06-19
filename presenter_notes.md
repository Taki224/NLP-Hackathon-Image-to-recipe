# Presenter Notes: Cross-Modal Recipe Retrieval Presentation

These presenter notes are structured for a group of **3 presenters** delivering a **10 to 15-minute presentation**. The slides are divided logically to ensure a balanced division of labor, thematic consistency, and an even distribution of speaking time.

## 👥 Division of Labor Summary

| Presenter | Role / Theme | Slides | Speaking Time |
| :--- | :--- | :--- | :--- |
| **Presenter 1** | **The Data & Problem Setup** (Problem context, 3-stage dataset creation, and DAR visual crops/imaginations). | Slides 1 - 3 | ~5.0 minutes |
| **Presenter 2** | **The Model & Optimization** (PEFT bottleneck adapters, setup parameters, quantitative results, and training complexity). | Slides 4 - 7 | ~6.5 minutes |
| **Presenter 3** | **Evaluation & Takeaways** (Retrieval output space, methodological hurdles, final learnings, and Q&A lead). | Slides 8 - 11 | ~4.0 minutes (+ Q&A) |

---

## 🎙️ Speaker Scripts by Slide

### Slide 1: Title Slide & Problem Context (1.5 mins) — `[PRESENTER 1]`
* **Visual:** Slide title and subtitles.
* **Speaker Script:**
> "Hello everyone. Today, my teammates and I will be presenting our project on Cross-Modal Recipe Retrieval. The goal of this work is to build an end-to-end retrieval system that matches a query image of a prepared dish to its corresponding full text recipe—which includes the title, structured ingredients list, and detailed cooking instructions—out of a database of 2 million recipes. We address the core challenge of cross-modal sequence length limits by fine-tuning lightweight post-model adapters on top of frozen vision-language foundation backbones."

---

### Slide 2: Curating the Dataset: Filtration (2 mins) — `[PRESENTER 1]`
* **Visual:** Pipeline timeline (Raw Matches $\rightarrow$ LongCLIP Text Filter $\rightarrow$ SigLIP Cosine Filter $\rightarrow$ VLM Validation $\rightarrow$ 74,000 Pairs).
* **Speaker Script:**
> "Let's start with how we created our training dataset. We paired Food101 images against 2 million recipes from Food.com. Simple keyword matching introduces substantial noise—for instance, matching cake frosting recipes to 'chocolate cake.' 
> 
> To resolve this, we designed a three-stage filtering pipeline. First, we ran a LongCLIP text ranking step to filter out recipes with exclusionary words. Second, we scored candidate pairs with SigLIP using a category-specific threshold. Third, we validated borderline cases using a local Llama 3 model. By removing recipe search caps and allowing controlled recipe reuse, we scaled our yield from an initial 8,000 pairs to a robust, diverse dataset of 74,000 high-quality training pairs."

---

### Slide 3: DAR: SAM & Visual Imaginations (1.5 mins) — `[PRESENTER 1]`
* **Visual:** Pie image segmentation visual showing the background crop, and details on SAM and LLM visual descriptions.
* **Speaker Script:**
> "To help our model discriminate between visually similar dishes, we created the DAR—or Data Augmentation for Recipe retrieval—framework. 
> 
> DAR operates on two levels. For the visual modality, visual images often contain plate edges, tables, or background clutter. We used Meta's Segment Anything Model (SAM) to crop out the background, leaving only the central food item on a white background. For the textual modality, we used Llama 3 to generate a ~30-word visual description focusing strictly on plating, textures, and colors. This dual modality augmentation prepares the model to learn fine-grained representation boundaries."

---

### Slide 4: Parameter-Efficient Adapters (1.5 mins) — `[PRESENTER 2]`
* **Visual:** Architecture bullets on the left, autoencoder bottleneck diagram on the right.
* **Speaker Script:**
> "For our model architecture, we opted for Parameter-Efficient Fine-Tuning. Unfreezing entire encoders like CLIP or LongCLIP requires massive GPU memory and risks severe overfitting. 
> 
> Instead, we freeze the base image and text encoders and train lightweight MLP bottleneck adapters on top of their output embeddings. The adapters map the 768-dimensional representations to a bottleneck dimension and project them back with a residual bypass. To ensure training starts stably, the final projection layer is initialized to zero, making the starting state an identity mapping. For optimization, we study both standard InfoNCE loss and Multi-Level Circle Loss."

---

### Slide 5: Adapter Setup & Parameters (1.5 mins) — `[PRESENTER 2]`
* **Visual:** Matrix table comparing the 4 trained configurations (Phase 1 Baseline, Phase 2 Precision, Phase 2 Scale, Phase 3 DAR).
* **Speaker Script:**
> "To isolate the effects of dataset size and model capacity, we trained four distinct adapter configurations. 
> 
> Phase 1 is a baseline trained on 5,000 pairs with a wide 256-dimensional bottleneck. Phase 2 Precision constricts this bottleneck to 64 dimensions on 8,000 pairs. Phase 2 Scale uses the same 64-dim bottleneck but scales data volume 10x to 74,000 pairs. Finally, Phase 3 DAR trains the bottleneck-64 adapter on our augmented 74,000 pair dataset using Multi-Level Circle Loss. The same 64-dim bottleneck was maintained across Phases 2 and 3 to ensure direct comparability."

---

### Slide 6: Quantitative Performance (2 mins) — `[PRESENTER 2]`
* **Visual:** Two bar charts comparing Recall@1/5/10 and Custom Weighted Score across models.
* **Speaker Script:**
> "Here are our quantitative results evaluated on an independent test set. 
> 
> Several interesting patterns emerge. First, our Phase 1 Baseline achieved the highest overall score of 33.6% Recall@1. This highlights a critical lesson: bottleneck capacity is a dominant factor, and the wider 256-dim bottleneck preserved representations much better than the 64-dim versions. Second, looking at the 64-dim models, scaling data from 8k to 74k yielded a huge +10.0% Recall@1 improvement. Third, Phase 3 DAR achieved the best performance among the 64-dim models at 32.6% Recall@1, proving that SAM background suppression and Circle Loss successfully align fine-grained features."

---

### Slide 7: Saturation Dynamics (1.5 mins) — `[PRESENTER 2]`
* **Visual:** Training loss curves and training durations (8k: ~6 mins, 74k: ~1.5h, DAR: ~3.7h).
* **Speaker Script:**
> "This slide illustrates the training dynamics and computational complexity. Across all runs, training loss converges and saturates quickly within 15 epochs. This saturation is a direct consequence of keeping the base encoders frozen—adapters cannot learn features beyond the baseline model's intrinsic resolution. 
> 
> Looking at durations, training on the 8k set takes under 6 minutes, while the 74k set takes 1.5 hours. Crucially, the DAR phase takes 3.7 hours—about 2.5 times longer. This is because DAR calculates Multi-Level Circle Loss across 4 cross-modal alignments, which requires four forward and backward passes per batch."

---

### Slide 8: Output Space Retrieval (1 min) — `[PRESENTER 3]`
* **Visual:** Pizza query visual matching Calzone and Pizza Margherita top results.
* **Speaker Script:**
> "To visualize our learned representations in action, we queried the final 2-million recipe index with a pepperoni pizza image. 
> 
> The top 3 retrieved results are: First, Three-Meat Calzone with a similarity score of 0.357. Second, Pizza Margherita with a Black Truffle Oil Crust. Third, Margherita Pizza with Tomato, Mozzarella, and Basil. This demonstrates that the joint embedding space is successfully aligned: the model recognizes semantic similarities in shape, dough, cheese, and ingredients, matching calzones and margherita pizzas alongside standard pepperoni query targets."

---

### Slide 9: Methodological Challenges (1 min) — `[PRESENTER 3]`
* **Visual:** 3 boxes representing the 77-Token Wall, Bottleneck Loss, and Granularity of Categories.
* **Speaker Script:**
> "To summarize, the three main methodological challenges we faced and overcame were: 
> 
> First, the 77-Token Wall, which we bypassed by migrating to LongCLIP. Second, MLP Bottleneck Information Loss, where our Phase 1 exploratory baseline taught us that wider bottlenecks are necessary to preserve high-dimensional representations. Third, the Granularity of Categories, where visual details get lost in broad classes. We resolved this by implementing the DAR framework using Segment Anything visual crops and LLM visual imaginations."

---

### Slide 10: Overall Learnings (1.5 mins) — `[PRESENTER 3]`
* **Visual:** Bullet points: Capacity Over Scale, DAR Effectiveness, Next Steps (unfreeze base layers).
* **Speaker Script:**
> "Our overall learnings point to a few key conclusions. 
> 
> First, adapter width is critical—we recommend using a bottleneck dimension of 256 or higher for culinary text. Second, background visual segmentation via SAM combined with Circle Loss is a highly effective way to enforce fine-grained cross-modal boundaries. Third, as a next step to beat standard CLIP's performance, we recommend unfreezing the last 1 to 2 layers of the LongCLIP base model during training, allowing the base representations to shift and adapt to the cooking instructions vocabulary."

---

### Slide 11: Thank You (0.5 mins) — `[PRESENTER 3]`
* **Visual:** Thank you text / Questions query.
* **Speaker Script:**
> "That concludes our presentation. Thank you for your attention. We are happy to take any questions you may have."
