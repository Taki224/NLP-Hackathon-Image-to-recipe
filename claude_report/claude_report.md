
A few things stand out from the data:

**Phase 1 Baseline (5k)** is the clear overall winner — it leads on Recall@1 (33.6%) and custom score (774), despite being trained on far less data than the scale variants.

**Phase 2 Precision (8k)** is a significant regression across all metrics. It drops to the lowest scores in every category, suggesting the precision-focused training strategy hurt generalization rather than helping it.

**Phase 2 Scale (74k)** and **Phase 3 DAR (74k)** are competitive with each other and close to Phase 1, but neither quite matches it on Recall@1 or the custom score — meaning the 15x more data didn't translate to better retrieval accuracy at the top rank.

The gap between Recall@1 and Recall@10 is quite large across all models (~60 percentage points), which suggests the correct match is often retrieved within the top 10 but frequently not ranked first.


