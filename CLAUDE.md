# Project: Online Shoppers Clustering (Práctica 4)

## Context

Academic clustering practice using `online_shoppers_intention.xlsx` — 12,330 rows, 18 columns. Goal: segment users by purchase intent using clustering algorithms. `Revenue` is excluded from clustering and used only for evaluation.

Work is done in **local `.py` files**, not the notebook. The notebook (`Práctica_4_INW_Clustering.ipynb`) is reference-only for structure/requirements.

## Documentation

**Keep `docs.md` updated as work progresses.** Every script, decision, and result goes there. Tidy, neat, and concise — no fluff.

## Status

**Preprocessing is complete and working.** Do not redo it. Start from `preprocessing.py`.

Next step: implement clustering algorithms. Import `X_df` and `y` from `preprocessing.py` into each clustering script.

## Scripts

| File | Status | Purpose |
|---|---|---|
| `eda.py` | Done | Exploratory analysis — stats, skew, correlations, value counts |
| `preprocessing.py` | Done | Full pipeline, outputs `X_df` (12330×55) and `y` |

## How to Reuse the Preprocessed Data

```python
import sys
sys.stdout.reconfigure(encoding='utf-8')
# Run preprocessing inline (no separate module yet):
exec(open('preprocessing.py').read())
# X_df and y are now available
```

Or refactor `preprocessing.py` to expose `get_data()` if needed.

## Preprocessing — What Was Done (Final)

| Step | Action |
|---|---|
| 1 | Drop `Revenue` |
| 2 | Drop `BounceRates` (r=0.913 with ExitRates) |
| 3 | Drop count cols: `Administrative`, `Informational`, `ProductRelated` (r=0.60–0.86 with Duration cols) |
| 4 | Binarize `SpecialDay` → 0/1 (89.9% zeros) |
| 5 | Merge `VisitorType='Other'` → `'New_Visitor'`; group rare OS/Browser/TrafficType codes → `'Other'` |
| 6 | `np.log1p` on continuous cols |
| 7 | Clip continuous cols at p99 (post log) |
| 8 | `StandardScaler` on all numeric cols |
| 9 | `OneHotEncoder` on Month, VisitorType, OperatingSystems, Browser, Region, TrafficType |

**Final numeric cols (7):** `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`, `ExitRates`, `PageValues`, `SpecialDay`, `Weekend`

**Final shape:** 12,330 × 55 — no NaNs.

## Algorithms to Implement (in order)

1. **K-Means** — elbow method + silhouette to pick k, then fit and evaluate
2. **Agglomerative (Hierarchical)** — try Ward linkage; dendrogram optional
3. **HDBSCAN** — density-based (replaces DBSCAN); tune min_cluster_size
4. **Gaussian Mixture Models** — model-based; tune n_components

## Evaluation Metrics (apply to every algorithm)

- **Silhouette Score** — cluster cohesion/separation (-1 to 1, higher is better)
- **SSE / Inertia** — only for K-Means (minimise)
- **Revenue entropy per cluster** — purity metric; lower entropy = purer cluster
- **Revenue distribution per cluster** — % of buyers in each cluster

## Dataset Columns (original, for reference)

**Kept numeric (continuous):** `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`, `ExitRates`, `PageValues`

**Kept binary:** `SpecialDay` (binarized), `Weekend`

**Dropped:** `BounceRates`, `Administrative`, `Informational`, `ProductRelated`, `Revenue`

**Categorical (OHE'd):** `Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`
