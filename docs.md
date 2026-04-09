# Clustering – Online Shoppers Intention

## Dataset

- **Source:** `online_shoppers_intention.xlsx`
- **Size:** 12,330 rows × 18 columns
- **Task:** Unsupervised clustering to segment users by purchase intent
- **Target (evaluation only):** `Revenue` (15.5% positive / purchase sessions)

---

## EDA Findings (`eda.py`)

| Finding | Detail |
|---|---|
| Nulls | None |
| Skewed cols | All duration cols skew > 5; PageValues skew 6.4 |
| Correlated pairs | BounceRates/ExitRates r=0.91; count/duration pairs r=0.60–0.86 |
| Zero inflation | SpecialDay 89.9%, Informational 78–80%, PageValues 77.9% |
| Rare categories | VisitorType 'Other' = 85 rows; several OS/Browser/TrafficType codes < 50 rows |

---

## Preprocessing (`preprocessing.py`)

**Output:** `X_df` (12,330 × 55), `y` (Revenue int, for evaluation only)

| Step | Action | Reason |
|---|---|---|
| 1 | Drop `Revenue` | Target — excluded from clustering |
| 2 | Drop `BounceRates` | r=0.913 with ExitRates — near-duplicate |
| 3 | Drop count cols (`Administrative`, `Informational`, `ProductRelated`) | r=0.60–0.86 with Duration counterparts; durations more informative |
| 4 | Binarize `SpecialDay` → 0/1 | 89.9% zeros — useless as continuous |
| 5 | Merge rare categories (VisitorType, OS, Browser, TrafficType) | Sparse codes add noise |
| 6 | `np.log1p` on continuous cols | Reduces heavy right-skew |
| 7 | Clip at p99 (post log) | Removes residual extremes |
| 8 | `StandardScaler` | Required for distance-based clustering |
| 9 | `OneHotEncoder` on all nominal cols | Correct encoding for non-ordinal variables |

---

## Clustering Results

### K-Means
*To be completed.*

### Agglomerative (Hierarchical)
*To be completed.*

### DBSCAN
*To be completed.*

### Gaussian Mixture Models
*To be completed.*

---

## Comparison

*To be completed after all algorithms are run.*

| Algorithm | k | Silhouette | SSE | Revenue entropy (avg) |
|---|---|---|---|---|
| K-Means | - | - | - | - |
| Agglomerative | - | - | — | - |
| DBSCAN | - | - | — | - |
| GMM | - | - | — | - |

---

## Scripts

| File | Purpose |
|---|---|
| `eda.py` | Exploratory data analysis |
| `preprocessing.py` | Full preprocessing pipeline |
| *(next)* `kmeans.py` | K-Means clustering |
| *(next)* `hierarchical.py` | Agglomerative clustering |
| *(next)* `dbscan.py` | DBSCAN clustering |
| *(next)* `gmm.py` | Gaussian Mixture Models |
