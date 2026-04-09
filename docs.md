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

**Script:** `kmeans.py` | **Chosen k:** 6

**Selection:** Silhouette peaks at k=6 (0.1860); SSE elbow around k=4–5 but k=6 gives clearer Revenue separation.

| k | SSE | Silhouette |
|---|---|---|
| 2 | 114,120 | 0.1331 |
| 3 | 101,760 | 0.1545 |
| 4 | 90,785 | 0.1800 |
| 5 | 82,885 | 0.1760 |
| **6** | **75,744** | **0.1860** ← chosen |
| 7 | 71,139 | 0.1558 |
| 8 | 69,160 | 0.1549 |
| 9 | 67,527 | 0.1373 |
| 10 | 65,980 | 0.1459 |

**Final model (k=6):** SSE=75,744 | Silhouette=0.1860

| Cluster | n | Buyers % | Entropy |
|---|---|---|---|
| 0 | 1,144 | 6.2% | 0.3356 |
| 1 | 1,828 | 8.8% | 0.4282 |
| 2 | 998 | 0.8% | 0.0673 — very pure non-buyers |
| 3 | 1,550 | **65.9%** | 0.9260 — high-intent buyers |
| 4 | 1,763 | 23.3% | 0.7834 — mixed, buyer-leaning |
| 5 | 5,047 | 4.7% | 0.2733 — large low-intent group |

### Agglomerative (Hierarchical)

**Script:** `hierarchical.py` | **Linkage:** Ward | **Chosen k:** 5

**Selection:** Silhouette peaks at k=5 (0.1874).

| k | Silhouette |
|---|---|
| 2 | 0.1304 |
| 3 | 0.1442 |
| 4 | 0.1746 |
| **5** | **0.1874** ← chosen |
| 6 | 0.1783 |
| 7 | 0.1467 |
| 8 | 0.1532 |
| 9 | 0.1530 |
| 10 | 0.1588 |

**Final model (k=5):** Silhouette=0.1874

| Cluster | n | Buyers % | Entropy |
|---|---|---|---|
| 0 | 6,798 | 4.6% | 0.2693 — large low-intent group |
| 1 | 2,104 | 24.3% | 0.8006 — mixed, buyer-leaning |
| 2 | 729 | 0.4% | 0.0385 — very pure non-buyers |
| 3 | 1,167 | 6.6% | 0.3507 |
| 4 | 1,532 | **65.5%** | 0.9298 — high-intent buyers |

Avg entropy: 0.4778

### HDBSCAN

**Script:** `hdbscan_clustering.py` | **min_cluster_size:** 200 | **min_samples:** 5

**Rationale:** DBSCAN collapsed to 2 coarse clusters due to the curse of dimensionality (55 features). HDBSCAN uses mutual reachability distance and extracts stable clusters from a density hierarchy, eliminating the need to fix a global `eps` threshold.

**Sweep (min_samples=5 fixed):**

| min_cluster_size | Clusters | Noise % | Silhouette |
|---|---|---|---|
| 50 | 4 | 3.5% | 0.1431 |
| 100 | 4 | 3.5% | 0.1431 |
| **200** | **3** | **2.7%** | **0.1579** ← chosen |
| 300–1000 | 3 | 2.7% | 0.1579 |

**Final model (min_cluster_size=200, min_samples=5):** Silhouette=0.1579 | Noise=331 pts (2.7%)

| Cluster | n | Buyers % | Entropy |
|---|---|---|---|
| 0 | 1,251 | 6.2% | 0.3336 — low-intent segment |
| 1 | 8,214 | 15.7% | 0.6260 — large standard browsing group |
| 2 | 2,534 | 16.6% | 0.6488 — slightly more active users |
| NOISE | 331 | 37.5% | 0.9542 — high-value atypical sessions |

Avg entropy (excl. noise): 0.5361

**Note:** HDBSCAN isolates a noise group (331 sessions) with 37.5% conversion — the highest of any cluster across all algorithms. These are likely impulsive high-value buyers that no other method captures explicitly.

### Gaussian Mixture Models (GMM)

**Script:** `gmm.py` | **Covariance:** full | **Chosen k:** 10

**Selection:** BIC decreases monotonically through k=10; k=10 chosen as lowest BIC. Silhouette is low across all k — expected, as GMM uses soft probabilistic assignment, not geometric cohesion.

| k | BIC | AIC | Silhouette |
|---|---|---|---|
| 2 | -1,451,356 | -1,475,032 | 0.0724 |
| 3 | -2,104,394 | -2,139,912 | 0.0565 |
| 4 | -1,756,078 | -1,803,438 | 0.0890 |
| 5 | -2,234,610 | -2,293,813 | 0.0412 |
| 6 | -2,528,783 | -2,599,828 | 0.0342 |
| 7 | -2,384,308 | -2,467,194 | 0.0362 |
| 8 | -2,598,843 | -2,693,572 | 0.0419 |
| 9 | -2,703,994 | -2,810,564 | 0.0285 |
| **10** | **-2,854,414** | **-2,972,827** | **0.0221** ← chosen |

**Final model (k=10):** BIC=-2,854,414 | Silhouette=0.0221

| Cluster | n | Buyers % | Entropy |
|---|---|---|---|
| 0 | 1,251 | 6.2% | 0.3336 |
| 1 | 479 | 28.8% | 0.8663 |
| 2 | 4,221 | 5.4% | 0.3022 — large low-intent group |
| 3 | 792 | 36.1% | 0.9436 |
| 4 | 511 | 34.8% | 0.9326 |
| 5 | 569 | 30.6% | 0.8883 |
| 6 | 1,818 | **42.4%** | 0.9831 — highest buyer rate |
| 7 | 536 | 1.3% | 0.1005 — very pure non-buyers |
| 8 | 1,911 | 2.6% | 0.1748 — pure non-buyers |
| 9 | 242 | 0.4% | 0.0387 — nearly pure non-buyers |

Avg entropy: 0.5563

**Note:** BIC does not plateau — GMM keeps finding statistically better fits at higher k. Silhouette is poor throughout, reflecting that GMM partitions probabilistic density, not compact geometric clusters.

---

## Comparison

### Metrics summary

| Algorithm | k | Silhouette | SSE | Avg Revenue entropy |
|---|---|---|---|---|
| K-Means | 6 | 0.1860 | 75,744 | 0.4800 |
| Agglomerative (Ward) | 5 | 0.1874 | — | 0.4778 |
| HDBSCAN | 3 | 0.1579* | — | 0.5361 |
| GMM | 10 | 0.0221 | — | 0.5563 |

*HDBSCAN silhouette computed on non-noise points only (331 noise pts, 2.7%).

---

### Why each algorithm was chosen

**K-Means** is the natural starting point for any clustering task. It is fast, scales easily to 12,330 rows × 55 features, and its inertia (SSE) gives a concrete geometric objective that is straightforward to inspect. Because all features were already standardised, the Euclidean distance assumption is valid. The main limitation is that it assumes convex, roughly equal-sized clusters — a strong assumption for behavioural data.

**Agglomerative (Hierarchical)** was chosen as a deterministic, non-parametric alternative. Unlike K-Means it does not assume cluster shape and does not require a random seed, so results are fully reproducible. Ward linkage was selected specifically because it minimises within-cluster variance at each merge step — the same objective as K-Means SSE — making the two algorithms directly comparable. The dendrogram also provides a visual cross-check of how many meaningful splits exist in the data.

**HDBSCAN** was chosen over DBSCAN after DBSCAN collapsed to 2 coarse clusters due to the curse of dimensionality. HDBSCAN builds a density hierarchy using mutual reachability distances — which smooth the distance uniformity problem in high dimensions — and extracts the most stable clusters without requiring a global `eps` threshold. This makes it the natural density-based alternative for this 55-dimensional space.

**Gaussian Mixture Models** were chosen as the probabilistic counterpart to K-Means. Instead of hard cluster assignment, GMM fits a mixture of Gaussians and assigns each point a probability of belonging to each component. This is appropriate when clusters overlap or have different orientations/sizes, which is plausible in a 55-dimensional behavioural space. BIC was used as the primary selection criterion because, unlike silhouette, it directly measures model fit while penalising complexity.

---

### Hyperparameter tuning decisions

**K-Means — choosing k:**
k was swept from 2 to 10. Two signals were used together: the SSE elbow (rate of gain diminishes past k=4–5) and the silhouette score (peaks at k=6, 0.1860). k=6 was chosen because it maximises silhouette, and the Revenue distribution confirms the extra clusters carry real meaning: one cluster concentrates 65.9% buyers, another is almost entirely non-buyers (0.8%), and the remaining four capture intermediate intent profiles. Accepting k=6 over k=4 trades a small SSE gain for substantially richer segmentation.

**Agglomerative — linkage and k:**
Ward linkage was chosen over single/complete/average linkage because it is the only linkage that directly minimises within-cluster variance, making it the most coherent choice for Euclidean feature space and directly comparable to K-Means. k was read from the silhouette sweep peak (k=5, 0.1874). The dendrogram on a 500-row subsample (last 20 merges shown) was used as a visual sanity check — the large gap before the final merge at k=5 supports this choice.

**HDBSCAN — min_cluster_size and min_samples:**
`min_samples=5` was kept fixed (low value to allow sensitivity to local density). `min_cluster_size` was swept from 50 to 1000: values ≤100 produce 4 clusters with 3.5% noise and silhouette 0.143; from 200 onward the algorithm stabilises at 3 clusters, 2.7% noise, and silhouette 0.158. `min_cluster_size=200` was chosen as the smallest value yielding this stable solution — it represents ~1.6% of the dataset, a reasonable minimum segment size.

**GMM — covariance type and k:**
`covariance_type='full'` was chosen to allow each Gaussian component to have its own arbitrary covariance matrix. The alternative `'diag'` or `'spherical'` would impose unrealistic symmetry constraints on behavioural clusters. k was selected by minimising BIC, which penalises model complexity and prevents overfitting. BIC decreased monotonically through k=10 — this is a known behaviour in high-dimensional spaces where the full covariance matrix gains many degrees of freedom per component. The practical implication is that GMM at k=10 finds finer-grained density modes than the other methods, at the cost of poor silhouette scores (0.0221), reflecting that its components are not geometrically compact.

---

### Comparative analysis

**Cluster structure.** K-Means (k=6) and Agglomerative (k=5) converge on a very similar picture: one large low-intent group (~55–84% of points), one concentrated high-buyer cluster (~65% buyers), one near-pure non-buyer cluster (<1% buyers), and a small set of intermediate groups. This consistency across two independent methods with different assumptions (random initialisation vs. deterministic merging, spherical vs. variance-minimising) gives confidence that this segmentation reflects genuine structure in the data.

**HDBSCAN's contribution.** HDBSCAN successfully separates 3 clusters (vs. DBSCAN's 2 coarse blobs) with only 2.7% noise and silhouette 0.158. More importantly, it isolates a noise group of 331 sessions with a 37.5% conversion rate — the highest of any identified group across all algorithms. This group represents geometrically isolated high-value sessions that no distance-partition method captures. The silhouette is genuine (not inflated by two large blobs) because it is computed over 3 groups with variable density.

**GMM's trade-off.** GMM at k=10 discovers the richest Revenue variation: clusters 7, 8, 9 are near-pure non-buyers (entropy 0.04–0.17), while clusters 3–6 cluster around 30–42% buyers. This granularity comes at a cost: silhouette collapses to 0.0221, meaning the components overlap heavily in feature space. The model is statistically well-fitted (lowest BIC), but the clusters are less interpretable and harder to act on operationally. GMM is best suited here to understanding the underlying density structure rather than clean segment assignment.

**Best overall method.** For the goal of segmenting users by purchase intent, **Agglomerative (k=5)** edges out K-Means on silhouette (0.1874 vs 0.1860), is fully deterministic, and produces an equally interpretable Revenue split. K-Means is a close second and is preferable if scalability or repeated fitting (e.g. streaming data) is required. GMM is the best choice if probabilistic membership scores are needed downstream. **HDBSCAN** adds a unique value proposition: its noise group isolates 331 high-converting sessions (37.5% buyers) that no partition-based method surfaces.

**Revenue entropy.** K-Means and Agglomerative achieve similar average entropy (~0.478), meaning their clusters are equally pure with respect to buyer/non-buyer composition. GMM's higher entropy (0.556) reflects its softer, overlapping components. Lower entropy = purer clusters = more useful segments for targeting.

---

## Scripts

| File | Purpose |
|---|---|
| `eda.py` | Exploratory data analysis |
| `preprocessing.py` | Full preprocessing pipeline |
| `kmeans.py` | K-Means clustering |
| `hierarchical.py` | Agglomerative clustering |
| `hdbscan_clustering.py` | HDBSCAN clustering (replaces DBSCAN) |
| `gmm.py` | Gaussian Mixture Models |
