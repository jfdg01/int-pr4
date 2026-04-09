import sys
sys.stdout.reconfigure(encoding='utf-8')

exec(open('preprocessing.py').read())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy

# ── k-distance plot to pick eps ───────────────────────────────────────────────
# Use k = 2*n_features - 1 as a common heuristic for min_samples
MIN_SAMPLES = 10
k = MIN_SAMPLES - 1

nbrs = NearestNeighbors(n_neighbors=k).fit(X_df)
distances, _ = nbrs.kneighbors(X_df)
k_distances = np.sort(distances[:, -1])[::-1]

plt.figure(figsize=(8, 4))
plt.plot(k_distances)
plt.title(f'k-Distance Plot (k={k}) — elbow → eps')
plt.xlabel('Points (sorted by distance)')
plt.ylabel(f'{k}-NN distance')
plt.tight_layout()
plt.savefig('dbscan_kdistance.png', dpi=150)
plt.show()
print("Saved: dbscan_kdistance.png")

# ── Eps sweep ─────────────────────────────────────────────────────────────────
EPS_VALUES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

print(f"\n── eps sweep (min_samples={MIN_SAMPLES}) ───────────────────────────────")
print(f"{'eps':>5}  {'clusters':>8}  {'noise':>8}  {'noise%':>7}  {'silhouette':>10}")

results_sweep = []
for eps in EPS_VALUES:
    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
    labels = db.fit_predict(X_df)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    noise_pct = n_noise / len(labels) * 100

    if n_clusters >= 2:
        # Silhouette only on non-noise points
        mask = labels != -1
        if mask.sum() > 1:
            sil = silhouette_score(X_df[mask], labels[mask], sample_size=3000, random_state=42)
        else:
            sil = float('nan')
    else:
        sil = float('nan')

    results_sweep.append((eps, n_clusters, n_noise, noise_pct, sil))
    print(f"{eps:>5.1f}  {n_clusters:>8d}  {n_noise:>8d}  {noise_pct:>6.1f}%  {sil:>10.4f}")

# ── Pick best eps: most clusters with silhouette >= 0 and noise < 30% ─────────
import math
valid = [(eps, nc, nn, npct, s) for eps, nc, nn, npct, s in results_sweep
         if nc >= 2 and not math.isnan(s) and npct < 30]

if valid:
    best = max(valid, key=lambda x: x[4])
    BEST_EPS = best[0]
else:
    candidates = [(eps, nc, nn, npct, s) for eps, nc, nn, npct, s in results_sweep if nc >= 2]
    BEST_EPS = min(candidates, key=lambda x: x[3])[0] if candidates else EPS_VALUES[-1]

print(f"\nChosen eps={BEST_EPS}, min_samples={MIN_SAMPLES}")

# ── Fit final model ───────────────────────────────────────────────────────────
db_final = DBSCAN(eps=BEST_EPS, min_samples=MIN_SAMPLES)
labels = db_final.fit_predict(X_df)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
noise_pct = n_noise / len(labels) * 100

mask = labels != -1
if mask.sum() > 1 and n_clusters >= 2:
    final_sil = silhouette_score(X_df[mask], labels[mask], sample_size=3000, random_state=42)
else:
    final_sil = float('nan')

print(f"\n── Final DBSCAN (eps={BEST_EPS}, min_samples={MIN_SAMPLES}) ──────────────")
print(f"Clusters found:   {n_clusters}")
print(f"Noise points:     {n_noise} ({noise_pct:.1f}%)")
print(f"Silhouette Score: {final_sil:.4f}" if not np.isnan(final_sil) else "Silhouette Score: N/A")

# ── Evaluation: Revenue per cluster ──────────────────────────────────────────
df_res = pd.DataFrame({'cluster': labels, 'Revenue': y})

print(f"\n── Revenue Distribution per Cluster ────────────────────────")
entropies = []
for c in sorted(df_res['cluster'].unique()):
    grp = df_res[df_res['cluster'] == c]['Revenue']
    n = len(grp)
    pct_buyers = grp.mean() * 100
    p = grp.mean()
    probs = np.array([p, 1 - p])
    probs = probs[probs > 0]
    h = entropy(probs, base=2)
    label_str = 'NOISE' if c == -1 else str(c)
    if c != -1:
        entropies.append(h)
    print(f"  Cluster {label_str:>5}: n={n:5d}  buyers={pct_buyers:5.1f}%  entropy={h:.4f}")

if entropies:
    print(f"\nAvg entropy (excl. noise): {np.mean(entropies):.4f}")
