import sys
sys.stdout.reconfigure(encoding='utf-8')

exec(open('preprocessing.py').read())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from scipy.stats import entropy

# ── min_cluster_size sweep ────────────────────────────────────────────────────
MCS_VALUES = [50, 100, 200, 300, 500, 750, 1000]

print(f"── min_cluster_size sweep ──────────────────────────────────────────────")
print(f"{'mcs':>6}  {'clusters':>8}  {'noise':>8}  {'noise%':>7}  {'silhouette':>10}")

results_sweep = []
for mcs in MCS_VALUES:
    hdb = HDBSCAN(min_cluster_size=mcs, min_samples=5)
    labels = hdb.fit_predict(X_df)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    noise_pct = n_noise / len(labels) * 100

    if n_clusters >= 2:
        mask = labels != -1
        if mask.sum() > 1:
            sil = silhouette_score(X_df[mask], labels[mask], sample_size=3000, random_state=42)
        else:
            sil = float('nan')
    else:
        sil = float('nan')

    results_sweep.append((mcs, n_clusters, n_noise, noise_pct, sil))
    print(f"{mcs:>6d}  {n_clusters:>8d}  {n_noise:>8d}  {noise_pct:>6.1f}%  "
          + (f"{sil:>10.4f}" if not np.isnan(sil) else f"{'N/A':>10}"))

# ── Pick best: >=2 clusters, noise <30%, max silhouette ───────────────────────
import math
valid = [(mcs, nc, nn, npct, s) for mcs, nc, nn, npct, s in results_sweep
         if nc >= 2 and not math.isnan(s) and npct < 30]

if valid:
    best = max(valid, key=lambda x: x[4])
    BEST_MCS = best[0]
else:
    candidates = [(mcs, nc, nn, npct, s) for mcs, nc, nn, npct, s in results_sweep if nc >= 2]
    BEST_MCS = min(candidates, key=lambda x: x[3])[0] if candidates else MCS_VALUES[2]

print(f"\nChosen min_cluster_size={BEST_MCS}")

# ── Fit final model ───────────────────────────────────────────────────────────
hdb_final = HDBSCAN(min_cluster_size=BEST_MCS, min_samples=5)
labels = hdb_final.fit_predict(X_df)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
noise_pct = n_noise / len(labels) * 100

mask = labels != -1
if mask.sum() > 1 and n_clusters >= 2:
    final_sil = silhouette_score(X_df[mask], labels[mask], sample_size=3000, random_state=42)
else:
    final_sil = float('nan')

print(f"\n── Final HDBSCAN (min_cluster_size={BEST_MCS}, min_samples=5) ──────────")
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
