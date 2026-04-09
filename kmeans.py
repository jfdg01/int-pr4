import sys
sys.stdout.reconfigure(encoding='utf-8')

exec(open('preprocessing.py').read())

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import entropy
import numpy as np
import pandas as pd

# ── Elbow + Silhouette sweep ──────────────────────────────────────────────────
K_RANGE = range(2, 11)
sse = []
sil = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_df)
    sse.append(km.inertia_)
    sil.append(silhouette_score(X_df, labels, sample_size=3000, random_state=42))
    print(f"k={k:2d}  SSE={km.inertia_:,.0f}  Silhouette={sil[-1]:.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(list(K_RANGE), sse, marker='o')
axes[0].set_title('Elbow — SSE vs k')
axes[0].set_xlabel('k')
axes[0].set_ylabel('SSE (Inertia)')

axes[1].plot(list(K_RANGE), sil, marker='o', color='orange')
axes[1].set_title('Silhouette Score vs k')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.savefig('kmeans_selection.png', dpi=150)
plt.show()
print("Saved: kmeans_selection.png")

# ── Fit final model ───────────────────────────────────────────────────────────
# k=6 chosen: highest silhouette (0.1860); SSE elbow at 4-5 but k=6 gives
# richer, more interpretable cluster structure with clearer Revenue separation.
BEST_K = 6

km_final = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
labels = km_final.fit_predict(X_df)

final_sil = silhouette_score(X_df, labels, sample_size=3000, random_state=42)
final_sse = km_final.inertia_

print(f"\n── Final K-Means (k={BEST_K}) ──────────────────────────────")
print(f"SSE (Inertia):    {final_sse:,.0f}")
print(f"Silhouette Score: {final_sil:.4f}")

# ── Evaluation: Revenue per cluster ──────────────────────────────────────────
results = pd.DataFrame({'cluster': labels, 'Revenue': y})

print(f"\n── Revenue Distribution per Cluster ────────────────────────")
for c in sorted(results['cluster'].unique()):
    grp = results[results['cluster'] == c]['Revenue']
    n = len(grp)
    pct_buyers = grp.mean() * 100
    # entropy: binary distribution [p_buyer, p_non_buyer]
    p = grp.mean()
    probs = np.array([p, 1 - p])
    probs = probs[probs > 0]
    h = entropy(probs, base=2)
    print(f"  Cluster {c}: n={n:5d}  buyers={pct_buyers:5.1f}%  entropy={h:.4f}")

# ── Cluster sizes ─────────────────────────────────────────────────────────────
print(f"\n── Cluster Sizes ───────────────────────────────────────────")
print(results['cluster'].value_counts().sort_index().to_string())
