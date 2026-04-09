import sys
sys.stdout.reconfigure(encoding='utf-8')

exec(open('preprocessing.py').read())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.stats import entropy
from scipy.cluster.hierarchy import dendrogram, linkage

# ── Silhouette sweep over n_clusters ─────────────────────────────────────────
K_RANGE = range(2, 11)
sil = []

for k in K_RANGE:
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agg.fit_predict(X_df)
    s = silhouette_score(X_df, labels, sample_size=3000, random_state=42)
    sil.append(s)
    print(f"k={k:2d}  Silhouette={s:.4f}")

# ── Silhouette plot ───────────────────────────────────────────────────────────
plt.figure(figsize=(7, 4))
plt.plot(list(K_RANGE), sil, marker='o', color='steelblue')
plt.title('Agglomerative — Silhouette Score vs k (Ward linkage)')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.savefig('hierarchical_silhouette.png', dpi=150)
plt.show()
print("Saved: hierarchical_silhouette.png")

# ── Dendrogram on a subsample ─────────────────────────────────────────────────
np.random.seed(42)
sample_idx = np.random.choice(len(X_df), size=500, replace=False)
X_sample = X_df.iloc[sample_idx].values

Z = linkage(X_sample, method='ward')

plt.figure(figsize=(14, 5))
dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrogram (Ward, 500-row subsample, last 20 merges)')
plt.xlabel('Cluster')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('hierarchical_dendrogram.png', dpi=150)
plt.show()
print("Saved: hierarchical_dendrogram.png")

# ── Fit final model ───────────────────────────────────────────────────────────
# Pick best k from silhouette sweep
BEST_K = int(K_RANGE.start + sil.index(max(sil)))

agg_final = AgglomerativeClustering(n_clusters=BEST_K, linkage='ward')
labels = agg_final.fit_predict(X_df)

final_sil = silhouette_score(X_df, labels, sample_size=3000, random_state=42)

print(f"\n── Final Agglomerative (k={BEST_K}, Ward) ──────────────────")
print(f"Silhouette Score: {final_sil:.4f}")

# ── Evaluation: Revenue per cluster ──────────────────────────────────────────
results = pd.DataFrame({'cluster': labels, 'Revenue': y})

print(f"\n── Revenue Distribution per Cluster ────────────────────────")
entropies = []
for c in sorted(results['cluster'].unique()):
    grp = results[results['cluster'] == c]['Revenue']
    n = len(grp)
    pct_buyers = grp.mean() * 100
    p = grp.mean()
    probs = np.array([p, 1 - p])
    probs = probs[probs > 0]
    h = entropy(probs, base=2)
    entropies.append(h)
    print(f"  Cluster {c}: n={n:5d}  buyers={pct_buyers:5.1f}%  entropy={h:.4f}")

print(f"\nAvg entropy: {np.mean(entropies):.4f}")

# ── Cluster sizes ─────────────────────────────────────────────────────────────
print(f"\n── Cluster Sizes ───────────────────────────────────────────")
print(results['cluster'].value_counts().sort_index().to_string())
