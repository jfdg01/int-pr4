import sys
sys.stdout.reconfigure(encoding='utf-8')

exec(open('preprocessing.py').read())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import entropy

# ── BIC/AIC + Silhouette sweep ────────────────────────────────────────────────
K_RANGE = range(2, 11)
bic = []
aic = []
sil = []

print(f"{'k':>3}  {'BIC':>12}  {'AIC':>12}  {'Silhouette':>10}")
for k in K_RANGE:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, max_iter=200)
    gmm.fit(X_df)
    labels = gmm.predict(X_df)
    b = gmm.bic(X_df)
    a = gmm.aic(X_df)
    s = silhouette_score(X_df, labels, sample_size=3000, random_state=42)
    bic.append(b)
    aic.append(a)
    sil.append(s)
    print(f"{k:>3}  {b:>12,.0f}  {a:>12,.0f}  {s:>10.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(list(K_RANGE), bic, marker='o', label='BIC')
axes[0].plot(list(K_RANGE), aic, marker='s', label='AIC')
axes[0].set_title('GMM — BIC & AIC vs k')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Score (lower is better)')
axes[0].legend()

axes[1].plot(list(K_RANGE), sil, marker='o', color='green')
axes[1].set_title('GMM — Silhouette Score vs k')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.savefig('gmm_selection.png', dpi=150)
plt.show()
print("Saved: gmm_selection.png")

# ── Pick best k: lowest BIC ───────────────────────────────────────────────────
BEST_K = list(K_RANGE)[bic.index(min(bic))]
print(f"\nChosen k={BEST_K} (lowest BIC)")

# ── Fit final model ───────────────────────────────────────────────────────────
gmm_final = GaussianMixture(n_components=BEST_K, covariance_type='full', random_state=42, max_iter=200)
gmm_final.fit(X_df)
labels = gmm_final.predict(X_df)

final_sil = silhouette_score(X_df, labels, sample_size=3000, random_state=42)
final_bic = gmm_final.bic(X_df)
final_aic = gmm_final.aic(X_df)

print(f"\n── Final GMM (k={BEST_K}, full covariance) ─────────────────")
print(f"BIC:              {final_bic:,.0f}")
print(f"AIC:              {final_aic:,.0f}")
print(f"Silhouette Score: {final_sil:.4f}")

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
    entropies.append(h)
    print(f"  Cluster {c}: n={n:5d}  buyers={pct_buyers:5.1f}%  entropy={h:.4f}")

print(f"\nAvg entropy: {np.mean(entropies):.4f}")

# ── Cluster sizes ─────────────────────────────────────────────────────────────
print(f"\n── Cluster Sizes ───────────────────────────────────────────")
print(df_res['cluster'].value_counts().sort_index().to_string())
