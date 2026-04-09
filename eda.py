import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

df = pd.read_excel('online_shoppers_intention.xlsx')

# ── Basic shape ──────────────────────────────────────────────────────────────
print(f"Shape: {df.shape}")
print(f"\nDtypes:\n{df.dtypes}")

# ── Nulls ────────────────────────────────────────────────────────────────────
print(f"\nNulls:\n{df.isnull().sum()}")

# ── Numeric columns ──────────────────────────────────────────────────────────
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['str', 'bool']).columns.tolist()

print(f"\nNumeric cols: {num_cols}")
print(f"Categorical cols: {cat_cols}")

# ── Descriptive stats ────────────────────────────────────────────────────────
print("\n── Numeric descriptive stats ──")
desc = df[num_cols].describe().T
desc['skew'] = df[num_cols].skew()
desc['kurt'] = df[num_cols].kurt()
print(desc[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurt']].to_string())

# ── Outlier check: values beyond 99th percentile ────────────────────────────
print("\n── Outliers (count above p99) ──")
for col in num_cols:
    p99 = df[col].quantile(0.99)
    n_out = (df[col] > p99).sum()
    print(f"  {col}: p99={p99:.2f}, outliers={n_out}")

# ── Categorical value counts ─────────────────────────────────────────────────
print("\n── Categorical value counts ──")
for col in cat_cols:
    vc = df[col].value_counts()
    print(f"\n{col}:")
    print(vc.to_string())

# ── Quasi-categorical columns (int but really categories) ───────────────────
quasi = ['OperatingSystems', 'Browser', 'Region', 'TrafficType']
print("\n── Quasi-categorical unique values & counts ──")
for col in quasi:
    vc = df[col].value_counts().sort_index()
    print(f"\n{col} ({df[col].nunique()} unique):")
    print(vc.to_string())

# ── Correlation matrix (numeric only) ───────────────────────────────────────
print("\n── Pearson correlation (|r| > 0.5) ──")
corr = df[num_cols].corr().abs()
pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
             .stack()
             .sort_values(ascending=False))
print(pairs[pairs > 0.5].to_string())

# ── Zero-inflation check ─────────────────────────────────────────────────────
print("\n── Zero % per numeric column ──")
for col in num_cols:
    pct = (df[col] == 0).mean() * 100
    print(f"  {col}: {pct:.1f}%")

# ── Revenue distribution ──────────────────────────────────────────────────────
print(f"\n── Revenue distribution ──")
print(df['Revenue'].value_counts())
print(f"Purchase rate: {df['Revenue'].mean()*100:.1f}%")
