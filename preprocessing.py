import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_excel('online_shoppers_intention.xlsx')
y = df['Revenue'].astype(int).reset_index(drop=True)   # kept aside for evaluation

# ── Step 1: Drop Revenue ─────────────────────────────────────────────────────
data = df.drop(columns=['Revenue']).copy()

# ── Step 2: Drop BounceRates (r=0.913 with ExitRates, nearly identical) ──────
data = data.drop(columns=['BounceRates'])

# ── Step 3: Drop count cols, keep Duration counterparts ──────────────────────
# Administrative/Informational/ProductRelated correlate 0.60-0.86 with their
# Duration cols. Durations carry more info (time on page, not just click count).
data = data.drop(columns=['Administrative', 'Informational', 'ProductRelated'])

# ── Step 4: Binarize SpecialDay (89.9% zeros, near-constant) ─────────────────
data['SpecialDay'] = (data['SpecialDay'] > 0).astype(int)

# ── Step 5: Merge rare categories ────────────────────────────────────────────
# VisitorType: 'Other' (85 rows) -> 'New_Visitor'
data['VisitorType'] = data['VisitorType'].replace('Other', 'New_Visitor')

# OperatingSystems: group {5,6,7,8} (all <=79 rows) -> 'Other'
data['OperatingSystems'] = data['OperatingSystems'].astype(str)
data['OperatingSystems'] = data['OperatingSystems'].replace({'5': 'Other', '6': 'Other', '7': 'Other', '8': 'Other'})

# Browser: group values with <50 rows -> 'Other'
browser_counts = df['Browser'].value_counts()
rare_browsers = browser_counts[browser_counts < 50].index.astype(str).tolist()
data['Browser'] = data['Browser'].astype(str)
data['Browser'] = data['Browser'].apply(lambda x: 'Other' if x in rare_browsers else x)

# TrafficType: group values with <50 rows -> 'Other'
traffic_counts = df['TrafficType'].value_counts()
rare_traffic = traffic_counts[traffic_counts < 50].index.astype(str).tolist()
data['TrafficType'] = data['TrafficType'].astype(str)
data['TrafficType'] = data['TrafficType'].apply(lambda x: 'Other' if x in rare_traffic else x)

# Region: balanced, keep as-is but convert to string for OHE
data['Region'] = data['Region'].astype(str)

# ── Define column roles after transformations ─────────────────────────────────
continuous_cols = [
    'Administrative_Duration',
    'Informational_Duration',
    'ProductRelated_Duration',
    'ExitRates',
    'PageValues',
]
binary_cols = ['SpecialDay', 'Weekend']   # already 0/1 after step 4; Weekend is bool
categorical_cols = ['Month', 'VisitorType', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']

# ── Step 6: log1p on skewed continuous cols ───────────────────────────────────
for col in continuous_cols:
    data[col] = np.log1p(data[col])

# ── Step 7: Clip at 99th percentile (post log-transform) ─────────────────────
for col in continuous_cols:
    p99 = data[col].quantile(0.99)
    data[col] = data[col].clip(upper=p99)

# ── Step 8 + 9: Scale numerics, OHE categoricals ─────────────────────────────
data['Weekend'] = data['Weekend'].astype(int)   # bool -> 0/1

numeric_to_scale = continuous_cols + binary_cols

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_to_scale),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
], remainder='drop')

X = preprocessor.fit_transform(data)

# ── Build final DataFrame ─────────────────────────────────────────────────────
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
all_feature_names = numeric_to_scale + cat_feature_names

X_df = pd.DataFrame(X, columns=all_feature_names)

# ── Verification ──────────────────────────────────────────────────────────────
print(f"Final shape:  {X_df.shape}")
print(f"Numeric cols: {numeric_to_scale}")
print(f"OHE cols:     {categorical_cols}")
print(f"\nSample stats (scaled numerics):")
print(X_df[numeric_to_scale].describe().round(3).to_string())
print(f"\nAny NaN: {X_df.isnull().any().any()}")
print(f"\nRevenue (y) distribution:\n{pd.Series(y).value_counts().to_string()}")
print("\nPreprocessing complete.")
