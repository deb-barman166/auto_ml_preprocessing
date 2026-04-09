"""
test_engine.py  -  Comprehensive self-test suite.
Creates a deliberately messy synthetic dataset and runs the full pipeline.
Verifies every guarantee: zero NaN, all numeric, shape sanity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

# ── build a nasty synthetic dataset ───────────────────────────────────────────
np.random.seed(42)
N = 500

data = {
    # numeric - some NaN
    "age":          np.where(np.random.rand(N) < 0.1, np.nan, np.random.randint(18, 80, N).astype(float)),
    "salary":       np.where(np.random.rand(N) < 0.08, np.nan, np.random.exponential(50000, N)),

    # numeric string
    "score":        [str(round(np.random.uniform(0,100), 2)) if np.random.rand() > 0.05 else "N/A" for _ in range(N)],

    # boolean string variations
    "is_active":    np.random.choice(["True","False","yes","no","1","0"], N),

    # low cardinality categorical (→ one-hot)
    "gender":       [None if np.random.rand() < 0.05 else np.random.choice(["male","female","other"]) for _ in range(N)],

    # medium cardinality (→ label encode)
    "department":   np.random.choice([f"dept_{i}" for i in range(30)], N),

    # high cardinality (→ frequency encode)
    "user_id_str":  [f"user_{np.random.randint(0,400)}" for _ in range(N)],

    # datetime string
    "signup_date":  pd.date_range("2020-01-01", periods=N, freq="D").strftime("%Y-%m-%d").tolist(),

    # multi-label (comma-separated)
    "interests":    np.random.choice(
        ["AI,ML","Python,Data","Web,AI","ML,DL,Python","Data,Web","AI","Python"], N
    ),

    # constant column (should be dropped)
    "constant_col": ["same_value"] * N,

    # high-null column (should be dropped)
    "mostly_null":  np.where(np.random.rand(N) < 0.85, np.nan, np.random.rand(N)),

    # target variable (numeric, no NaN)
    "target":       np.random.randint(0, 2, N),

    # duplicate rows test — append 20 copies of first row
}

df_raw = pd.DataFrame(data)
# inject duplicates
df_raw = pd.concat([df_raw, df_raw.iloc[:20]], ignore_index=True)

print(f"Raw shape: {df_raw.shape}")
print(f"Raw dtypes:\n{df_raw.dtypes}\n")
print(f"Raw NaN counts:\n{df_raw.isna().sum()}\n")

# ── run the pipeline ───────────────────────────────────────────────────────────
from main import preprocess
from config import build_manual_config

config = build_manual_config(
    drop_columns=[],
    missing_strategies={"age": "median", "salary": "mean", "gender": "unknown"},
    encoding_strategies={"gender": "onehot", "department": "label", "user_id_str": "frequency"},
    derived_features=[
        {"type": "ratio", "col_a": "salary", "col_b": "age", "name": "salary_per_age"},
        {"type": "square", "col": "age"},
    ],
)
config["global"]["log_transform_skewed"] = True
config["global"]["normalize_text"]       = True

os.makedirs("output", exist_ok=True)
df_out, summary = preprocess(df_raw, config=config, output_path="output/processed.csv")

# ── assertions ─────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  AUTOMATED TEST ASSERTIONS")
print("="*55)

def check(condition, msg):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    return condition

all_pass = True
all_pass &= check(df_out.isna().sum().sum() == 0,          "Zero NaN values")
inf_check = df_out.select_dtypes(include=[np.number]).apply(pd.to_numeric, errors="coerce").apply(np.isinf).sum().sum()
all_pass &= check(inf_check == 0, "Zero Inf values")
all_pass &= check(len(df_out.select_dtypes(include=["object","category","bool"]).columns) == 0,
                                                            "No non-numeric columns")
all_pass &= check(df_out.shape[0] > 0,                     "Non-empty rows")
all_pass &= check(df_out.shape[1] > 0,                     "Non-empty columns")
all_pass &= check(df_out.shape[0] < df_raw.shape[0],       "Duplicates removed")
all_pass &= check("constant_col" not in df_out.columns,     "Constant column dropped")
all_pass &= check("mostly_null"  not in df_out.columns,     "High-null column dropped")
all_pass &= check("signup_date"  not in df_out.columns,     "Datetime column extracted")
all_pass &= check(any("signup_date_year" in c for c in df_out.columns),
                                                            "Datetime year feature created")
all_pass &= check("salary_per_age" in df_out.columns,      "Derived ratio feature created")
all_pass &= check("age_sq"         in df_out.columns,      "Derived square feature created")

print("="*55)
print(f"  Final shape   : {df_out.shape}")
print(f"  All columns numeric: {df_out.select_dtypes(include=[np.number]).shape[1] == df_out.shape[1]}")
print(f"  Overall       : {'ALL PASS ✅' if all_pass else 'SOME TESTS FAILED ❌'}")
print("="*55)
print(f"\n  Sample output (first 3 rows, first 8 cols):")
print(df_out.iloc[:3, :8].to_string())
