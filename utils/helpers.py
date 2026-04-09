"""
utils/helpers.py  –  Pure utility functions (no ML, fully deterministic).
"""
from __future__ import annotations
import re, numpy as np, pandas as pd
from typing import Any

# ── type sniffing ──────────────────────────────────────────────────────────────

BOOL_TRUE_SET  = {"true", "yes", "1", "y", "t", "on"}
BOOL_FALSE_SET = {"false", "no", "0", "n", "f", "off"}

def is_bool_like(series: pd.Series) -> bool:
    """Return True if a string column contains only boolean-representable values."""
    clean = series.dropna().astype(str).str.strip().str.lower()
    return clean.isin(BOOL_TRUE_SET | BOOL_FALSE_SET).all() and len(clean) > 0

def is_numeric_string(series: pd.Series) -> bool:
    """Return True if a string column can be safely cast to float."""
    clean = series.dropna().astype(str).str.strip()
    if len(clean) == 0:
        return False
    try:
        pd.to_numeric(clean, errors="raise")
        return True
    except (ValueError, TypeError):
        return False

def is_datetime_like(series: pd.Series, sample: int = 200) -> bool:
    """Heuristically detect datetime strings via regex patterns."""
    DATE_RE = re.compile(
        r"""(\d{4}[-/]\d{1,2}[-/]\d{1,2})|   # YYYY-MM-DD
            (\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|  # DD-MM-YYYY
            (\d{4}-\d{2}-\d{2}T\d{2}:\d{2})   # ISO-8601
        """, re.VERBOSE)
    subset = series.dropna().astype(str).head(sample)
    if len(subset) == 0:
        return False
    hits = subset.apply(lambda x: bool(DATE_RE.search(x.strip()))).sum()
    return (hits / len(subset)) >= 0.8

def is_multilabel(series: pd.Series, sep: str = ",") -> bool:
    """Detect if most non-null values contain the separator (multi-label)."""
    clean = series.dropna().astype(str).str.strip()
    if len(clean) == 0:
        return False
    frac = clean.str.contains(sep, regex=False).mean()
    return frac >= 0.5

# ── series helpers ─────────────────────────────────────────────────────────────

def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def bool_series_to_int(series: pd.Series) -> pd.Series:
    mapping = {**{k: 1 for k in BOOL_TRUE_SET}, **{k: 0 for k in BOOL_FALSE_SET}}
    return series.astype(str).str.strip().str.lower().map(mapping).astype("Int64")

def variance(series: pd.Series) -> float:
    try:
        return float(series.var(numeric_only=True))
    except Exception:
        return 0.0

def missing_ratio(series: pd.Series) -> float:
    return series.isna().mean()

def cardinality(series: pd.Series) -> int:
    return series.nunique(dropna=True)

# ── dataframe helpers ──────────────────────────────────────────────────────────

def replace_inf(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ±inf with NaN so downstream imputation catches them."""
    return df.replace([np.inf, -np.inf], np.nan)

def memory_usage_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1024 ** 2

def downcast_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast int64/float64 to smaller types for memory efficiency."""
    for col in df.select_dtypes(include=["int64", "Int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer", errors="ignore")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float", errors="ignore")
    return df


def normalize_str_dtypes(df) -> "pd.DataFrame":
    """
    Convert pandas 3.x ArrowStringArray / StringDtype columns to plain object.
    Must be called before any str.contains / .unique() / sort operations.
    """
    import pandas as pd
    for col in df.columns:
        dtype_str = str(df[col].dtype).lower()
        if dtype_str in ('string', 'str') or dtype_str.startswith('string'):
            df[col] = df[col].astype(object)
    return df
