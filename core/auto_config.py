"""
core/auto_config.py  –  Automatically generate processing config from raw data.
No ML / NLP. Purely rule-based column profiling. pandas 3.x compatible.
"""
from __future__ import annotations
import pandas as pd
from typing import Any
from utils.helpers import (
    is_bool_like, is_numeric_string, is_datetime_like,
    is_multilabel, missing_ratio, cardinality, normalize_str_dtypes,
)
from utils.logger import get_logger

log = get_logger("AutoConfig")

LOW_CARD_THRESH  = 10
MED_CARD_THRESH  = 50
NULL_DROP_THRESH = 0.60
VARIANCE_THRESH  = 1e-10


def profile_column(col: pd.Series) -> dict[str, Any]:
    """Return a diagnostic profile for a single column."""
    # normalise ArrowStringArray before profiling
    if str(col.dtype).lower() in ('string', 'str') or str(col.dtype).startswith('string'):
        col = col.astype(object)

    dtype    = str(col.dtype)
    n_total  = len(col)
    null_r   = col.isna().mean() if n_total else 0.0
    n_unique = int(col.nunique(dropna=True))
    is_const = n_unique <= 1

    profile: dict[str, Any] = {
        "dtype":         dtype,
        "null_count":    int(col.isna().sum()),
        "null_ratio":    round(float(null_r), 4),
        "unique_count":  n_unique,
        "is_constant":   is_const,
        "inferred_type": None,
        "encoding":      None,
        "action":        None,
    }

    if null_r > NULL_DROP_THRESH or is_const:
        profile["action"] = "drop"
        return profile

    profile["action"] = "keep"

    # ── semantic type inference ────────────────────────────────────────────────
    if col.dtype == bool or (col.dtype == object and is_bool_like(col)):
        profile["inferred_type"] = "boolean"
        profile["encoding"]      = "bool_to_int"

    elif col.dtype.kind in ("i", "u", "f") or (col.dtype == object and is_numeric_string(col)):
        profile["inferred_type"] = "numeric"
        profile["encoding"]      = "none"
        num = pd.to_numeric(col, errors="coerce")
        v   = num.var()
        if v is not None and float(v) < VARIANCE_THRESH:
            profile["action"]        = "drop"
            profile["inferred_type"] = "constant_numeric"

    elif col.dtype == object and is_datetime_like(col):
        profile["inferred_type"] = "datetime"
        profile["encoding"]      = "datetime_extract"

    elif str(col.dtype).startswith("datetime"):
        profile["inferred_type"] = "datetime"
        profile["encoding"]      = "datetime_extract"

    elif col.dtype == object and is_multilabel(col):
        profile["inferred_type"] = "multilabel"
        profile["encoding"]      = "multihot"

    elif col.dtype in ("object", "category"):
        if   n_unique <= LOW_CARD_THRESH:
            profile["inferred_type"] = "categorical_low"
            profile["encoding"]      = "onehot"
        elif n_unique <= MED_CARD_THRESH:
            profile["inferred_type"] = "categorical_med"
            profile["encoding"]      = "label"
        else:
            profile["inferred_type"] = "categorical_high"
            profile["encoding"]      = "frequency"
    else:
        profile["inferred_type"] = "unknown"
        profile["encoding"]      = "label"

    return profile


def generate_auto_config(df: pd.DataFrame, user_overrides: dict = None) -> dict:
    user_overrides = user_overrides or {}

    # normalise before profiling
    df = normalize_str_dtypes(df.copy())

    config: dict[str, Any] = {
        "columns": {},
        "global": {
            "drop_duplicates":        True,
            "null_drop_threshold":    NULL_DROP_THRESH,
            "variance_threshold":     VARIANCE_THRESH,
            "low_card_threshold":     LOW_CARD_THRESH,
            "med_card_threshold":     MED_CARD_THRESH,
            "normalize":              False,
            "log_transform_skewed":   False,
            "datetime_include_time":  False,
            "normalize_text":         True,
        },
    }

    log.info(f"Profiling {len(df.columns)} columns …")

    for col_name in df.columns:
        profile = profile_column(df[col_name])
        col_cfg = dict(profile)

        if col_name in user_overrides:
            col_cfg.update(user_overrides[col_name])
            log.info(f"  [{col_name}] — user override applied: {user_overrides[col_name]}")
        else:
            log.info(
                f"  [{col_name}] — type={col_cfg['inferred_type']}, "
                f"encoding={col_cfg['encoding']}, action={col_cfg['action']}, "
                f"null_ratio={col_cfg['null_ratio']}"
            )

        config["columns"][col_name] = col_cfg

    return config
