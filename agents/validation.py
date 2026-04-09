"""
agents/validation.py  –  ValidationAgent
Final quality gate — ensures output is ML-ready.
Checks:
  - Zero NaN values
  - Zero ±Inf values
  - No non-numeric columns
  - Shape sanity
  - Duplicate column names
  - Memory usage report
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from agents.base import BaseAgent
from utils.helpers import memory_usage_mb


class ValidationAgent(BaseAgent):

    def run(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        df     = df.copy()
        errors = []
        warns  = []

        # ── 1. no non-numeric columns ──────────────────────────────────────────
        bad_dtypes = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if bad_dtypes:
            errors.append(f"Non-numeric columns remain: {bad_dtypes}")

        bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
        if bool_cols:
            self.log.warning(f"Bool columns found — converting to int: {bool_cols}")
            for c in bool_cols:
                df[c] = df[c].astype(int)

        # ── 2. NaN check ───────────────────────────────────────────────────────
        nan_counts = df.isna().sum()
        nan_cols   = nan_counts[nan_counts > 0].to_dict()
        if nan_cols:
            errors.append(f"NaN values remain: {nan_cols}")
            # auto-fix
            self.log.warning("Auto-fixing remaining NaNs → 0")
            df.fillna(0, inplace=True)

        # ── 3. Inf check ──────────────────────────────────────────────────────
        num_df  = df.select_dtypes(include=[np.number]).astype(float)
        inf_mask = np.isinf(num_df).any()
        inf_cols  = inf_mask[inf_mask].index.tolist()
        if inf_cols:
            warns.append(f"Inf values found in: {inf_cols} — replacing with NaN→0")
            df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        # ── 4. duplicate column names ──────────────────────────────────────────
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        if dup_cols:
            warns.append(f"Duplicate column names: {dup_cols}")
            df.columns = pd.io.parsers.base_parser.ParserBase({
                "names": df.columns, "usecols": None
            })._maybe_dedup_names(df.columns)

        # ── 5. all-zero columns ────────────────────────────────────────────────
        zero_cols = [c for c in df.columns if (df[c] == 0).all()]
        if zero_cols:
            warns.append(f"All-zero columns detected (may be noise): {zero_cols}")

        # ── 6. shape sanity ────────────────────────────────────────────────────
        if df.shape[0] == 0:
            errors.append("DataFrame has 0 rows after processing!")
        if df.shape[1] == 0:
            errors.append("DataFrame has 0 columns after processing!")

        # ── 7. log summary ────────────────────────────────────────────────────
        mem = memory_usage_mb(df)
        self.log.info(f"Memory usage: {mem:.2f} MB")
        self.log.info(f"Final shape: {df.shape}")
        self.log.info(f"Dtypes: {df.dtypes.value_counts().to_dict()}")

        for w in warns:
            self.log.warning(w)

        if errors:
            for e in errors:
                self.log.error(f"VALIDATION FAILED: {e}")
            # still return best-effort df but mark in report
        else:
            self.log.info("Validation PASSED — dataset is fully ML-ready.")

        self.report.update({
            "validation_errors":   errors,
            "validation_warnings": warns,
            "final_shape":         df.shape,
            "memory_mb":           round(mem, 3),
            "dtypes_summary":      df.dtypes.value_counts().to_dict(),
        })
        return df
