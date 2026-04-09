"""
agents/cleaning.py  –  CleaningAgent
Handles: duplicate removal, whitespace, column drops (null / constant / low-var),
text normalisation, pandas 3.x ArrowStringArray normalisation.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from agents.base import BaseAgent
from utils.helpers import missing_ratio, normalize_str_dtypes


class CleaningAgent(BaseAgent):

    def run(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        df = df.copy()

        # ── 0a. normalise pandas 3.x ArrowStringArray → object dtype ──────────
        df = normalize_str_dtypes(df)

        gcfg  = config.get("global", {})
        ccfg  = config.get("columns", {})
        self.report.update({"dropped_cols": [], "dropped_rows": 0,
                            "renamed_cols": [], "whitespace_cleaned": []})

        # ── 0b. strip column name whitespace ──────────────────────────────────
        df.columns = [str(c).strip() for c in df.columns]

        # ── 1. drop user-specified columns ────────────────────────────────────
        user_drops = config.get("cleaning", {}).get("drop_columns", [])
        existing   = [c for c in user_drops if c in df.columns]
        if existing:
            df.drop(columns=existing, inplace=True)
            self.report["dropped_cols"].extend(existing)
            self.log.info(f"User-forced drop: {existing}")

        # ── 2. remove duplicate rows ──────────────────────────────────────────
        if gcfg.get("drop_duplicates", True):
            before = len(df)
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)
            removed = before - len(df)
            self.report["dropped_rows"] = removed
            if removed:
                self.log.info(f"Removed {removed} duplicate rows.")

        # ── 3. drop columns marked 'drop' in config ───────────────────────────
        null_thr = gcfg.get("null_drop_threshold", 0.60)

        for col in list(df.columns):
            col_meta = ccfg.get(col, {})
            action   = col_meta.get("action", "keep")

            if action == "drop":
                df.drop(columns=[col], inplace=True)
                self.report["dropped_cols"].append(col)
                self.log.info(f"Dropped [{col}] — action=drop (null_ratio={col_meta.get('null_ratio','?')})")
                continue

            if col not in df.columns:
                continue

            nr = missing_ratio(df[col])
            if nr > null_thr:
                df.drop(columns=[col], inplace=True)
                self.report["dropped_cols"].append(col)
                self.log.warning(f"Dropped [{col}] — null_ratio={nr:.2%} > {null_thr:.0%}")
                continue

            if df[col].nunique(dropna=True) <= 1:
                df.drop(columns=[col], inplace=True)
                self.report["dropped_cols"].append(col)
                self.log.warning(f"Dropped [{col}] — constant column")
                continue

        # ── 4. string whitespace trim + optional lowercase ────────────────────
        normalize_text = config.get("cleaning", {}).get("normalize_text",
                         config.get("global", {}).get("normalize_text", True))
        for col in df.select_dtypes(include="object").columns:
            orig = df[col].copy()
            df[col] = df[col].astype(str).where(df[col].notna(), other=None)
            df[col] = df[col].str.strip()
            if normalize_text:
                df[col] = df[col].str.lower()
            # pandas 3.x str accessor may return ArrowStringArray – force back to object
            df[col] = df[col].astype(object)
            self.report["whitespace_cleaned"].append(col)

        # ── 5. replace bare "none" / "nan" strings ────────────────────────────
        df.replace("none", np.nan, inplace=True)
        df.replace("nan",  np.nan, inplace=True)

        self.log.info(
            f"Cleaning done. Dropped cols: {len(self.report['dropped_cols'])}, "
            f"Dropped rows: {self.report['dropped_rows']}"
        )
        return df
