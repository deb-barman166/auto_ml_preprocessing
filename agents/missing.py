"""
agents/missing.py  –  MissingValueAgent
Guarantees zero NaN after execution.
Strategies: median / mean / mode / constant / ffill / bfill / drop_row
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from agents.base import BaseAgent
from utils.helpers import normalize_str_dtypes


class MissingValueAgent(BaseAgent):

    # default strategy per dtype family
    _DEFAULTS = {
        "numeric":     "median",
        "categorical": "unknown",
        "boolean":     "mode",
        "datetime":    "ffill",
    }

    def run(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        df = df.copy()
        manual  = config.get("missing", {})        # column-level overrides
        ccfg    = config.get("columns", {})
        filled  = {}

        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue

            # ── resolve strategy ──────────────────────────────────────────────
            strategy = manual.get(col)             # user override first

            if not strategy:
                dtype_family = self._dtype_family(df[col], ccfg.get(col, {}))
                strategy     = self._DEFAULTS[dtype_family]

            # ── apply strategy ────────────────────────────────────────────────
            before = df[col].isna().sum()
            df[col] = self._impute(df[col], strategy)
            after  = df[col].isna().sum()

            filled[col] = {"strategy": strategy, "filled": int(before - after)}
            self.log.info(f"[{col}] {strategy} — filled {before - after} / {before} nulls")

        # ── safety net: drop any remaining null rows (edge cases) ─────────────
        remaining = df.isna().any(axis=1).sum()
        if remaining:
            self.log.warning(f"{remaining} rows still contain NaN — dropping them.")
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

        # Re-normalise: filling with string values can re-trigger ArrowStringArray
        df = normalize_str_dtypes(df)
        self.report["filled_columns"] = filled
        self.log.info(f"Missing-value pass complete. Total NaN remaining: {df.isna().sum().sum()}")
        return df

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _dtype_family(series: pd.Series, col_meta: dict) -> str:
        inferred = col_meta.get("inferred_type", "")
        if "bool" in inferred:
            return "boolean"
        if "datetime" in inferred:
            return "datetime"
        if "numeric" in inferred or series.dtype in (
            "int64","int32","float64","float32","Int64","Float64"
        ):
            return "numeric"
        return "categorical"

    @staticmethod
    def _impute(series: pd.Series, strategy: str) -> pd.Series:
        s = strategy.lower().strip()

        if s == "median":
            num = pd.to_numeric(series, errors="coerce")
            val = num.median()
            return series.fillna(val if pd.notna(val) else 0)

        elif s == "mean":
            num = pd.to_numeric(series, errors="coerce")
            val = num.mean()
            return series.fillna(val if pd.notna(val) else 0)

        elif s == "mode":
            mode_vals = series.mode(dropna=True)
            val = mode_vals.iloc[0] if len(mode_vals) > 0 else (
                False if series.dtype == bool else "Unknown"
            )
            return series.fillna(val)

        elif s in ("unknown", "constant"):
            return series.fillna("Unknown")

        elif s == "zero":
            return series.fillna(0)

        elif s == "false":
            return series.fillna(False)

        elif s == "ffill":
            filled = series.ffill()
            # fallback: if still NaN at head
            return filled.bfill().fillna("Unknown")

        elif s == "bfill":
            filled = series.bfill()
            return filled.ffill().fillna("Unknown")

        elif s == "drop_row":
            return series   # rows will be dropped later

        elif s.startswith("const:"):
            val = s.split(":", 1)[1]
            return series.fillna(val)

        else:
            # fallback: median for numeric, unknown for others
            try:
                num = pd.to_numeric(series, errors="coerce")
                val = num.median()
                if pd.notna(val):
                    return series.fillna(val)
            except Exception:
                pass
            return series.fillna("Unknown")
