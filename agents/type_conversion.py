"""
agents/type_conversion.py  –  TypeConversionAgent
- Converts numeric strings → float
- Converts boolean strings → int (0/1)
- Parses datetime columns → extracts year/month/day/weekday[/hour/minute]
- Handles mixed-type columns
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from agents.base import BaseAgent
from utils.helpers import is_bool_like, is_numeric_string, bool_series_to_int


class TypeConversionAgent(BaseAgent):

    def run(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        df     = df.copy()
        ccfg   = config.get("columns", {})
        gcfg   = config.get("global", {})
        manual = config.get("type_conversion", {})     # col → forced dtype
        inc_time = gcfg.get("datetime_include_time", False)

        conversions = []

        for col in list(df.columns):
            meta     = ccfg.get(col, {})
            inferred = meta.get("inferred_type", "")
            forced   = manual.get(col)               # user override

            # ── user-forced dtype ──────────────────────────────────────────────
            if forced:
                df = self._force_dtype(df, col, forced)
                conversions.append((col, forced))
                continue

            # ── auto conversions ───────────────────────────────────────────────
            if inferred == "boolean" or (
                df[col].dtype == object and is_bool_like(df[col])
            ):
                df[col] = bool_series_to_int(df[col])
                conversions.append((col, "bool→int"))

            elif inferred == "datetime" or df[col].dtype in (
                "datetime64[ns]", "datetime64[us]"
            ):
                df = self._extract_datetime(df, col, inc_time)
                conversions.append((col, "datetime→features"))

            elif df[col].dtype == object and is_numeric_string(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                conversions.append((col, "numstr→float"))

            elif df[col].dtype == object:
                # try partial conversion (mixed type) — coerce non-numeric to NaN
                converted = pd.to_numeric(df[col], errors="coerce")
                # only apply if ≥ 80% successfully converted
                success_ratio = converted.notna().mean()
                if success_ratio >= 0.8:
                    df[col] = converted
                    conversions.append((col, f"mixed→numeric({success_ratio:.0%})"))
                # else leave for encoding

        self.report["conversions"] = conversions
        self.log.info(f"TypeConversion: {len(conversions)} conversions applied.")
        return df

    # ── helpers ────────────────────────────────────────────────────────────────

    def _extract_datetime(
        self, df: pd.DataFrame, col: str, inc_time: bool
    ) -> pd.DataFrame:
        try:
            parsed = pd.to_datetime(df[col], infer_format=True, errors="coerce")
        except Exception:
            parsed = pd.to_datetime(df[col], errors="coerce")

        prefix = col
        df[f"{prefix}_year"]    = parsed.dt.year.astype("Int64")
        df[f"{prefix}_month"]   = parsed.dt.month.astype("Int64")
        df[f"{prefix}_day"]     = parsed.dt.day.astype("Int64")
        df[f"{prefix}_weekday"] = parsed.dt.weekday.astype("Int64")   # 0=Mon

        if inc_time:
            df[f"{prefix}_hour"]   = parsed.dt.hour.astype("Int64")
            df[f"{prefix}_minute"] = parsed.dt.minute.astype("Int64")

        df.drop(columns=[col], inplace=True)
        self.log.info(f"Datetime [{col}] → {prefix}_year/month/day/weekday extracted.")
        return df

    def _force_dtype(self, df: pd.DataFrame, col: str, dtype: str) -> pd.DataFrame:
        dtype = dtype.lower()
        try:
            if dtype in ("int", "integer"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            elif dtype in ("str", "string", "object"):
                df[col] = df[col].astype(str)
            elif dtype == "bool":
                df[col] = bool_series_to_int(df[col])
            elif dtype == "datetime":
                df = self._extract_datetime(df, col, inc_time=False)
        except Exception as e:
            self.log.warning(f"Force-dtype [{col}→{dtype}] failed: {e}")
        return df
