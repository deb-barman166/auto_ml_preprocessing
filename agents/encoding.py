"""
agents/encoding.py  –  EncodingAgent
Handles ALL categorical → numeric transformations:
  A. One-Hot Encoding   (low cardinality)
  B. Label Encoding     (medium cardinality)
  C. Frequency Encoding (high cardinality)
  D. Boolean → 0/1
  E. Multi-Hot Encoding (comma-separated multi-label)
Guarantees: NO object / category / bool columns remain.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from agents.base import BaseAgent
from utils.helpers import cardinality, is_multilabel, is_bool_like, bool_series_to_int


class EncodingAgent(BaseAgent):

    def run(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        df    = df.copy()
        ccfg  = config.get("columns", {})
        gcfg  = config.get("global", {})
        man_enc = config.get("encoding", {})     # user overrides

        low_t = gcfg.get("low_card_threshold", 10)
        med_t = gcfg.get("med_card_threshold", 50)

        enc_log = {}

        # iterate over columns that are still non-numeric
        for col in list(df.columns):
            if col not in df.columns:
                continue
            if not self._needs_encoding(df[col]):
                continue

            meta     = ccfg.get(col, {})
            inferred = meta.get("inferred_type", "")
            enc_strat = man_enc.get(col) or meta.get("encoding", "auto")

            # ── resolve final strategy ─────────────────────────────────────────
            if enc_strat == "skip":
                self.log.warning(f"[{col}] encoding=skip — dropping column.")
                df.drop(columns=[col], inplace=True)
                continue

            if enc_strat == "auto":
                enc_strat = self._auto_strategy(df[col], low_t, med_t, inferred)

            # ── apply encoding ─────────────────────────────────────────────────
            if enc_strat == "bool_to_int":
                df[col] = bool_series_to_int(df[col])
                enc_log[col] = "bool_to_int"

            elif enc_strat == "onehot":
                df = self._onehot(df, col)
                enc_log[col] = f"onehot"

            elif enc_strat == "label":
                df[col] = self._label(df[col])
                enc_log[col] = "label"

            elif enc_strat == "frequency":
                df[col] = self._frequency(df[col])
                enc_log[col] = "frequency"

            elif enc_strat == "multihot":
                df = self._multihot(df, col)
                enc_log[col] = "multihot"

            else:
                # fallback
                df[col] = self._label(df[col])
                enc_log[col] = "label(fallback)"

        # ── final sweep: any remaining object/bool/category columns ───────────
        residual = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        for col in residual:
            self.log.warning(f"[{col}] still non-numeric after encoding — applying label fallback.")
            df[col] = self._label(df[col].astype(str))
            enc_log[col] = "label(final_fallback)"

        self.report["encoding_log"] = enc_log
        self.log.info(f"Encoding done. {len(enc_log)} columns encoded.")
        return df

    # ── strategy helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _needs_encoding(series: pd.Series) -> bool:
        dtype_str = str(series.dtype).lower()
        return (series.dtype in ("object", "category", "bool") or
                dtype_str in ("str", "string") or
                dtype_str.startswith("string"))

    @staticmethod
    def _auto_strategy(series: pd.Series, low_t: int, med_t: int, inferred: str) -> str:
        if "bool" in inferred or is_bool_like(series):
            return "bool_to_int"
        if "multilabel" in inferred or is_multilabel(series):
            return "multihot"
        n = cardinality(series)
        if n <= low_t:
            return "onehot"
        elif n <= med_t:
            return "label"
        else:
            return "frequency"

    # ── encoding implementations ───────────────────────────────────────────────

    def _onehot(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        dummies = pd.get_dummies(
            df[col].astype(str),
            prefix=col,
            prefix_sep="_",
            drop_first=False,
            dtype=np.uint8,
        )
        df.drop(columns=[col], inplace=True)
        df = pd.concat([df, dummies], axis=1)
        self.log.info(f"[{col}] one-hot → {dummies.shape[1]} new columns")
        return df

    @staticmethod
    def _label(series: pd.Series) -> pd.Series:
        categories = list(series.astype(str).unique())
        categories = sorted(categories)
        mapping = {v: i for i, v in enumerate(categories)}
        return series.astype(str).map(mapping).astype("int64")

    @staticmethod
    def _frequency(series: pd.Series) -> pd.Series:
        freq_map = series.astype(str).value_counts(normalize=True)
        return series.astype(str).map(freq_map).astype("float64")

    def _multihot(self, df: pd.DataFrame, col: str, sep: str = ",") -> pd.DataFrame:
        # collect all unique labels
        all_labels: set = set()
        for val in df[col].dropna().astype(str):
            for token in val.split(sep):
                t = token.strip()
                if t:
                    all_labels.add(t)

        all_labels = sorted(all_labels)
        self.log.info(f"[{col}] multi-hot → {len(all_labels)} binary columns")

        for label in all_labels:
            safe_label = label.replace(" ", "_").replace("/", "_")
            new_col = f"{col}_{safe_label}"
            df[new_col] = df[col].astype(str).apply(
                lambda x, lb=label: 1 if lb in [t.strip() for t in x.split(sep)] else 0
            ).astype(np.uint8)

        df.drop(columns=[col], inplace=True)
        return df
