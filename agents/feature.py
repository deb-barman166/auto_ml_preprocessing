"""
agents/feature.py  –  FeatureTransformationAgent
Handles:
  - Normalization (min-max) / Standardization (z-score) — optional
  - Log transform for skewed numeric columns — optional
  - User-defined derived features (ratio / difference / aggregation)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from agents.base import BaseAgent


class FeatureTransformationAgent(BaseAgent):

    def run(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        df   = df.copy()
        gcfg = config.get("global", {})
        fcfg = config.get("features", {})    # derived feature rules
        applied = []

        # ── 1. log transform skewed columns ───────────────────────────────────
        if gcfg.get("log_transform_skewed", False):
            df = self._log_transform(df, applied)

        # ── 2. normalise (min-max) or standardise (z-score) ───────────────────
        scale_mode = gcfg.get("normalize", False)
        if scale_mode:
            method = gcfg.get("scale_method", "minmax")
            df     = self._scale(df, method, applied)

        # ── 3. user-defined derived features ──────────────────────────────────
        for rule in fcfg.get("derived", []):
            df = self._apply_rule(df, rule, applied)

        self.report["feature_transforms"] = applied
        self.log.info(f"Feature transforms applied: {len(applied)}")
        return df

    # ── helpers ────────────────────────────────────────────────────────────────

    def _log_transform(self, df: pd.DataFrame, log_list: list) -> pd.DataFrame:
        """Apply log1p to numeric columns with skewness > 1.0."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            s = df[col].dropna()
            # only positive-valued cols
            if (s > 0).all():
                try:
                    skew_val = float(s.skew())
                except Exception:
                    continue
                if abs(skew_val) > 1.0:
                    df[col] = np.log1p(df[col])
                    log_list.append(f"log1p({col})")
                    self.log.info(f"log1p applied to [{col}] (skew={skew_val:.2f})")
        return df

    def _scale(self, df: pd.DataFrame, method: str, log_list: list) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            s = df[col]
            if method == "minmax":
                mn, mx = s.min(), s.max()
                rng = mx - mn
                if rng != 0:
                    df[col] = (s - mn) / rng
                    log_list.append(f"minmax({col})")
            elif method in ("zscore", "standard"):
                mu, sigma = s.mean(), s.std()
                if sigma and sigma != 0:
                    df[col] = (s - mu) / sigma
                    log_list.append(f"zscore({col})")
        self.log.info(f"Scaling ({method}) applied to {len(num_cols)} numeric columns.")
        return df

    def _apply_rule(self, df: pd.DataFrame, rule: dict, log_list: list) -> pd.DataFrame:
        """
        Supported rule formats:
          {"type": "ratio",      "col_a": "A", "col_b": "B", "name": "A_div_B"}
          {"type": "diff",       "col_a": "A", "col_b": "B", "name": "A_minus_B"}
          {"type": "product",    "col_a": "A", "col_b": "B", "name": "A_mul_B"}
          {"type": "agg_mean",   "cols":  ["A","B","C"],       "name": "mean_ABC"}
          {"type": "agg_sum",    "cols":  ["A","B","C"],       "name": "sum_ABC"}
          {"type": "log1p",      "col":   "A"}
          {"type": "square",     "col":   "A"}
          {"type": "sqrt",       "col":   "A"}
        """
        t = rule.get("type", "")
        try:
            if t == "ratio":
                a, b, name = rule["col_a"], rule["col_b"], rule.get("name", f"{rule['col_a']}_div_{rule['col_b']}")
                df[name] = df[a] / df[b].replace(0, np.nan)
                df[name] = df[name].fillna(0)

            elif t == "diff":
                a, b, name = rule["col_a"], rule["col_b"], rule.get("name", f"{rule['col_a']}_minus_{rule['col_b']}")
                df[name] = df[a] - df[b]

            elif t == "product":
                a, b, name = rule["col_a"], rule["col_b"], rule.get("name", f"{rule['col_a']}_mul_{rule['col_b']}")
                df[name] = df[a] * df[b]

            elif t == "agg_mean":
                cols, name = rule["cols"], rule["name"]
                df[name] = df[cols].mean(axis=1)

            elif t == "agg_sum":
                cols, name = rule["cols"], rule["name"]
                df[name] = df[cols].sum(axis=1)

            elif t == "log1p":
                col = rule["col"]
                df[col] = np.log1p(df[col])
                name = f"log1p({col})"

            elif t == "square":
                col = rule["col"]
                df[col + "_sq"] = df[col] ** 2
                name = col + "_sq"

            elif t == "sqrt":
                col = rule["col"]
                df[col + "_sqrt"] = np.sqrt(df[col].clip(lower=0))
                name = col + "_sqrt"
            else:
                self.log.warning(f"Unknown rule type: {t}")
                return df

            log_list.append(name)
            self.log.info(f"Derived feature created: {name}")

        except KeyError as e:
            self.log.error(f"Rule {rule} failed — missing column: {e}")
        except Exception as e:
            self.log.error(f"Rule {rule} failed: {e}")

        return df
