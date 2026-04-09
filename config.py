"""
config.py  –  Manual config system.
Provides a typed Config class + JSON loader + sensible defaults.

Usage (manual mode):
    from config import Config
    cfg = Config.from_json("my_config.json")
    # or
    cfg = Config(mode="manual", missing={"age":"median"}, encoding={"city":"onehot"})
    config_dict = cfg.to_dict()
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:
    # ── pipeline-level settings ────────────────────────────────────────────────
    mode: str = "auto"                           # "auto" | "manual"
    active_agents: list = field(default_factory=lambda: [
        "cleaning", "type_conversion", "missing",
        "encoding", "feature", "validation"
    ])

    # ── global thresholds ──────────────────────────────────────────────────────
    drop_duplicates:        bool  = True
    null_drop_threshold:    float = 0.60
    variance_threshold:     float = 1e-10
    low_card_threshold:     int   = 10
    med_card_threshold:     int   = 50
    normalize:              bool  = False
    scale_method:           str   = "minmax"     # "minmax" | "zscore"
    log_transform_skewed:   bool  = False
    datetime_include_time:  bool  = False
    normalize_text:         bool  = True

    # ── per-column overrides (col_name → strategy string) ─────────────────────
    cleaning:        dict = field(default_factory=dict)   # {drop_columns:[...], ...}
    missing:         dict = field(default_factory=dict)   # {col: "median"|"mean"|...}
    encoding:        dict = field(default_factory=dict)   # {col: "onehot"|"label"|...}
    type_conversion: dict = field(default_factory=dict)   # {col: "int"|"float"|...}
    features:        dict = field(default_factory=dict)   # {derived: [...rules]}

    # ── output ─────────────────────────────────────────────────────────────────
    output_path: str = "output/processed.csv"

    # ── class methods ──────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        # pull known fields; ignore unknown keys gracefully
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert to nested dict for pipeline consumption."""
        return {
            "mode":           self.mode,
            "active_agents":  self.active_agents,
            "global": {
                "drop_duplicates":       self.drop_duplicates,
                "null_drop_threshold":   self.null_drop_threshold,
                "variance_threshold":    self.variance_threshold,
                "low_card_threshold":    self.low_card_threshold,
                "med_card_threshold":    self.med_card_threshold,
                "normalize":             self.normalize,
                "scale_method":          self.scale_method,
                "log_transform_skewed":  self.log_transform_skewed,
                "datetime_include_time": self.datetime_include_time,
                "normalize_text":        self.normalize_text,
            },
            "cleaning":        self.cleaning,
            "missing":         self.missing,
            "encoding":        self.encoding,
            "type_conversion": self.type_conversion,
            "features":        self.features,
            "output_path":     self.output_path,
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ── convenience builder ────────────────────────────────────────────────────────

def build_auto_config() -> dict:
    """Return default AUTO mode config dict."""
    return Config().to_dict()


def build_manual_config(
    drop_columns: list[str] = None,
    missing_strategies: dict[str, str] = None,
    encoding_strategies: dict[str, str] = None,
    derived_features: list[dict] = None,
    **kwargs: Any,
) -> dict:
    """Convenience builder for manual mode configs."""
    cfg = Config(mode="manual", **kwargs)
    if drop_columns:
        cfg.cleaning["drop_columns"] = drop_columns
    if missing_strategies:
        cfg.missing = missing_strategies
    if encoding_strategies:
        cfg.encoding = encoding_strategies
    if derived_features:
        cfg.features["derived"] = derived_features
    return cfg.to_dict()
