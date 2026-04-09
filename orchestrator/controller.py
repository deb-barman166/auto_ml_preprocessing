"""
orchestrator/controller.py  –  Central Pipeline Controller
Orchestrates all agents in sequence or hybrid-safe mode.
Maintains a full audit trail / summary.
"""
from __future__ import annotations
import time
import pandas as pd
from typing import Optional

from agents.cleaning       import CleaningAgent
from agents.missing        import MissingValueAgent
from agents.type_conversion import TypeConversionAgent
from agents.encoding       import EncodingAgent
from agents.feature        import FeatureTransformationAgent
from agents.validation     import ValidationAgent
from utils.logger          import get_logger


class PipelineController:
    """
    Runs agents in the following default order:
      1. CleaningAgent
      2. TypeConversionAgent
      3. MissingValueAgent
      4. EncodingAgent
      5. FeatureTransformationAgent
      6. ValidationAgent

    'active_agents' in config can restrict which agents run.
    'mode' can be 'sequential' (default) or 'safe' (catches per-agent errors).
    """

    AGENT_ORDER = [
        "cleaning",
        "type_conversion",
        "missing",
        "encoding",
        "feature",
        "validation",
    ]

    _AGENT_MAP = {
        "cleaning":        CleaningAgent,
        "type_conversion": TypeConversionAgent,
        "missing":         MissingValueAgent,
        "encoding":        EncodingAgent,
        "feature":         FeatureTransformationAgent,
        "validation":      ValidationAgent,
    }

    def __init__(self, config: dict):
        self.config = config
        self.log    = get_logger("Controller")
        self._active_agents: list[str] = config.get(
            "active_agents", self.AGENT_ORDER
        )
        self._mode: str = config.get("mode", "sequential")
        self.summary: dict = {}

    # ── public API ─────────────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        original_shape = df.shape
        self.log.info("=" * 60)
        self.log.info("  AutoML Preprocessing Engine  —  Pipeline Start")
        self.log.info(f"  Input shape : {original_shape}")
        self.log.info(f"  Mode        : {self._mode}")
        self.log.info(f"  Agents      : {self._active_agents}")
        self.log.info("=" * 60)

        t_start = time.perf_counter()
        agent_reports = {}

        for name in self.AGENT_ORDER:
            if name not in self._active_agents:
                self.log.info(f"[{name}] — skipped (not in active_agents)")
                continue

            agent_cls = self._AGENT_MAP[name]
            agent     = agent_cls()

            if self._mode == "safe":
                try:
                    df = agent.execute(df, self.config)
                except Exception as exc:
                    self.log.error(f"[{name}] crashed in safe mode: {exc}. Continuing…")
            else:
                df = agent.execute(df, self.config)

            agent_reports[name] = agent.report

        total_s = round(time.perf_counter() - t_start, 3)

        # ── build summary ──────────────────────────────────────────────────────
        self.summary = self._build_summary(
            original_shape, df.shape, agent_reports, total_s
        )
        self._print_summary()
        return df

    # ── private helpers ────────────────────────────────────────────────────────

    def _build_summary(
        self,
        orig_shape: tuple,
        final_shape: tuple,
        reports: dict,
        elapsed: float,
    ) -> dict:
        clean_rep  = reports.get("cleaning", {})
        miss_rep   = reports.get("missing",  {})
        enc_rep    = reports.get("encoding", {})
        feat_rep   = reports.get("feature",  {})
        val_rep    = reports.get("validation", {})

        return {
            "original_shape":        orig_shape,
            "final_shape":           final_shape,
            "rows_removed":          orig_shape[0] - final_shape[0],
            "cols_removed":          orig_shape[1] - (len(clean_rep.get("dropped_cols", [])) - 0),
            "dropped_cols":          clean_rep.get("dropped_cols", []),
            "duplicate_rows_removed":clean_rep.get("dropped_rows", 0),
            "columns_imputed":       list(miss_rep.get("filled_columns", {}).keys()),
            "encoding_log":          enc_rep.get("encoding_log", {}),
            "feature_transforms":    feat_rep.get("feature_transforms", []),
            "validation_errors":     val_rep.get("validation_errors", []),
            "validation_warnings":   val_rep.get("validation_warnings", []),
            "memory_mb":             val_rep.get("memory_mb", 0),
            "total_elapsed_s":       elapsed,
        }

    def _print_summary(self):
        s = self.summary
        self.log.info("=" * 60)
        self.log.info("  PIPELINE COMPLETE — SUMMARY")
        self.log.info("=" * 60)
        self.log.info(f"  Original shape   : {s['original_shape']}")
        self.log.info(f"  Final shape      : {s['final_shape']}")
        self.log.info(f"  Duplicate rows   : {s['duplicate_rows_removed']}")
        self.log.info(f"  Dropped columns  : {s['dropped_cols']}")
        self.log.info(f"  Imputed columns  : {s['columns_imputed']}")
        self.log.info(f"  Encodings applied: {len(s['encoding_log'])}")
        self.log.info(f"  Derived features : {s['feature_transforms']}")
        self.log.info(f"  Validation errors: {s['validation_errors']}")
        self.log.info(f"  Memory           : {s['memory_mb']} MB")
        self.log.info(f"  Total time       : {s['total_elapsed_s']}s")
        self.log.info("=" * 60)
