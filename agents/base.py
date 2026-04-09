"""
agents/base.py  –  Abstract base class for all pipeline agents.
"""
from __future__ import annotations
import abc
import time
import pandas as pd
from utils.logger import get_logger


class BaseAgent(abc.ABC):
    """
    Every agent must implement `run(df, config)` and return a DataFrame.
    Execution time is automatically measured and logged.
    """

    def __init__(self):
        self.log    = get_logger(self.__class__.__name__)
        self.report: dict = {}   # collects per-agent transformation metadata

    # ── public interface ───────────────────────────────────────────────────────
    def execute(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Wrapper that times `run()` and catches errors gracefully."""
        self.log.info(f"Starting  —  input shape: {df.shape}")
        t0 = time.perf_counter()
        try:
            result = self.run(df, config)
        except Exception as exc:
            self.log.error(f"Agent failed: {exc}")
            raise
        elapsed = time.perf_counter() - t0
        self.log.info(f"Completed —  output shape: {result.shape}  [{elapsed:.3f}s]")
        self.report["elapsed_s"] = round(elapsed, 4)
        return result

    @abc.abstractmethod
    def run(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Override in each sub-class. Must return a DataFrame."""
        ...
