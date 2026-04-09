"""
main.py  –  CLI entry point for the AutoML Preprocessing Engine.

Usage examples
--------------
  # Auto mode (all defaults):
  python main.py --source data/titanic.csv

  # Auto mode + save log:
  python main.py --source https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv --log

  # Manual config from JSON:
  python main.py --source data/raw.csv --config my_config.json

  # Scale output + log transform:
  python main.py --source data/raw.csv --normalize --log-skew

  # Restrict agents:
  python main.py --source data/raw.csv --agents cleaning missing encoding
"""
from __future__ import annotations
import argparse
import sys
import os

# ── ensure project root is on sys.path ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from core.loader       import load_data
from core.auto_config  import generate_auto_config
from config            import Config, build_auto_config
from orchestrator.controller import PipelineController
from utils.logger      import get_logger, get_file_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="automl_preprocess",
        description="Production-grade, rule-based Data Preprocessing Engine",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--source",    required=True,  help="Path or URL to input dataset")
    p.add_argument("--config",    default=None,   help="Path to manual JSON config file")
    p.add_argument("--output",    default="output/processed.csv",
                                                  help="Output CSV path")
    p.add_argument("--agents",    nargs="*",       help="Override active agents list")
    p.add_argument("--mode",      default="auto",  choices=["auto","manual","safe"],
                                                  help="Pipeline mode")
    p.add_argument("--normalize", action="store_true", help="Enable min-max normalisation")
    p.add_argument("--zscore",    action="store_true", help="Enable z-score standardisation")
    p.add_argument("--log-skew",  action="store_true", help="Log-transform skewed columns")
    p.add_argument("--no-text-lower", action="store_true",
                                                  help="Disable text lowercasing")
    p.add_argument("--log",       action="store_true", help="Save preprocessing.log file")
    p.add_argument("--sep",       default=None,   help="CSV separator (default auto-detect)")
    return p.parse_args()


def main():
    args   = parse_args()
    logger = get_file_logger("Main", "preprocessing.log") if args.log else get_logger("Main")

    # ── 1. load raw data ───────────────────────────────────────────────────────
    logger.info(f"Source: {args.source}")
    df_raw = load_data(args.source, sep=args.sep)

    # ── 2. build config ────────────────────────────────────────────────────────
    if args.config:
        logger.info(f"Loading manual config from: {args.config}")
        cfg_obj = Config.from_json(args.config)
        config  = cfg_obj.to_dict()
    else:
        config = build_auto_config()

    # apply CLI overrides
    if args.agents:
        config["active_agents"] = args.agents
    if args.mode:
        config["mode"] = args.mode

    if args.normalize or args.zscore:
        config["global"]["normalize"]    = True
        config["global"]["scale_method"] = "zscore" if args.zscore else "minmax"
    if args.log_skew:
        config["global"]["log_transform_skewed"] = True
    if args.no_text_lower:
        config["global"]["normalize_text"] = False

    config["output_path"] = args.output

    # ── 3. auto-profile columns ────────────────────────────────────────────────
    config = generate_auto_config(df_raw, user_overrides={})
    # re-merge CLI-level globals (auto_config replaces global section)
    config["global"].update({
        "normalize":            args.normalize or args.zscore,
        "scale_method":         "zscore" if args.zscore else "minmax",
        "log_transform_skewed": args.log_skew,
        "normalize_text":       not args.no_text_lower,
    })
    if args.agents:
        config["active_agents"] = args.agents
    config["mode"]        = args.mode
    config["output_path"] = args.output

    # ── 4. run pipeline ────────────────────────────────────────────────────────
    controller = PipelineController(config)
    df_out     = controller.run(df_raw)

    # ── 5. save output ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_out.to_csv(args.output, index=False)
    logger.info(f"Output saved → {args.output}  (shape={df_out.shape})")

    # ── 6. print final summary dict ────────────────────────────────────────────
    s = controller.summary
    print("\n" + "=" * 55)
    print("  PREPROCESSING COMPLETE")
    print("=" * 55)
    print(f"  Input  : {s.get('original_shape')}")
    print(f"  Output : {s.get('final_shape')}")
    print(f"  Time   : {s.get('total_elapsed_s')}s")
    print(f"  Output saved to: {args.output}")
    print("=" * 55)

    return df_out


# ── programmatic API ───────────────────────────────────────────────────────────

def preprocess(
    source,
    config: dict = None,
    output_path: str = "output/processed.csv",
    save: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Programmatic API — import and call directly from your notebook or code.

    Parameters
    ----------
    source      : str (path/URL) or pd.DataFrame
    config      : optional config dict (build_auto_config() / build_manual_config())
    output_path : where to save the processed CSV
    save        : whether to write output CSV

    Returns
    -------
    (processed_df, summary_dict)
    """
    log = get_logger("PreprocessAPI")

    df_raw = load_data(source)
    log.info(f"Raw data loaded — shape: {df_raw.shape}")

    # generate column profiles
    auto_cfg = generate_auto_config(df_raw)

    # merge user config on top
    if config:
        auto_cfg["global"].update(config.get("global", {}))
        auto_cfg.update({k: v for k, v in config.items() if k not in ("columns", "global")})

    auto_cfg["output_path"] = output_path

    controller = PipelineController(auto_cfg)
    df_out     = controller.run(df_raw)

    if save:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df_out.to_csv(output_path, index=False)
        log.info(f"Saved → {output_path}")

    return df_out, controller.summary


if __name__ == "__main__":
    main()
