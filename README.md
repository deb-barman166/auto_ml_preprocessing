# AutoML Preprocessing Engine 🚀

> **Production-grade, 100% rule-based, fully deterministic data preprocessing and conversion engine.**
> Input: *any* raw dataset → Output: perfectly clean, fully numeric, ML/DL-ready DataFrame.

---

## Features

| Capability | Details |
|---|---|
| **Universal Detection** | int, float, bool, datetime, category, mixed-type, multi-label |
| **Cleaning** | Duplicate removal, whitespace trim, text normalisation, high-null drop |
| **Missing Values** | Median/mean/mode/constant/ffill/bfill per column — zero NaN guaranteed |
| **Type Conversion** | Numeric strings, boolean strings, datetime extraction |
| **Encoding** | One-Hot / Label / Frequency / Bool→int / Multi-Hot (auto + manual) |
| **Feature Engineering** | Log transform, scaling, ratio/diff/product/agg derived features |
| **Validation** | No NaN, no Inf, no non-numeric columns — full audit report |
| **Config System** | AUTO mode (zero config) + MANUAL mode (full per-column control) |
| **Performance** | Vectorised pandas/numpy, efficient memory usage, large dataset support |

---

## Project Structure

```
auto_ml_preprocessing/
├── main.py                  # CLI entry point + programmatic API
├── config.py                # Config dataclass + JSON loader
│
├── orchestrator/
│   └── controller.py        # Pipeline orchestrator
│
├── agents/
│   ├── base.py              # Abstract base class
│   ├── cleaning.py          # CleaningAgent
│   ├── missing.py           # MissingValueAgent
│   ├── type_conversion.py   # TypeConversionAgent
│   ├── encoding.py          # EncodingAgent
│   ├── feature.py           # FeatureTransformationAgent
│   └── validation.py        # ValidationAgent
│
├── core/
│   ├── loader.py            # Universal data loader
│   └── auto_config.py       # Auto column profiler
│
├── utils/
│   ├── logger.py            # Colourised structured logger
│   └── helpers.py           # Pure utility functions
│
├── output/
│   └── processed.csv        # Default output location
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
cd auto_ml_preprocessing
pip install -r requirements.txt
```

---

## Usage

### 1. CLI — Auto Mode (zero config)

```bash
# Local CSV
python main.py --source data/titanic.csv

# Remote URL
python main.py --source https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

# With scaling + log transform + file log
python main.py --source data/raw.csv --normalize --log-skew --log

# Z-score instead of min-max
python main.py --source data/raw.csv --zscore

# Custom output path
python main.py --source data/raw.csv --output results/clean.csv

# Run only specific agents
python main.py --source data/raw.csv --agents cleaning missing encoding
```

### 2. CLI — Manual Config (JSON)

```bash
python main.py --source data/raw.csv --config my_config.json --mode manual
```

**`my_config.json`** example:

```json
{
  "mode": "manual",
  "active_agents": ["cleaning", "missing", "encoding"],
  "cleaning": {
    "drop_columns": ["id", "row_num"],
    "normalize_text": true
  },
  "missing": {
    "age":    "median",
    "salary": "mean",
    "city":   "unknown",
    "active": "false"
  },
  "encoding": {
    "gender":  "label",
    "city":    "onehot",
    "country": "frequency",
    "tags":    "multihot"
  },
  "global": {
    "null_drop_threshold":   0.6,
    "low_card_threshold":    10,
    "med_card_threshold":    50,
    "normalize":             false,
    "log_transform_skewed":  false,
    "datetime_include_time": false
  }
}
```

---

### 3. Programmatic API (use in notebooks / code)

```python
from main import preprocess
from config import build_auto_config, build_manual_config

# ── AUTO MODE ──────────────────────────────────────────────────────────────────
df_clean, summary = preprocess("data/titanic.csv")
print(summary)

# ── MANUAL MODE ────────────────────────────────────────────────────────────────
config = build_manual_config(
    drop_columns       = ["PassengerId", "Name", "Ticket"],
    missing_strategies = {"Age": "median", "Embarked": "mode"},
    encoding_strategies= {"Sex": "label", "Embarked": "onehot"},
    derived_features   = [
        {"type": "ratio", "col_a": "Fare", "col_b": "Age", "name": "fare_per_age"}
    ],
)

df_clean, summary = preprocess("data/titanic.csv", config=config)

# access output
print(df_clean.head())
print(df_clean.dtypes)
print(f"Shape: {df_clean.shape}")
print(f"NaN total: {df_clean.isna().sum().sum()}")
```

---

## Encoding Strategy Reference

| Situation | Auto Strategy | Manual Value |
|---|---|---|
| ≤10 unique values | One-Hot Encoding | `"onehot"` |
| 11–50 unique values | Label Encoding | `"label"` |
| >50 unique values | Frequency Encoding | `"frequency"` |
| Boolean-like strings | Bool → 0/1 | `"bool_to_int"` |
| Comma-separated multi-values | Multi-Hot Encoding | `"multihot"` |
| Skip (drop column) | — | `"skip"` |

---

## Missing Value Strategy Reference

| Strategy | Behaviour |
|---|---|
| `median` | Fill with column median (numeric) |
| `mean` | Fill with column mean (numeric) |
| `mode` | Fill with most frequent value |
| `unknown` | Fill with string `"Unknown"` |
| `zero` | Fill with `0` |
| `false` | Fill with `0` (boolean columns) |
| `ffill` | Forward-fill, then back-fill |
| `bfill` | Back-fill, then forward-fill |
| `drop_row` | Drop rows where column is null |
| `const:VALUE` | Fill with literal `VALUE` |

---

## Derived Feature Rules

```python
# Ratio: A / B
{"type": "ratio",   "col_a": "revenue",  "col_b": "cost",   "name": "profit_margin"}

# Difference: A - B
{"type": "diff",    "col_a": "price",    "col_b": "discount","name": "net_price"}

# Product: A * B
{"type": "product", "col_a": "quantity", "col_b": "price",   "name": "total_value"}

# Row mean of multiple cols
{"type": "agg_mean","cols": ["score1","score2","score3"],     "name": "avg_score"}

# Row sum
{"type": "agg_sum", "cols": ["tax","fee","price"],            "name": "total_cost"}

# Log1p of a column
{"type": "log1p",   "col": "revenue"}

# Square
{"type": "square",  "col": "age"}

# Square root
{"type": "sqrt",    "col": "area"}
```

---

## Pipeline Flow

```
Raw Dataset (CSV / URL / DataFrame)
         │
         ▼
   ┌─────────────┐
   │   Loader    │  ← CSV, TSV, Excel, JSON, Parquet
   └──────┬──────┘
          │
          ▼
   ┌─────────────────┐
   │  Auto-Profiler  │  ← Column type inference, cardinality, null ratio
   └──────┬──────────┘
          │
          ▼
   ┌─────────────────┐
   │ CleaningAgent   │  ← Duplicates, nulls, whitespace, constant cols
   └──────┬──────────┘
          │
          ▼
   ┌──────────────────────┐
   │ TypeConversionAgent  │  ← numstr→float, boolstr→int, datetime→features
   └──────┬───────────────┘
          │
          ▼
   ┌──────────────────────┐
   │  MissingValueAgent   │  ← median/mean/mode/ffill per column
   └──────┬───────────────┘
          │
          ▼
   ┌──────────────────────┐
   │   EncodingAgent      │  ← OneHot / Label / Frequency / MultiHot
   └──────┬───────────────┘
          │
          ▼
   ┌──────────────────────────────┐
   │ FeatureTransformationAgent   │  ← Scaling, log, derived features
   └──────┬───────────────────────┘
          │
          ▼
   ┌──────────────────────┐
   │  ValidationAgent     │  ← Zero NaN, no Inf, all numeric
   └──────┬───────────────┘
          │
          ▼
   output/processed.csv  ← Fully numeric, ML-ready dataset
```

---

## Extending with a New Agent

```python
# agents/my_custom_agent.py
from agents.base import BaseAgent
import pandas as pd

class MyCustomAgent(BaseAgent):
    def run(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        self.log.info("Running my custom transform...")
        # ... your logic ...
        return df
```

Then register it in `orchestrator/controller.py`:

```python
from agents.my_custom_agent import MyCustomAgent

_AGENT_MAP = {
    ...
    "my_custom": MyCustomAgent,
}
AGENT_ORDER = [..., "my_custom"]
```

---

## Output Guarantee

Every processed dataset is guaranteed to be:

- ✅ Fully numeric (int / float only)
- ✅ Zero missing values
- ✅ Zero infinite values
- ✅ No object / category / bool columns
- ✅ Ready for sklearn, PyTorch, TensorFlow, XGBoost, etc.

---

*Built with Python · pandas · numpy · No ML/AI — 100% deterministic rule-based engine*
