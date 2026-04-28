# quant-trading-backtest

A BTC trend-following backtest that combines a **Chandelier Exit** indicator,
a **Markov Regime Switching** filter, and a **Random Forest** classifier to
generate and evaluate ML-confirmed long signals.

---

## Project Structure

```
quant_backtest/
│
├── pipeline.py                         # End-to-end runner
│
├── data/
│   ├── download.py                     # Step 1 — Binance OHLC download
│   └── merge.py                        # Step 4 — Merge per-feature CSVs
│
├── strategy/
│   ├── chandelier_exit.py              # Step 2 — CE signals + stop distances
│   └── optimise_parameters.py          # Optional — grid search CE params
│
├── ml/
│   ├── markov_regimes.py               # Step 3 — 2-regime MRS model
│   ├── feature_engineering.py          # Step 5 — EMA features + target
│   └── train_random_forest.py          # Step 6 — RF classifier
│
└── backtest/
    ├── metrics.py                      # Step 7a — Raw CE trade metrics
    └── signal_filter.py                # Step 7b — ML-filtered backtest
```

---

## Pipeline

```
Binance API
    │
    ▼
data/download.py          →  OHLC_daily_from2020.csv
    │
    ▼
strategy/chandelier_exit.py  →  CE_calc.csv
    │
    ▼
ml/markov_regimes.py      →  MRS_2017_2026_regimes_1w.csv
    │
    ▼
data/merge.py             →  final_dataset.csv
    │
    ▼
ml/feature_engineering.py →  final_dataset+target+ema.csv
    │
    ▼
ml/train_random_forest.py →  final_dataset_with_RF.csv  +  rf_30day_model.pkl
    │
    ▼
backtest/signal_filter.py →  metrics printed to stdout
```

---

## Quick Start

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels joblib python-binance

# Set your API keys in data/download.py and ml/markov_regimes.py, then:
python -m quant_backtest.pipeline
```

To run any single step independently:

```bash
python -m quant_backtest.data.download
python -m quant_backtest.strategy.chandelier_exit
python -m quant_backtest.ml.markov_regimes
python -m quant_backtest.ml.feature_engineering
python -m quant_backtest.ml.train_random_forest
python -m quant_backtest.backtest.metrics
python -m quant_backtest.backtest.signal_filter
```

---

## Key Parameters

| Parameter | Default | Location |
|---|---|---|
| CE period | 10 | `strategy/chandelier_exit.py` |
| CE multiplier | 3.5 | `strategy/chandelier_exit.py` |
| MRS regimes | 2 | `ml/markov_regimes.py` |
| RF trees | 300 | `ml/train_random_forest.py` |
| RF max depth | 3 | `ml/train_random_forest.py` |
| ML threshold | 0.51 | `backtest/signal_filter.py` |
| Target horizon | 30 bars | `ml/feature_engineering.py` |

---

## Outputs

| File | Description |
|---|---|
| `CE_calc.csv` | OHLC + CE stops + signals |
| `MRS_2017_2026_regimes_1w.csv` | Weekly data + regime labels |
| `final_dataset.csv` | Merged feature matrix |
| `final_dataset+target+ema.csv` | Training-ready dataset |
| `final_dataset_with_RF.csv` | Test set + `ml_prob` column |
| `rf_30day_model.pkl` | Serialised RF model |
| `trades_ohnefilter_longonly.csv` | Per-trade return log (no filter) |
| `*.png` | Diagnostic and performance charts |
