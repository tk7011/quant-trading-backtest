"""
pipeline.py
-----------
End-to-end pipeline runner for the BTC Chandelier Exit + Markov Regime +
Random Forest backtest system.

Steps
-----
1.  Download OHLC data from Binance
2.  Compute Chandelier Exit signals + stop-distance features
3.  Fit Markov Regime Switching model and append regime labels
4.  Merge all feature CSVs into one dataset
5.  Add EMA features and binary target label
6.  Train Random Forest classifier
7.  Apply ML-filtered signal backtest and report metrics

Each step can be run independently by executing the corresponding module
directly (see each file's ``if __name__ == "__main__":`` block).

Usage
-----
    python pipeline.py
"""

from quant_backtest.data.download import download_data
from quant_backtest.data.merge import merge_feature_csvs
from quant_backtest.strategy.chandelier_exit import chandelier_exit
from quant_backtest.ml.markov_regimes import (
    download_data as download_data_weekly,
    test_stationarity,
    find_best_ar_order,
    fit_mrs_model,
)
from quant_backtest.ml.feature_engineering import add_ema_features, add_target
from quant_backtest.ml.train_random_forest import train_and_evaluate
from quant_backtest.backtest.metrics import build_trade_log, compute_metrics
from quant_backtest.backtest.signal_filter import apply_ml_filter, evaluate_filtered_strategy

import pandas as pd

# ---------------------------------------------------------------------------
# Step 1 — Download daily OHLC
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1 — Downloading daily OHLC data")
print("=" * 60)
ohlc = download_data(
    symbol="BTCUSDT",
    interval="1d",
    start_date="2017-09-09",
    end_date="2026-02-18",
    return_type="pct",
)

# ---------------------------------------------------------------------------
# Step 2 — Chandelier Exit
# ---------------------------------------------------------------------------
print("\nStep 2 — Computing Chandelier Exit signals")
ce_df = chandelier_exit(ohlc, period=10, multiplier=3.5, use_close=True)

# ---------------------------------------------------------------------------
# Step 3 — Markov Regime Switching
# ---------------------------------------------------------------------------
print("\nStep 3 — Fitting Markov Regime Switching model")
weekly = download_data_weekly("BTCUSDT", "1w", "2017-09-09", "2026-02-17", return_type="pct")
test_stationarity(weekly)
best_ar = find_best_ar_order(weekly, max_order=5)
mrs_results = fit_mrs_model(weekly, ar_order=best_ar)

regime_probs = mrs_results.smoothed_marginal_probabilities
weekly["Regime_2"]    = regime_probs.idxmax(axis=1)
weekly["Prob_Regime0"] = regime_probs[0]
weekly["Prob_Regime1"] = regime_probs[1]
weekly.to_csv("MRS_2017_2026_regimes_1w.csv")

# ---------------------------------------------------------------------------
# Step 4 — Merge feature CSVs
# ---------------------------------------------------------------------------
print("\nStep 4 — Merging feature CSVs")
merged = merge_feature_csvs("folder_for_glob/*.csv")

# ---------------------------------------------------------------------------
# Step 5 — Feature engineering
# ---------------------------------------------------------------------------
print("\nStep 5 — Engineering EMA features and target")
merged = add_ema_features(merged)
merged.to_csv("final_dataset+ema.csv", index=False)

merged = add_target(merged, horizon=30)
merged.to_csv("final_dataset+target+ema.csv", index=False)

# ---------------------------------------------------------------------------
# Step 6 — Train Random Forest
# ---------------------------------------------------------------------------
print("\nStep 6 — Training Random Forest")
model = train_and_evaluate(merged)

# ---------------------------------------------------------------------------
# Step 7 — Signal filter backtest
# ---------------------------------------------------------------------------
print("\nStep 7 — ML-filtered signal backtest")
rf_df = pd.read_csv("final_dataset_with_RF.csv")
rf_df = apply_ml_filter(rf_df)
metrics = evaluate_filtered_strategy(rf_df)

print("\n===== Final Filtered Strategy Metrics =====")
for key, val in metrics.items():
    print(f"  {key:<15} {val}")
