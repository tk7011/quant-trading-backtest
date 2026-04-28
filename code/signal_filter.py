"""
backtest/signal_filter.py
--------------------------
Applies a Random Forest probability threshold to the raw Chandelier Exit
signals, generating filtered long-only positions, and then evaluates the
resulting strategy.

A confirmed buy occurs when ``Signal == 1`` AND ``ml_prob > THRESHOLD``.
The position stays open until the next ``Signal == -1``.

Pipeline position
-----------------
Runs *after* ``ml/train_random_forest.py`` produces
``final_dataset_with_RF.csv``.

Inputs
------
final_dataset_with_RF.csv

Outputs
-------
new_csv_expecting_a_bug.csv    — dataset with ``filtered_signals`` column
Prints strategy metrics to stdout.
"""

import bisect

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ML_THRESHOLD = 0.51


# ---------------------------------------------------------------------------
# Signal filtering
# ---------------------------------------------------------------------------
def apply_ml_filter(df: pd.DataFrame, threshold: float = ML_THRESHOLD) -> pd.DataFrame:
    """Filter CE buy signals using the Random Forest probability.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with columns ``["Signal", "Returns", "ml_prob"]``.
    threshold : float
        Minimum ``ml_prob`` value required to confirm a buy signal.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with a new ``filtered_signals`` column (``0`` or ``1``).
    """
    df = df.copy()
    df["filtered_signals"] = 0

    confirmed_buys = df.index[(df["Signal"] == 1) & (df["ml_prob"] > threshold)].tolist()
    sells = df.index[df["Signal"] == -1].tolist()

    for buy_idx in confirmed_buys:
        # Binary search for the next sell after this buy
        pos = bisect.bisect_right(sells, buy_idx)
        sell_idx = sells[pos] if pos < len(sells) else df.index[-1]
        df.loc[buy_idx:sell_idx, "filtered_signals"] = 1

    return df


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------
def evaluate_filtered_strategy(df: pd.DataFrame) -> dict:
    """Compute Sharpe ratio, max drawdown, total return, and trade count.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``filtered_signals`` and ``Returns`` columns.

    Returns
    -------
    dict
        Dictionary with keys:
        ``sharpe, max_drawdown, total_return, n_trades``.
    """
    df = df.copy()
    df["strategy_return"] = df["filtered_signals"] * (df["Returns"] / 100.0)

    std = df["strategy_return"].std()
    sharpe = df["strategy_return"].mean() / std * np.sqrt(252) if std != 0 else 0.0

    equity = (1 + df["strategy_return"]).cumprod()
    drawdown = (equity - equity.cummax()) / equity.cummax()

    trade_entries = (df["filtered_signals"] == 1) & (
        df["filtered_signals"].shift(1).fillna(0) == 0
    )

    return {
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "total_return": (equity.iloc[-1] - 1) * 100,
        "n_trades": int(trade_entries.sum()),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("final_dataset_with_RF.csv")

    df = apply_ml_filter(df, threshold=ML_THRESHOLD)
    m = evaluate_filtered_strategy(df)

    df.to_csv("new_csv_expecting_a_bug.csv", index=False)

    print(f"Threshold    : {ML_THRESHOLD}")
    print(f"Sharpe       : {m['sharpe']:.4f}")
    print(f"Max Drawdown : {m['max_drawdown']:.4f}")
    print(f"Total Return : {m['total_return']:.2f} %")
    print(f"Trades       : {m['n_trades']}")
