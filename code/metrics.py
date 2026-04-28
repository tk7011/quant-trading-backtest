"""
backtest/metrics.py
-------------------
Computes per-trade and aggregate backtest metrics for the long-only Chandelier
Exit strategy (no ML filter applied).

Pipeline position
-----------------
Runs *after* ``strategy/chandelier_exit.py`` produces ``CE_calc.csv``.

Inputs
------
CE_calc.csv

Outputs
-------
trades_ohnefilter_longonly.csv   — per-trade return log
Prints aggregate metrics to stdout.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Trade log construction
# ---------------------------------------------------------------------------
def build_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a bar-level signal series into a per-trade return log.

    Only long trades are tracked (``Signal == 1`` opens, ``Signal == -1``
    closes). Returns accumulate bar by bar while a position is open.

    Parameters
    ----------
    df : pd.DataFrame
        Bar-level data with ``Date``, ``Signal``, and ``Returns`` columns.

    Returns
    -------
    pd.DataFrame
        One row per completed (or still-open) trade with columns
        ``["begin_date", "end_date", "total_return", "direction"]``.
    """
    trades = pd.DataFrame(columns=["begin_date", "end_date", "total_return", "direction"])
    trades.loc[0, "total_return"] = 0

    i = 0          # current trade index
    last = 0       # current position: 1 = long, 0 = flat

    for row in range(len(df)):
        signal = df.loc[row, "Signal"]

        # Accumulate returns while in a position
        if last != 0:
            trades.loc[i, "total_return"] += df.loc[row, "Returns"] * last

        if signal == 1:
            trades.loc[i, "direction"] = "long"
            trades.loc[i, "total_return"] = 0
            trades.loc[i, "begin_date"] = df.loc[row + 1, "Date"]
            last = 1

        elif signal == -1:
            if last != 0:
                trades.loc[i, "end_date"] = df.loc[row, "Date"]
                i += 1
            last = 0

    # Close any still-open trade at the last bar
    if last != 0:
        trades.loc[i, "end_date"] = df.loc[len(df) - 1, "Date"]

    return trades


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
def compute_metrics(trades: pd.DataFrame) -> dict:
    """Compute Sharpe ratio, win rate, total return, and max drawdown.

    Parameters
    ----------
    trades : pd.DataFrame
        Per-trade log from :func:`build_trade_log`.

    Returns
    -------
    dict
        Dictionary with keys:
        ``n_trades, win_rate, loss_rate, sharpe, total_return, max_drawdown``.
    """
    trades = trades.copy()
    trades["total_return"] = trades["total_return"] / 100

    trades["cum_equity"] = (1 + trades["total_return"]).cumprod()
    trades["cummax"] = trades["cum_equity"].cummax()
    trades["drawdown"] = trades["cum_equity"] / trades["cummax"] - 1

    return {
        "n_trades": len(trades),
        "win_rate": (trades["total_return"] > 0).mean(),
        "loss_rate": (trades["total_return"] < 0).mean(),
        "sharpe": trades["total_return"].mean() / trades["total_return"].std(),
        "total_return": trades["cum_equity"].iloc[-1] - 1,
        "max_drawdown": trades["drawdown"].min(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("CE_calc.csv")
    df = df.reset_index(drop=True)

    trades = build_trade_log(df)
    trades.to_csv("trades_ohnefilter_longonly.csv", index=False)

    m = compute_metrics(trades)

    print("===== Trade Metrics =====")
    print(f"Number of trades : {m['n_trades']}")
    print(f"Win rate         : {round(m['win_rate'], 4)}")
    print(f"Loss rate        : {round(m['loss_rate'], 4)}")
    print(f"Sharpe ratio     : {round(m['sharpe'], 4)}")
    print(f"Total return     : {round(m['total_return'] * 100, 2)} %")
    print(f"Max Drawdown     : {round(m['max_drawdown'] * 100, 2)} %")
