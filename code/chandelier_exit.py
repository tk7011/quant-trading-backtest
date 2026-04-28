"""
strategy/chandelier_exit.py
---------------------------
Implements the Chandelier Exit (CE) trend-following indicator and generates
long/short signals based on ATR-derived stop levels.

This module is used in two contexts:

1. **Signal generation** (``CE_distance.py`` / ``CE_distance.txt``)
   Computes stop-level distances that serve as ML features.

2. **Parameter optimisation** (``best_ce_parameters.py``)
   Evaluates every (period, multiplier) combination via Sharpe ratio.

Outputs
-------
CE_calc.csv   — OHLC + CE features + trading signals
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Core indicator
# ---------------------------------------------------------------------------
def chandelier_exit(
    df: pd.DataFrame,
    period: int = 22,
    multiplier: float = 3.0,
    use_close: bool = True,
) -> pd.DataFrame:
    """Compute Chandelier Exit stop levels and generate trading signals.

    The long stop rises with the highest close (or high) over *period* bars
    minus ``multiplier × ATR``. The short stop falls with the lowest close
    (or low) plus ``multiplier × ATR``. Signals flip when price crosses the
    opposite stop.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame with a DatetimeIndex and columns
        ``["Open", "High", "Low", "Close", "Returns"]``.
    period : int
        Look-back window for ATR and highest/lowest price calculations.
    multiplier : float
        ATR multiplier controlling stop distance.
    use_close : bool
        If ``True``, use closing prices for highest/lowest; otherwise use
        High/Low candle extremes.

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with:
        ``["LongStop", "ShortStop", "stopdist_long", "stopdist_short", "Signal"]``
        and saved to ``CE_calc.csv``.
    """
    df = df.copy()

    # --- True Range & ATR ---
    df["prev_close"] = df["Close"].shift(1)
    df["TR"] = (
        df[["High", "Low", "prev_close"]].max(axis=1)
        - df[["High", "Low", "prev_close"]].min(axis=1)
    )
    df["ATR"] = df["TR"].ewm(alpha=1 / period, adjust=False).mean()

    # --- Rolling highest / lowest ---
    if use_close:
        df["highest"] = df["Close"].shift(1).rolling(period).max()
        df["lowest"] = df["Close"].shift(1).rolling(period).min()
    else:
        df["highest"] = df["High"].shift(1).rolling(period).max()
        df["lowest"] = df["Low"].shift(1).rolling(period).min()

    # --- Initialise stop levels at bar `period` ---
    first_idx = df.index[period]
    df.loc[first_idx, "LongStop"] = (
        df.loc[first_idx, "highest"] - multiplier * df.loc[first_idx, "ATR"]
    )
    df.loc[first_idx, "ShortStop"] = (
        df.loc[first_idx, "lowest"] + multiplier * df.loc[first_idx, "ATR"]
    )

    # --- Ratchet stops forward ---
    for i in range(period + 1, len(df)):
        curr, prev = df.index[i], df.index[i - 1]
        new_long = df["highest"].loc[curr] - multiplier * df["ATR"].loc[curr]
        new_short = df["lowest"].loc[curr] + multiplier * df["ATR"].loc[curr]

        df.loc[curr, "LongStop"] = (
            max(new_long, df.loc[prev, "LongStop"])
            if df.loc[prev, "Close"] > df.loc[prev, "LongStop"]
            else new_long
        )
        df.loc[curr, "ShortStop"] = (
            min(new_short, df.loc[prev, "ShortStop"])
            if df.loc[prev, "Close"] < df.loc[prev, "ShortStop"]
            else new_short
        )

        # Distance features (normalised by ATR)
        df["stopdist_long"] = (df["Close"] - df["LongStop"]) / df["ATR"]
        df["stopdist_short"] = (df["ShortStop"] - df["Close"]) / df["ATR"]

    # --- Signal generation ---
    df["Signal"] = 0
    pos = 0
    for i in range(1, len(df)):
        prev, curr = df.index[i - 1], df.index[i]
        if pos == 0:
            if df.loc[curr, "Close"] > df.loc[prev, "ShortStop"]:
                df.loc[curr, "Signal"], pos = 1, 1
            elif df.loc[curr, "Close"] < df.loc[prev, "LongStop"]:
                df.loc[curr, "Signal"], pos = -1, -1
        elif pos == 1 and df.loc[curr, "Close"] < df.loc[prev, "LongStop"]:
            df.loc[curr, "Signal"], pos = -1, -1
        elif pos == -1 and df.loc[curr, "Close"] > df.loc[prev, "ShortStop"]:
            df.loc[curr, "Signal"], pos = 1, 1

    output_cols = [
        "Open", "Close", "Volume", "Returns",
        "High", "Low", "LongStop", "ShortStop",
        "stopdist_long", "stopdist_short", "Signal",
    ]
    result = df[output_cols]
    result.to_csv("CE_calc.csv")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(
        "OHLC_daily_from2020.csv",
        parse_dates=["Date"],
        index_col="Date",
    )
    chandelier_exit(df, period=10, multiplier=3.5, use_close=True)
    print("CE_calc.csv written.")
