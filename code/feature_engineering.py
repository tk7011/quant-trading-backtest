"""
ml/feature_engineering.py
--------------------------
Adds EMA-derived features and the binary classification target to the merged
dataset, producing a training-ready CSV.

Pipeline position
-----------------
Runs *after* ``data/merge.py`` and *before* ``ml/train_random_forest.py``.

Inputs
------
final_dataset.csv

Outputs
-------
final_dataset+ema.csv          — dataset with EMA columns
final_dataset+target+ema.csv   — dataset with EMA columns + target label
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# EMA features
# ---------------------------------------------------------------------------
def add_ema_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA columns and price-minus-EMA distance features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with at minimum a ``Close`` column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with:
        ``["EMA_10", "EMA_20", "EMA_50", "pice/ema50", "pice/ema20"]``.
    """
    df = df.copy()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["pice/ema50"] = df["Close"] - df["EMA_50"]
    df["pice/ema20"] = df["Close"] - df["EMA_20"]
    return df.dropna()


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------
def add_target(df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    """Add a binary forward-return target and a continuous 1-day target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with a ``Close`` and ``Returns`` column.
    horizon : int
        Rolling window (in bars) used for the binary ``target`` column.
        A bar is labelled ``1`` if the next *horizon* bars' cumulative return
        is positive, ``0`` otherwise.

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with:
        ``["target_1d", "target"]``.
    """
    df = df.copy()
    df["target_1d"] = df["Close"].pct_change(periods=1).shift(-1)
    df["target"] = (df["Returns"].rolling(horizon).sum().shift(-horizon) > 0).astype(int)
    return df.dropna()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("final_dataset.csv")

    df = add_ema_features(df)
    df.to_csv("final_dataset+ema.csv", index=False)
    print(f"EMA features written  →  final_dataset+ema.csv  ({len(df)} rows)")

    df = add_target(df, horizon=30)
    df.to_csv("final_dataset+target+ema.csv", index=False)
    print(f"Target added          →  final_dataset+target+ema.csv  ({len(df)} rows)")
