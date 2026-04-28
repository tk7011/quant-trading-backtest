"""
strategy/optimise_parameters.py
--------------------------------
Grid-searches Chandelier Exit (period, multiplier) combinations and reports
the parameter set that maximises the annualised Sharpe ratio over the full
dataset (in-sample optimisation).

Usage
-----
Run this script directly after downloading OHLC data::

    python -m quant_backtest.strategy.optimise_parameters

Output
------
Prints the best parameter set to stdout.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from binance.client import Client

from quant_backtest.strategy.chandelier_exit import chandelier_exit

# ---------------------------------------------------------------------------
# Binance client
# ---------------------------------------------------------------------------
API_KEY = ""
SECRET_KEY = ""
client = Client(API_KEY, SECRET_KEY)

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "period": [1, 2, 4, 8, 10, 12, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    "multiplier": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
}


# ---------------------------------------------------------------------------
# Helper: data download (self-contained for this script)
# ---------------------------------------------------------------------------
def _download_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    klines = client.get_historical_klines(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_1DAY,
        start_str=start_ts,
        end_str=end_ts,
    )
    raw_columns = [
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Trades",
        "Taker buy base", "Taker buy quote", "Ignore",
    ]
    df = pd.DataFrame(klines, columns=raw_columns)
    df[["Open", "High", "Low", "Close", "Volume"]] = (
        df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    )
    df["Date"] = pd.to_datetime(df["Close time"], unit="ms")
    df.set_index("Date", inplace=True)
    df["Returns"] = df["Close"].pct_change() * 100
    return df.dropna()[["Open", "High", "Low", "Close", "Volume", "Returns"]]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_strategy(df: pd.DataFrame, period: int, multiplier: float) -> tuple[float, float]:
    """Return (annualised Sharpe ratio, total return %) for one parameter set.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLC + Returns dataset.
    period : int
        CE look-back period.
    multiplier : float
        CE ATR multiplier.

    Returns
    -------
    tuple[float, float]
        ``(sharpe, total_return)``
    """
    result = chandelier_exit(df, period=period, multiplier=multiplier)
    result["Strategy_Returns"] = result["Signal"].shift(1) * result["Returns"]

    sharpe = (
        result["Strategy_Returns"].mean()
        / result["Strategy_Returns"].std()
        * np.sqrt(252)
    )
    total_return = result["Strategy_Returns"].sum()
    return float(sharpe), float(total_return)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = _download_data("BTCUSDT", "2020-01-01", "2026-02-18")

    results = []
    for period in PARAM_GRID["period"]:
        for multiplier in PARAM_GRID["multiplier"]:
            sharpe, ret = evaluate_strategy(data, period, multiplier)
            results.append({
                "period": period,
                "multiplier": multiplier,
                "sharpe": sharpe,
                "return": ret,
            })

    best = max(results, key=lambda x: x["sharpe"])
    print("Best Global Parameters:", best)
