"""
data/download.py
----------------
Downloads historical OHLC candlestick data from Binance and persists it to
CSV for use by downstream pipeline steps.

Outputs
-------
OHLC_daily_from2020.csv
"""

from datetime import datetime

import numpy as np
import pandas as pd
from binance.client import Client

# ---------------------------------------------------------------------------
# Binance client
# ---------------------------------------------------------------------------
API_KEY = ""
SECRET_KEY = ""
client = Client(API_KEY, SECRET_KEY)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def download_data(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    return_type: str = "pct",
) -> pd.DataFrame:
    """Download historical kline data from Binance.

    Parameters
    ----------
    symbol : str
        Binance trading pair (e.g. ``"BTCUSDT"``).
    interval : str
        Binance kline interval (e.g. ``Client.KLINE_INTERVAL_1DAY``).
    start_date : str
        Start date in ``"YYYY-MM-DD"`` format.
    end_date : str
        End date in ``"YYYY-MM-DD"`` format.
    return_type : str
        ``"pct"`` for percentage returns, ``"log"`` for log returns.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with columns
        ``["Open", "High", "Low", "Close", "Volume", "Returns"]``.

    Raises
    ------
    ValueError
        If *return_type* is not ``"pct"`` or ``"log"``.
    """
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
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

    if return_type == "pct":
        df["Returns"] = df["Close"].pct_change() * 100
    elif return_type == "log":
        df["Returns"] = np.log(df["Close"]).diff() * 100
    else:
        raise ValueError("Invalid return_type: use 'pct' or 'log'.")

    df = df.dropna()[["Open", "High", "Low", "Close", "Volume", "Returns"]]
    df.to_csv("OHLC_daily_from2020.csv")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = download_data(
        symbol="BTCUSDT",
        interval=Client.KLINE_INTERVAL_1DAY,
        start_date="2017-09-09",
        end_date="2026-02-18",
        return_type="pct",
    )
    print(f"Downloaded {len(data)} rows.")
