import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from binance.client import Client

plt.style.use("default")

# =============================================================================
# Binance Client Setup
# =============================================================================
API_KEY = ""
SECRET_KEY = ""
client = Client(API_KEY, SECRET_KEY)

# =============================================================================
# Data Download
# =============================================================================
def download_data(symbol: str, start_date: str, end_date: str, return_type: str = "pct") -> pd.DataFrame:
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    klines = client.get_historical_klines(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_1DAY,
        start_str=start_ts,
        end_str=end_ts
    )

    df = pd.DataFrame(
        klines,
        columns=[
            "Open time", "Open", "High", "Low", "Close", "Volume",
            "Close time", "Quote asset volume", "Trades",
            "Taker buy base", "Taker buy quote", "Ignore"
        ],
    )

    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df["Date"] = pd.to_datetime(df["Close time"], unit="ms")
    df.set_index("Date", inplace=True)

    if return_type == "pct":
        df["Returns"] = df["Close"].pct_change() * 100
    elif return_type == "log":
        df["Returns"] = np.log(df["Close"]).diff() * 100
    else:
        raise ValueError("Invalid return_type: use 'pct' or 'log'.")
    
    return df.dropna()[["Open", "High", "Low", "Close", "Volume", "Returns"]]

# =============================================================================
# Strategy: Chandelier Exit
# =============================================================================
def chandelier_exit(df, period=22, multiplier=3, use_close=True):
    df = df.copy()
    df["prev_close"] = df["Close"].shift(1)
    df["TR"] = df[["High", "Low", "prev_close"]].max(axis=1) - df[["High", "Low", "prev_close"]].min(axis=1)
    df["ATR"] = df["TR"].ewm(alpha=1 / period, adjust=False).mean()

    if use_close:
        df["highest"] = df["Close"].shift(1).rolling(period).max()
        df["lowest"] = df["Close"].shift(1).rolling(period).min()
    else:
        df["highest"] = df["High"].shift(1).rolling(period).max()
        df["lowest"] = df["Low"].shift(1).rolling(period).min()

    first_idx = df.index[period]
    df.loc[first_idx, "LongStop"] = df.loc[first_idx, "highest"] - multiplier * df.loc[first_idx, "ATR"]
    df.loc[first_idx, "ShortStop"] = df.loc[first_idx, "lowest"] + multiplier * df.loc[first_idx, "ATR"]

    for i in range(period + 1, len(df)):
        curr, prev = df.index[i], df.index[i - 1]
        new_long = df["highest"].loc[curr] - multiplier * df["ATR"].loc[curr]
        new_short = df["lowest"].loc[curr] + multiplier * df["ATR"].loc[curr]

        df.loc[curr, "LongStop"] = max(new_long, df.loc[prev, "LongStop"]) if df.loc[prev, "Close"] > df.loc[prev, "LongStop"] else new_long
        df.loc[curr, "ShortStop"] = min(new_short, df.loc[prev, "ShortStop"]) if df.loc[prev, "Close"] < df.loc[prev, "ShortStop"] else new_short

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

    return df

# =============================================================================
# Strategy Evaluation (global)
# =============================================================================
def evaluate_strategy(df, period, multiplier):
    strat_df = chandelier_exit(df, period=period, multiplier=multiplier)
    strat_df["Strategy_Returns"] = strat_df["Signal"].shift(1) * strat_df["Returns"]

    sharpe = strat_df["Strategy_Returns"].mean() / strat_df["Strategy_Returns"].std() * np.sqrt(252)
    total_return = strat_df["Strategy_Returns"].sum()
    return float(sharpe), float(total_return)

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    data = download_data("BTCUSDT", "2020-01-01", "2026-02-18", return_type="pct")

    param_grid = {"period": [1,2,4,8,10,12,15,16,18,20,21,22,23,24,25,26,27,28,29,30],
                  "multiplier": [1,1.5,2,2.5, 3.0, 3.5, 4.0, 4.5,5]}

    results_list = []
    for period in param_grid["period"]:
        for mult in param_grid["multiplier"]:
            sharpe, ret = evaluate_strategy(data, period, mult)
            results_list.append({"period": period, "multiplier": mult, "sharpe": sharpe, "return": ret})

    best_params = max(results_list, key=lambda x: x["sharpe"])
    print("Best Global Parameters:", best_params)
