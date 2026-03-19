import numpy as np
import pandas as pd
df=pd.read_csv("final_dataset.csv")
df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
df["pice/ema50"] = df["Close"]-df["EMA_50"]
df["pice/ema20"] = df["Close"]-df["EMA_20"]
df=df.dropna()
df.to_csv("final_dataset+ema.csv")