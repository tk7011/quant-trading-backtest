"""
ml/markov_regimes.py
--------------------
Fits a two-regime Markov Regime Switching (MRS) model on BTC daily returns,
runs stationarity and diagnostic tests, produces diagnostic plots, and
appends regime labels / probabilities to the dataset.

Pipeline role
-------------
This step runs *before* feature engineering. Its outputs (``Regime_2``,
``Prob_Regime0``, ``Prob_Regime1``) are consumed by ``data/merge.py`` and
later used as ML features in ``ml/train_random_forest.py``.

Outputs
-------
MRS_2017_2026_regimes_1w.csv   — dataset with regime columns appended
variance_plot.png
Transition_matrix_2regimes.png
returns_plot_2regimes.png
Regimes_plot_2regimes.png
Regime_Probabilities_plot_2regimes.png
Vol_MRS_bvol.png
"""

import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from binance.client import Client
from scipy.stats import normaltest
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import adfuller, kpss

plt.style.use("default")

# ---------------------------------------------------------------------------
# Binance client
# ---------------------------------------------------------------------------
client = Client(api_key="", api_secret="")


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_data(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    return_type: str = "log",
) -> pd.DataFrame:
    """Download historical kline data from Binance.

    Parameters
    ----------
    symbol, interval, start_date, end_date : str
        Standard Binance query parameters.
    return_type : str
        ``"pct"`` or ``"log"``.

    Returns
    -------
    pd.DataFrame
        OHLC + Returns, date-indexed.
    """
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    klines = client.get_historical_klines(symbol, interval, start_ts, end_ts)
    if not klines:
        raise ValueError("Data download failed: empty dataset returned.")

    raw_columns = [
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base", "Taker buy quote", "Ignore",
    ]
    df = pd.DataFrame(klines, columns=raw_columns)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    df["Date"] = pd.to_datetime(df["Close time"], unit="ms")
    df.set_index("Date", inplace=True)

    if return_type == "pct":
        df["Returns"] = df["Close"].pct_change() * 100
    elif return_type == "log":
        df["Returns"] = np.log(df["Close"]).diff() * 100
    else:
        raise ValueError("Invalid return_type: use 'pct' or 'log'.")

    return df.dropna()[["Open", "High", "Low", "Close", "Volume", "Returns"]]


# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------
def test_stationarity(data: pd.DataFrame) -> None:
    """Print ADF and KPSS stationarity test results for the Returns series."""
    print("\n--- Stationarity Tests ---")
    returns = data["Returns"].values

    adf_stat, adf_pval, *_ = adfuller(returns)
    print(f"ADF  statistic : {adf_stat:.4f}  |  p-value : {adf_pval:.4f}")

    kpss_stat, kpss_pval, *_ = kpss(returns, nlags="auto")
    print(f"KPSS statistic : {kpss_stat:.4f}  |  p-value : {kpss_pval:.4f}")

    print("\n--- Interpretation ---")
    print("ADF :", "Likely non-stationary." if adf_pval > 0.05 else "Likely stationary.")
    print("KPSS:", "Likely non-stationary." if kpss_pval < 0.05 else "Likely stationary.")


# ---------------------------------------------------------------------------
# AR order selection
# ---------------------------------------------------------------------------
def find_best_ar_order(data: pd.DataFrame, max_order: int = 5) -> int:
    """Select the AR lag order that minimises AIC for a 2-regime MRS model.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with a ``Returns`` column.
    max_order : int
        Maximum AR order to consider.

    Returns
    -------
    int
        Best AR order found.
    """
    best_aic, best_order = np.inf, 0
    for order in range(0, max_order + 1):
        try:
            model = MarkovRegression(
                endog=data["Returns"],
                k_regimes=2,
                order=order,
                switching_variance=True,
            )
            results = model.fit(disp=False)
            if results.aic < best_aic:
                best_aic, best_order = results.aic, order
        except Exception:
            continue
    return best_order


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
def fit_mrs_model(data: pd.DataFrame, ar_order: int = 0) -> MarkovRegression:
    """Fit a 2-regime Markov Regime Switching model.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with a ``Returns`` column.
    ar_order : int
        AR lag order (from :func:`find_best_ar_order`).

    Returns
    -------
    MarkovRegression
        Fitted statsmodels results object.
    """
    model = MarkovRegression(
        endog=data["Returns"],
        k_regimes=2,
        order=ar_order,
        switching_variance=True,
    )
    return model.fit()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def diagnose_model(results: MarkovRegression) -> None:
    """Run Ljung-Box autocorrelation and D'Agostino-Pearson normality tests."""
    print("\n--- Model Diagnostics ---")

    lb = acorr_ljungbox(results.resid, lags=[10], return_df=True)
    print("Ljung-Box Test (lag=10):")
    print(lb)

    stat, pval = normaltest(results.resid)
    print(f"\nNormality Test — statistic: {stat:.4f}  |  p-value: {pval:.4f}")
    print(
        "Residuals are"
        + (" NOT" if pval < 0.05 else "")
        + " normally distributed."
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_volatility_comparison(results: MarkovRegression) -> None:
    """Bar chart of estimated variance per regime."""
    variance_params = [
        p for p in results.model.param_names
        if p.startswith("sigma2") and p in results.params
    ]
    variances = [results.params[p] for p in variance_params]

    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=[f"Regime {i + 1}" for i in range(len(variances))],
        y=variances,
        palette="Blues_d",
    )
    plt.title("Estimated Variance by Regime")
    plt.ylabel("Variance")
    plt.savefig("variance_plot.png", dpi=500)
    plt.close()


def plot_transition_matrix(results: MarkovRegression) -> None:
    """Heatmap of the regime transition probability matrix."""
    tm = results.regime_transition
    if tm.ndim == 3:
        tm = tm[:, :, 0]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        tm,
        annot=True,
        fmt=".4f",
        cmap="Reds",
        xticklabels=[f"Regime {i + 1}" for i in range(results.k_regimes)],
        yticklabels=[f"Regime {i + 1}" for i in range(results.k_regimes)],
    )
    plt.title("Regime Transition Matrix")
    plt.xlabel("Next Regime")
    plt.ylabel("Current Regime")
    plt.savefig("Transition_matrix_2regimes.png", dpi=500)
    plt.close()


def plot_returns(data: pd.DataFrame) -> None:
    """Line chart of the Returns time series."""
    plt.figure(figsize=(12, 4))
    plt.plot(data.index, data["Returns"], label="Returns", color="steelblue")
    plt.title("Time Series of Returns")
    plt.legend()
    plt.savefig("returns_plot_2regimes.png", dpi=500)
    plt.close()


def plot_regimes(data: pd.DataFrame, results: MarkovRegression) -> None:
    """Scatter plot of returns coloured by regime assignment."""
    regimes = results.smoothed_marginal_probabilities.idxmax(axis=1)

    plt.figure(figsize=(12, 4))
    for regime in results.smoothed_marginal_probabilities.columns:
        mask = regimes == regime
        plt.scatter(
            data.index[mask],
            data["Returns"][mask],
            label=f"Regime {regime + 1}",
            alpha=0.6,
        )
    plt.title("Regime Assignment (Scatter)")
    plt.legend()
    plt.savefig("Regimes_plot_2regimes.png", dpi=500)
    plt.close()


def plot_regime_probabilities(results: MarkovRegression) -> None:
    """Smoothed marginal regime probabilities over time."""
    plt.figure(figsize=(12, 4))
    for i in range(results.k_regimes):
        plt.plot(
            results.smoothed_marginal_probabilities[i],
            label=f"Regime {i + 1} Probability",
        )
    plt.title("Regime Probabilities Over Time")
    plt.legend()
    plt.savefig("Regime_Probabilities_plot_2regimes.png", dpi=500)
    plt.close()


def interpret_results(results: MarkovRegression) -> None:
    """Print model summary, AIC/BIC/log-likelihood, and transition matrix."""
    print("\n--- Model Summary ---")
    print(results.summary())
    print(f"\nAIC : {results.aic:.2f}")
    print(f"BIC : {results.bic:.2f}")
    print(f"Log-Likelihood : {results.llf:.2f}")
    print("\n--- Transition Matrix ---")
    print(results.regime_transition)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Downloading data...")
    data = download_data("BTCUSDT", "1w", "2017-09-09", "2026-02-17", return_type="pct")
    print(f"Downloaded {data.shape[0]} rows.\n{data.head()}")

    test_stationarity(data)

    best_ar = find_best_ar_order(data, max_order=5)
    print(f"\nBest AR order (AIC): {best_ar}")

    results = fit_mrs_model(data, ar_order=best_ar)
    diagnose_model(results)
    interpret_results(results)

    # --- Diagnostic plots ---
    plot_volatility_comparison(results)
    plot_transition_matrix(results)
    plot_returns(data)
    plot_regimes(data, results)
    plot_regime_probabilities(results)

    # --- Append regime labels & probabilities ---
    regime_probs = results.smoothed_marginal_probabilities
    data["Regime_2"] = regime_probs.idxmax(axis=1)
    data["Prob_Regime0"] = regime_probs[0]
    data["Prob_Regime1"] = regime_probs[1]
    data.to_csv("MRS_2017_2026_regimes_1w.csv")
    print("\nMRS_2017_2026_regimes_1w.csv written.")

    # --- Rolling volatility + regime price chart ---
    data["Volatility_10"] = data["Returns"].rolling(10).std()
    data["Volatility_20"] = data["Returns"].rolling(20).std()
    data["Volatility_50"] = data["Returns"].rolling(50).std()

    data_plot = data.tail(400)
    fig, ax1 = plt.subplots(figsize=(14, 8))

    colors = ["blue", "yellow"]
    for regime in [0, 1]:
        subset = data_plot[data_plot["Regime_2"] == regime]
        ax1.scatter(
            subset.index,
            subset["Close"],
            label=f"Regime {regime}",
            alpha=0.6,
            s=30,
            color=colors[regime],
        )

    ax1.plot(data_plot.index, data_plot["Close"], color="black", label="BTC Close Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("BTC Price (USDT)")

    fig.suptitle("BTC Price with Regimes", fontsize=14, fontweight="bold")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
    plt.tight_layout()
    plt.savefig("Vol_MRS_bvol.png", dpi=300)
    plt.close()

    print(f"\nWorking directory: {os.getcwd()}")
