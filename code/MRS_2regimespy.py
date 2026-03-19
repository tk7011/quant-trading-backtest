from binance.client import Client
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns  # For heatmap and enhanced visualizations
from sklearn.preprocessing import RobustScaler

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import normaltest
from typing import Optional, Tuple
import os

plt.style.use('default')
#################################################################################################################
from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize Binance client (replace with your API keys if needed)
client = Client(api_key="", api_secret="")

def download_data(symbol: str, interval: str, start_date: str, end_date: str, return_type: str = 'log') -> pd.DataFrame:
    """
    Downloads historical crypto data from Binance and calculates returns.

    Parameters:
        symbol (str): Binance trading pair, e.g. 'BTCUSDT'
        interval (str): Binance interval, e.g. '1d'
        start_date (str): 'YYYY-MM-DD'
        end_date (str): 'YYYY-MM-DD'
        return_type (str): 'pct' for percent change, 'log' for log returns

    Returns:
        pd.DataFrame: Date-indexed DataFrame with OHLC and Returns
    """
    try:
        # Binance requires timestamps in milliseconds
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        # Fetch historical klines
        klines = client.get_historical_klines(symbol, interval, start_ts, end_ts)
        if not klines:
            raise ValueError("Data download failed: Empty dataset returned.")

        # Create DataFrame
        df = pd.DataFrame(klines, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base', 'Taker buy quote', 'Ignore'
        ])
        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        df['Date'] = pd.to_datetime(df['Close time'], unit='ms')
        
        df.set_index('Date', inplace=True)

        # Calculate returns
        if return_type == 'pct':
            df['Returns'] = df['Close'].pct_change() * 100
        elif return_type == 'log':
            df['Returns'] = np.log(df['Close']).diff() * 100
        else:
            raise ValueError("Invalid return_type: use 'pct' or 'log'.")

        df = df.dropna()
        data=df[['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']]
        return data

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

################################################################################
def test_stationarity(data: pd.DataFrame) -> None:
    """
    Checks the stationarity of the time series using ADF and KPSS tests.
    """
    print("\n--- Stationarity Tests ---")
    returns = data['Returns'].values

    # ADF test
    adf_result = adfuller(returns)
    print(f"ADF Test Statistic: {adf_result[0]:.4f}")
    print(f"ADF p-value: {adf_result[1]:.4f}")

    # KPSS test
    kpss_result = kpss(returns, nlags='auto')
    print(f"\nKPSS Test Statistic: {kpss_result[0]:.4f}")
    print(f"KPSS p-value: {kpss_result[1]:.4f}")

    # Interpretation
    print("\n--- Interpretation ---")
    if adf_result[1] > 0.05:
        print("ADF: The series is likely non-stationary (unit root present).")
    else:
        print("ADF: The series is likely stationary.")

    if kpss_result[1] < 0.05:
        print("KPSS: The series is likely non-stationary.")
    else:
        print("KPSS: The series is likely stationary.")


def find_best_ar_order(data: pd.DataFrame, max_order: int = 5) -> int:
    """
    Finds the best AR(p) order based on the AIC criterion.
    Tries orders from 0 up to max_order.
    """
    best_aic = np.inf
    best_order = 0

    for order in range(0, max_order + 1):
        try:
            model = MarkovRegression(
                endog=data['Returns'],
                k_regimes=2,
                order=order,
                switching_variance=True
            )
            results = model.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
        except:
            continue
    return best_order
############################################################################################################
def fit_mrs_model(
    data: pd.DataFrame,
    ar_order: int = 0
) -> MarkovRegression:
    """
    Builds and fits a two-regime Markov Regime Switching model on the given data.
    """
    # k_regimes=2 is fixed to reflect our two-regime assumption
    model = MarkovRegression(
        endog=data['Returns'],
        k_regimes=2,
        order=ar_order,
        switching_variance=True
    )
    results = model.fit()
    return results
#########################################################################################################################
def diagnose_model(results: MarkovRegression) -> None:
    """
    Performs model diagnostics:
    - Ljung-Box test for autocorrelation
    - D'Agostino-Pearson normality test
    """
    print("\n--- Model Diagnostics ---")

    print("Ljung-Box Test Results (lag=10):")
    lb_test = acorr_ljungbox(results.resid, lags=[10], return_df=True)
    print(lb_test)

    print("\nNormality Test (D'Agostino-Pearson):")
    normality = normaltest(results.resid)
    print(f"Statistic: {normality.statistic:.4f}")
    print(f"p-value: {normality.pvalue:.4f}")

    if normality.pvalue < 0.05:
        print("Conclusion: Residuals are not normally distributed.")
    else:
        print("Conclusion: Residuals appear to be normally distributed.")
##########################################################################################################
def plot_volatility_comparison(results: MarkovRegression) -> None:
    """
    Plots each regime's estimated variance as a bar chart.
    """
    # Get all variance parameter names that exist in the results
    variance_params = [p for p in results.model.param_names 
                       if p.startswith('sigma2') and p in results.params]
    
    # Directly access the values using the parameter name as key
    variances = [results.params[p] for p in variance_params]
    
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=[f"Regime {i+1}" for i in range(len(variances))],
        y=variances,
        palette="Blues_d"
    )
    plt.title("Estimated Variance by Regime")
    plt.ylabel("Variance")
    plt.savefig("variance_plot.png", dpi=500)
    
############################################
def plot_transition_matrix(results: MarkovRegression) -> None:
    """
    Displays the regime transition matrix as a heatmap.
    """
    transition_matrix = results.regime_transition

    if transition_matrix.ndim == 3:
        transition_matrix = transition_matrix[:, :, 0]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt=".4f",
        cmap="Reds",
        xticklabels=[f"Regime {i+1}" for i in range(results.k_regimes)],
        yticklabels=[f"Regime {i+1}" for i in range(results.k_regimes)]
    )
    plt.title("Regime Transition Matrix")
    plt.xlabel("Next Regime")
    plt.ylabel("Current Regime")
    # Save the figure in the current working directory
    plt.savefig("Transition_matrix_2regimes.png", dpi=500)

################################################################################################
def plot_returns(data: pd.DataFrame) -> None:
    """
    Plots the time series returns.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(data.index, data['Returns'], label='Returns', color='steelblue')
    plt.title("Time Series of Returns")
    plt.legend()
    # Save the figure in the current working directory
    plt.savefig("returns_plot_2regimes.png", dpi=500)


def plot_regimes(data: pd.DataFrame, results: MarkovRegression) -> None:
    """
    Scatter plot showing which data points belong to which regime.
    """
    regimes = results.smoothed_marginal_probabilities.idxmax(axis=1)

    plt.figure(figsize=(12, 4))
    for regime in results.smoothed_marginal_probabilities.columns:
        mask = (regimes == regime)
        plt.scatter(data.index[mask], data['Returns'][mask],
                    label=f"Regime {regime+1}", alpha=0.6)
    plt.title("Regime Assignment (Scatter)")
    plt.legend()
    # Save the figure in the current working directory
    plt.savefig("Regimes_plot_2regimes.png", dpi=500)

def plot_regime_probabilities(results: MarkovRegression) -> None:
    """
    Plots the smoothed marginal probabilities for each regime over time.
    """
    plt.figure(figsize=(12, 4))
    for i in range(results.k_regimes):
        plt.plot(results.smoothed_marginal_probabilities[i],
                 label=f"Regime {i+1} Probability")
    plt.title("Regime Probabilities Over Time")
    plt.legend()
    # Save the figure in the current working directory
    plt.savefig("Regime_Probabilities_plot_2regimes.png", dpi=500)

#########################
def interpret_results(results: MarkovRegression) -> None:
    """
    Prints key metrics and parameters from the fitted model:
    - Summary
    - AIC, BIC, log-likelihood
    - Regime variances
    - Transition matrix
    """
    print("\n--- Model Summary ---")
    print(results.summary())

    print("\n--- Key Parameters ---")
    print(f"AIC: {results.aic:.2f}")
    print(f"BIC: {results.bic:.2f}")
    print(f"Log-Likelihood: {results.llf:.2f}")

    
    print("\n--- Transition Matrix ---")
    print(results.regime_transition)










# 1) Data Download
print("Starting data download...")
data = download_data("BTCUSDT","1w",'2017-09-09','2026-02-17', return_type='pct')
print(f"\nDownloaded data shape: {data.shape}")
print(f"Sample data:\n{data.head()}")
test_stationarity(data)
best_ar = find_best_ar_order(data,5)
print(f"\nBest AR order (by AIC) for 3 regimes: {best_ar}")

results = fit_mrs_model(data, ar_order=best_ar)

diagnose_model(results)
plot_volatility_comparison(results)
plot_transition_matrix(results)
plot_regimes(data, results)
plot_regime_probabilities(results)
interpret_results(results)
plot_returns(data)

# Probabilities of each regime at each time
regime_probs = results.smoothed_marginal_probabilities

# Assign the regime with the highest probability
data["Regime_2"] = regime_probs.idxmax(axis=1)
data["Prob_Regime0"] = regime_probs[0]
data["Prob_Regime1"] = regime_probs[1]

print(data)
data.to_csv("MRS_2017_2026_regimes_1w.csv")

# Plot results
# =======================
# 1) Price with Regimes
# =======================
# Calculate rolling volatility (30-day std of returns)
# --- Compute rolling volatilities ---

data["Volatility_10"] = data["Returns"].rolling(10).std()
data["Volatility_20"] = data["Returns"].rolling(20).std()
data["Volatility_50"] = data["Returns"].rolling(50).std()

# --- Take last 100 points for plotting ---
data_50 = data.tail(400)

# --- Plot setup ---
fig, ax1 = plt.subplots(figsize=(14, 8))

# --- Regimes as scatter ---
colors = ["blue", "yellow"]
for regime in [0,1]:
    regime_data = data_50[data_50["Regime_2"] == regime]
    ax1.scatter(
        regime_data.index,
        regime_data["Close"],
        label=f"Regime {regime}",
        alpha=0.6,
        s=30,
        color=colors[regime]
    )

# --- Price line ---
ax1.plot(data_50.index, data_50["Close"], color="black", label="BTC Close Price")
ax1.set_xlabel("Date")
ax1.set_ylabel("BTC Price (USDT)", color="black")
ax1.tick_params(axis="y", labelcolor="black")
"""
# --- Secondary axis for volatilities ---
ax2 = ax1.twinx()
ax2.plot(data_50.index, data_50["Volatility_10"], color="orange", label="10D Volatility")
ax2.plot(data_50.index, data_50["Volatility_20"], color="red", label="20D Volatility")
ax2.set_ylabel("Volatility (Std of Returns)", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")

# --- BVOL (if you want it) ---
#ax2.plot(data_bvol_50.index, data_bvol_50["BVOL24H"], color="green", label="BVOL")
"""
# --- Title & Legend ---
fig.suptitle("BTC Price with Regimes and Volatility", fontsize=14, fontweight="bold")
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))

plt.tight_layout()
plt.savefig("Vol_MRS_bvol.png", dpi=300)
print(os.getcwd())