"""
data/merge.py
-------------
Combines per-feature CSV files produced by earlier pipeline steps into a
single ``final_dataset.csv`` ready for feature engineering.

Expected inputs (inside ``folder_for_glob/``)
---------------------------------------------
- CE_calc.csv          (Chandelier Exit features)
- MRS_2017_2026_*.csv  (Markov regime labels & probabilities)
- Any additional CSVs with a ``Date`` index

Output
------
final_dataset.csv
"""

import glob

import pandas as pd

# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------
KEEP_COLUMNS = [
    "Returns",
    "Volume",
    "High",
    "Low",
    "stopdist_long",
    "stopdist_short",
    "Signal",
    "Regime_2",
    "Prob_Regime0",
    "Prob_Regime1",
    "ROC_5",
    "ROC_22",
]


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def merge_feature_csvs(glob_pattern: str = "folder_for_glob/*.csv") -> pd.DataFrame:
    """Read all CSVs matching *glob_pattern* and merge them on the Date index.

    Parameters
    ----------
    glob_pattern : str
        Glob expression pointing at the folder that holds the per-feature CSVs.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing :data:`KEEP_COLUMNS`, with NaN rows dropped.
    """
    files = glob.glob(glob_pattern)
    dfs = [pd.read_csv(f, parse_dates=["Date"], index_col="Date") for f in files]

    merged = pd.concat(dfs, axis=1)[KEEP_COLUMNS]
    merged = merged.dropna()
    merged.to_csv("final_dataset.csv")
    return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = merge_feature_csvs()
    print(f"Merged dataset shape: {df.shape}")
    print(df.head())
