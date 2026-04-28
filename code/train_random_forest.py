"""
ml/train_random_forest.py
--------------------------
Trains a Random Forest classifier to predict the sign of the next 30-bar
cumulative return, then appends ML probabilities to the test-set CSV for use
by the signal filter.

Pipeline position
-----------------
Runs *after* ``ml/feature_engineering.py`` and *before*
``backtest/signal_filter.py``.

Inputs
------
final_dataset+target+ema.csv

Outputs
-------
rf_30day_model.pkl             — serialised RandomForestClassifier
final_dataset_with_RF.csv      — test-set rows + ``ml_prob`` column
RF Feature Importance.png
roc_curve.png
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURES = [
    "pice/ema50",
    "stopdist_short",
    "stopdist_long",
    "Prob_Regime0",
]
TARGET = "target"
TRAIN_RATIO = 0.7
PROBABILITY_THRESHOLD = 0.55

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features=0.8,
    random_state=42,
    n_jobs=-1,
)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame) -> RandomForestClassifier:
    """Fit a Random Forest on the training split and evaluate on the test split.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing ``FEATURES`` columns and ``TARGET`` column.

    Returns
    -------
    RandomForestClassifier
        Fitted model (also saved to ``rf_30day_model.pkl``).
    """
    split = int(len(df) * TRAIN_RATIO)
    X_train, X_test = df[FEATURES].iloc[:split], df[FEATURES].iloc[split:]
    y_train, y_test = df[TARGET].iloc[:split], df[TARGET].iloc[split:]

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    joblib.dump(rf, "rf_30day_model.pkl")

    # --- Predictions ---
    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= PROBABILITY_THRESHOLD).astype(int)

    # --- Console metrics ---
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    # --- Feature importances ---
    print("\nFeature Importances:")
    for feat, imp in zip(FEATURES, rf.feature_importances_):
        print(f"  {feat:<20} {imp:.4f}")

    # --- Feature importance plot ---
    plt.figure()
    plt.barh(FEATURES, rf.feature_importances_)
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig("RF Feature Importance.png")
    plt.close()

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Random Forest")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()

    # --- Append probabilities to test set and save ---
    df_test = df.iloc[split:].copy()
    df_test["ml_prob"] = y_prob
    df_test.to_csv("final_dataset_with_RF.csv", index=False)

    print(f"\nml_prob stats:\n{df_test['ml_prob'].describe()}")
    return rf


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("final_dataset+target+ema.csv")
    model = train_and_evaluate(df)
    print("\nModel saved → rf_30day_model.pkl")
