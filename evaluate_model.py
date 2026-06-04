"""
evaluate_model.py — how good is the model actually?
Generates R², MAE, feature importance chart, and prediction vs actual plot.

Run: python evaluate_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "Adj Close", "Close", "High", "Low", "Open", "Volume",
    "daily_return", "ma_7", "ma_30", "ma_90", "volatility_30"
]


def evaluate():
    df = pd.read_csv("data/combined_features.csv").dropna()
    model = joblib.load("models/random_forest_model.joblib")

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols]
    y = df["target_30d_return"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    # ── Metrics ───────────────────────────────────────────────
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    directional_accuracy = np.mean(np.sign(y_pred) == np.sign(y_test)) * 100

    print("\n=== Model Evaluation ===")
    print(f"R² Score:              {r2:.4f}")
    print(f"MAE:                   {mae:.4f}")
    print(f"RMSE:                  {rmse:.4f}")
    print(f"Directional Accuracy:  {directional_accuracy:.1f}%")

    # ── Plots ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f0f")

    for ax in axes:
        ax.set_facecolor("#0f0f0f")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 1. Predicted vs Actual scatter
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.3, s=5, color="#378ADD")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax1.plot(lims, lims, color="#E24B4A", linewidth=1.5, linestyle="--", label="Perfect prediction")
    ax1.set_xlabel("Actual 30d Return", color="white")
    ax1.set_ylabel("Predicted 30d Return", color="white")
    ax1.set_title(f"Predicted vs Actual  (R²={r2:.3f})", color="white")
    ax1.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)

    # 2. Feature importance
    ax2 = axes[1]
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)
    colors = ["#E24B4A" if i >= len(importances) - 3 else "#378ADD"
              for i in range(len(importances))]
    ax2.barh(importances.index, importances.values, color=colors)
    ax2.set_title("Feature Importance", color="white")
    ax2.set_xlabel("Importance", color="white")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_evaluation.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()
    print("Evaluation chart saved → outputs/model_evaluation.png")

    return {
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "directional_accuracy": round(directional_accuracy, 1),
    }


if __name__ == "__main__":
    evaluate()
