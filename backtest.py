"""
backtest.py — did the predictions actually work?
Simulates buying the top predicted stocks and measures real returns.

Run: python backtest.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import joblib

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "Adj Close", "Close", "High", "Low", "Open", "Volume",
    "daily_return", "ma_7", "ma_30", "ma_90", "volatility_30"
]


def run_backtest(top_n: int = 3):
    df = pd.read_csv("data/combined_features.csv").dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    model = joblib.load("models/random_forest_model.joblib")

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    df["predicted_return"] = model.predict(df[feature_cols])

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df = df.dropna(subset=[price_col])

    dates = sorted(df["Date"].unique())
    results = []

    for i in range(0, len(dates) - 30, 30):
        signal_date = dates[i]
        hold_date = dates[min(i + 30, len(dates) - 1)]

        signal_df = df[df["Date"] == signal_date]
        top_tickers = (
            signal_df.sort_values("predicted_return", ascending=False)
            .head(top_n)["ticker"]
            .tolist()
        )

        portfolio_returns = []
        for ticker in top_tickers:
            entry = df[(df["Date"] == signal_date) & (df["ticker"] == ticker)]
            exit_ = df[(df["Date"] == hold_date) & (df["ticker"] == ticker)]

            if entry.empty or exit_.empty:
                continue

            entry_price = float(entry[price_col].values[0])
            exit_price = float(exit_[price_col].values[0])
            if entry_price == 0:
                continue
            actual_return = (exit_price - entry_price) / entry_price
            portfolio_returns.append(actual_return)

        if portfolio_returns:
            avg_return = np.mean(portfolio_returns)
            results.append({
                "signal_date": signal_date,
                "hold_date": hold_date,
                "top_tickers": ", ".join(top_tickers),
                "avg_actual_return": round(avg_return * 100, 2),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)

    total_periods = len(results_df)
    win_rate = (results_df["avg_actual_return"] > 0).mean() * 100
    avg_return = results_df["avg_actual_return"].mean()
    best = results_df["avg_actual_return"].max()
    worst = results_df["avg_actual_return"].min()

    print("\n=== Backtest Results ===")
    print(f"Periods tested:    {total_periods}")
    print(f"Win rate:          {win_rate:.1f}%")
    print(f"Avg return/period: {avg_return:.2f}%")
    print(f"Best period:       {best:.2f}%")
    print(f"Worst period:      {worst:.2f}%")
    print(results_df.to_string(index=False))

    results_df["cumulative_return"] = (
        (1 + results_df["avg_actual_return"] / 100).cumprod() - 1
    ) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor("#0f0f0f")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#0f0f0f")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    colors = ["#4CAF50" if r > 0 else "#E24B4A" for r in results_df["avg_actual_return"]]
    ax1.bar(range(len(results_df)), results_df["avg_actual_return"], color=colors)
    ax1.axhline(0, color="white", linewidth=0.5, linestyle="--")
    ax1.set_title("Per-Period Portfolio Return (%)", color="white")
    ax1.set_xlabel("Period", color="white")
    ax1.set_ylabel("Return (%)", color="white")

    ax2.plot(range(len(results_df)), results_df["cumulative_return"], color="#378ADD", linewidth=2)
    ax2.fill_between(range(len(results_df)), results_df["cumulative_return"], alpha=0.2, color="#378ADD")
    ax2.axhline(0, color="white", linewidth=0.5, linestyle="--")
    ax2.set_title("Cumulative Portfolio Return (%)", color="white")
    ax2.set_xlabel("Period", color="white")
    ax2.set_ylabel("Cumulative Return (%)", color="white")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "backtest_returns.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()
    print(f"\nBacktest chart saved → outputs/backtest_returns.png")

    return results_df, {"win_rate": win_rate, "avg_return": avg_return, "best": best, "worst": worst}


if __name__ == "__main__":
    run_backtest(top_n=3)
