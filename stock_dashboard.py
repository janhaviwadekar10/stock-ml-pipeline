"""
dashboard.py — Streamlit dashboard for Smart Stock Screener
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Smart Stock Screener", page_icon="📈", layout="wide")

FEATURE_COLS = [
    "Adj Close", "Close", "High", "Low", "Open", "Volume",
    "daily_return", "ma_7", "ma_30", "ma_90", "volatility_30"
]

TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA"]

st.title("📈 Smart Stock Screener")
st.caption("ML-powered equity ranking · Random Forest · 30-day return prediction")

with st.sidebar:
    st.header("Settings")
    selected_tickers = st.multiselect("Tickers", TICKERS, default=TICKERS)
    top_n = st.slider("Top N picks", 1, 7, 3)
    show_eval = st.checkbox("Show model evaluation", value=True)
    show_backtest = st.checkbox("Show backtest results", value=True)
    run = st.button("Run Screener 🚀", use_container_width=True)

def dark_fig(figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax

def dark_fig2(figsize=(14, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#0f0f0f")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return fig, axes

@st.cache_data(show_spinner=False)
def load_data_and_predict(tickers):
    frames = []
    for ticker in tickers:
        path = Path(f"data/{ticker}_features.csv")
        if path.exists():
            df = pd.read_csv(path)
            df["ticker"] = ticker
            frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True).dropna()
    model = joblib.load("models/random_forest_model.joblib")
    feature_cols = [c for c in FEATURE_COLS if c in combined.columns]
    combined["predicted_return"] = model.predict(combined[feature_cols])
    return combined, model, feature_cols

if run:
    if not Path("models/random_forest_model.joblib").exists():
        st.error("Model not found. Run main_pipeline.py first!")
        st.stop()

    with st.spinner("Loading data and scoring stocks..."):
        result = load_data_and_predict(selected_tickers)
        if result is None:
            st.error("No data found. Run main_pipeline.py first!")
            st.stop()
        df, model, feature_cols = result

    latest_date = df["Date"].max()
    latest = df[df["Date"] == latest_date].sort_values("predicted_return", ascending=False)
    top_picks = latest.head(top_n)

    # ── Metrics ───────────────────────────────────────────────
    st.markdown("### Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers Scored", len(selected_tickers))
    c2.metric("Latest Date", latest_date)
    c3.metric("Top Pick", top_picks.iloc[0]["ticker"] if len(top_picks) else "—")
    c4.metric("Predicted Return", f"{top_picks.iloc[0]['predicted_return']*100:.2f}%" if len(top_picks) else "—")

    st.divider()

    # ── Top picks ─────────────────────────────────────────────
    st.markdown(f"### Top {top_n} Predicted Stocks")
    display = top_picks[["ticker", "predicted_return", "volatility_30", "ma_7", "ma_30"]].copy()
    display["predicted_return"] = (display["predicted_return"] * 100).round(2).astype(str) + "%"
    display["volatility_30"] = display["volatility_30"].round(4)
    display["ma_7"] = display["ma_7"].round(2)
    display["ma_30"] = display["ma_30"].round(2)
    display.columns = ["Ticker", "Predicted 30d Return", "Volatility (30d)", "MA 7", "MA 30"]
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()

    # ── Price chart for top pick ───────────────────────────────
    top_ticker = top_picks.iloc[0]["ticker"]
    st.markdown(f"### Price History — {top_ticker}")
    ticker_df = df[df["ticker"] == top_ticker].sort_values("Date")
    price_col = "Adj Close" if "Adj Close" in ticker_df.columns else "Close"

    fig, ax = dark_fig(figsize=(12, 3))
    ax.plot(pd.to_datetime(ticker_df["Date"]), ticker_df[price_col], color="#378ADD", linewidth=1.5)
    ax.plot(pd.to_datetime(ticker_df["Date"]), ticker_df["ma_30"], color="#E24B4A",
            linewidth=1, linestyle="--", label="MA 30")
    ax.plot(pd.to_datetime(ticker_df["Date"]), ticker_df["ma_90"], color="#F7C948",
            linewidth=1, linestyle="--", label="MA 90")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price ($)", color="white")
    ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    st.pyplot(fig)

    st.divider()

    # ── Predicted return ranking bar chart ────────────────────
    st.markdown("### Predicted Return Ranking")
    fig2, ax2 = dark_fig(figsize=(10, 4))
    colors = ["#4CAF50" if r > 0 else "#E24B4A" for r in latest["predicted_return"]]
    ax2.bar(latest["ticker"], latest["predicted_return"] * 100, color=colors)
    ax2.axhline(0, color="white", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Predicted 30d Return (%)", color="white")
    st.pyplot(fig2)

    # ── Model evaluation ──────────────────────────────────────
    if show_eval:
        st.divider()
        st.markdown("### Model Evaluation")
        X = df[feature_cols]
        y = df["target_30d_return"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        dir_acc = np.mean(np.sign(y_pred) == np.sign(y_test)) * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score", f"{r2:.3f}")
        m2.metric("MAE", f"{mae:.4f}")
        m3.metric("Directional Accuracy", f"{dir_acc:.1f}%")

        fig3, axes = dark_fig2()
        axes[0].scatter(y_test, y_pred, alpha=0.3, s=5, color="#378ADD")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        axes[0].plot(lims, lims, color="#E24B4A", linewidth=1.5, linestyle="--")
        axes[0].set_xlabel("Actual", color="white")
        axes[0].set_ylabel("Predicted", color="white")
        axes[0].set_title(f"Predicted vs Actual (R²={r2:.3f})", color="white")

        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
        bar_colors = ["#E24B4A" if i >= len(importances) - 3 else "#378ADD"
                      for i in range(len(importances))]
        axes[1].barh(importances.index, importances.values, color=bar_colors)
        axes[1].set_title("Feature Importance", color="white")
        st.pyplot(fig3)

    # ── Backtest ──────────────────────────────────────────────
    if show_backtest:
        backtest_path = Path("outputs/backtest_results.csv")
        if backtest_path.exists():
            st.divider()
            st.markdown("### Backtest Results")
            bt = pd.read_csv(backtest_path)
            win_rate = (bt["avg_actual_return"] > 0).mean() * 100
            avg_ret = bt["avg_actual_return"].mean()

            b1, b2, b3 = st.columns(3)
            b1.metric("Win Rate", f"{win_rate:.1f}%")
            b2.metric("Avg Return / Period", f"{avg_ret:.2f}%")
            b3.metric("Periods Tested", len(bt))

            bt["cumulative"] = (1 + bt["avg_actual_return"] / 100).cumprod() * 100 - 100
            fig4, ax4 = dark_fig(figsize=(12, 3))
            ax4.plot(range(len(bt)), bt["cumulative"], color="#4CAF50", linewidth=2)
            ax4.fill_between(range(len(bt)), bt["cumulative"], alpha=0.2, color="#4CAF50")
            ax4.axhline(0, color="white", linewidth=0.5, linestyle="--")
            ax4.set_title("Cumulative Backtest Return (%)", color="white")
            st.pyplot(fig4)

            st.dataframe(bt, use_container_width=True, hide_index=True)
        else:
            st.info("Run backtest.py first to see backtest results here.")

else:
    st.info("Select tickers from the sidebar and click Run Screener to get started.")
    if Path("outputs/model_evaluation.png").exists():
        st.image("outputs/model_evaluation.png", caption="Last model evaluation run")
