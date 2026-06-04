# 📈 Smart Stock Screener

ML-powered equity ranking pipeline that predicts 30-day returns, backtests strategy performance, and surfaces top picks via an interactive dashboard.

---

## Results

| Metric | Value |
|--------|-------|
| R² Score | 0.8595 |
| Directional Accuracy | 91.2% |
| Backtest Win Rate | 88.0% |
| Avg Return / 30-day Period | 12.67% |
| Best Period | +40.66% (TSLA, NVDA, META) |
| Periods Tested | 25 |

---

## Architecture

```
yfinance API → feature engineering → Random Forest model
                                           │
                              ┌────────────┼────────────┐
                         backtest      evaluate      predict
                              └────────────┼────────────┘
                                           │
                                  Streamlit dashboard
```

## What it does

Ingests historical price data for a basket of equities, engineers technical indicators as ML features, trains a Random Forest regressor to predict 30-day forward returns, and ranks tickers by predicted performance. A walk-forward backtest validates whether the predictions actually translated into real returns.

---

## Files

| File | What it does |
|------|-------------|
| `main_pipeline.py` | Download data, build features, train model |
| `predict_and_infer.py` | Load model, score tickers, export top picks |
| `evaluate_model.py` | R², MAE, directional accuracy, feature importance chart |
| `backtest.py` | Walk-forward backtest — did predictions make money? |
| `visualize_and_analyze.py` | PCA, t-SNE, KMeans clustering, correlation heatmap |
| `stock_dashboard.py` | Interactive Streamlit dashboard |

---

## Features engineered

- 7-day, 30-day, 90-day moving averages
- Daily return (pct change)
- 30-day rolling volatility
- Target: 30-day forward return

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. Download data + train model
python main_pipeline.py

# 2. Evaluate model performance
python evaluate_model.py

# 3. Run backtest
python backtest.py

# 4. Score latest tickers
python predict_and_infer.py

# 5. Launch dashboard
streamlit run stock_dashboard.py
```

---

## Tech stack

Python · scikit-learn · pandas · NumPy · yfinance · Matplotlib · Streamlit · Random Forest · Feature Engineering
