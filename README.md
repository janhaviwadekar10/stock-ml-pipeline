**Smart Stock Screener**

This project implements a stock screening pipeline that predicts 30-day equity returns using Python, yfinance, and scikit-learn. It engineers technical indicators (moving averages, volatility), trains a Random Forest model, ranks tickers based on predicted performance, and generates visualizations to support data-driven analysis.

**What this repo does:**
* pulls stock data using yfinance
* builds features (moving averages, daily returns, volatility)
* trains a Random Forest regressor for 30-day return
* scores each ticker on the latest date
* builds quick plots (PCA, t-SNE, clusters, heatmap)

**Structure:**
* "main_pipeline.py" - download data, build features, train model
* "predict_and_infer.py" - load model, score everything, exporting top picks
* "visualize_and_analyze.py" - generate plots and clustering diagnostics

**Install deps:**

pip install -r requirements.txt

**Run:**
* python main_pipeline.py
* python predict_and_infer.py
* python visualize_and_analyze.py

**Next steps (planned)**
* simple backtests (did these predictions actually pan out)
* auto-run cron or airflow
* small web dashboard (streamlit or flask)
