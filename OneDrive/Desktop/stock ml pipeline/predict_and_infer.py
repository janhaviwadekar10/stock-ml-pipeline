import pandas as pd
import joblib
import os

def main():
    # make sure outputs folder exists
    os.makedirs("outputs", exist_ok=True)

    # load trained model + load combined features back in
    model = joblib.load("models/random_forest_model.joblib")
    df = pd.read_csv("data/combined_features.csv").dropna()

    # pick the same feature columns we trained on
    feature_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume',
                    'daily_return', 'ma_7', 'ma_30', 'ma_90', 'volatility_30']
    X = df[feature_cols]

    # run model to get predicted returns
    df['predicted_return'] = model.predict(X)

    # use only the most recent date
    latest_date = df['Date'].max()
    df_filtered = df[df['Date'] == latest_date]

    # sort by predicted return and then pick top per ticker
    df_sorted = df_filtered.sort_values(by='predicted_return', ascending=False)
    top_per_ticker = df_sorted.groupby('ticker').head(1)
    top_10_diverse = top_per_ticker.sort_values(by='predicted_return', ascending=False).head(10)

    # save to csv
    top_10_diverse.to_csv("outputs/top_10_predictions.csv", index=False)

    print(f" Top 10 Stock Predictions for {latest_date}:")
    print(top_10_diverse[['ticker', 'Date', 'predicted_return']])

if __name__ == "__main__":
    main()