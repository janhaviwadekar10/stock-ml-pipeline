import os
from pathlib import Path
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# data prep
def downloaddata(tickers, start_date, end_date, output_folder="data"):
    
    # here we are pulling stock data from yfinance, build a few basic features, and save one CSV per ticker.
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        print(f"[data] pulling {ticker} from {start_date} → {end_date}...")
        
        # get the raw price data
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

        # if yahoo gives us nothing, just skip and keep moving
        if df.empty:
            print(f"[data] nothing came back for {ticker}, skipping.")
            continue

        # some tickers have "Adj Close", some don't so we are just using what exists
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

        # daily percent change
        df["daily_return"] = df[price_col].pct_change()
        
        # short / medium / longer moving averages
        df["ma_7"] = df[price_col].rolling(7).mean()
        df["ma_30"] = df[price_col].rolling(30).mean()
        df["ma_90"] = df[price_col].rolling(90).mean()
        
        # rolling volatility over ~1 month (trading days)
        df["volatility_30"] = df["daily_return"].rolling(30).std()
        
        # what we’re trying to predict: return 30 days ahead
        df["target_30d_return"] = df[price_col].shift(-30) / df[price_col] - 1

        # drop the NA rows created by rolling windows / shifting
        df = df.dropna().reset_index()

        # keep track of which row belongs to which ticker
        df["ticker"] = ticker

        # save per-ticker features
        out_path = out_dir / f"{ticker}_features.csv"
        df.to_csv(out_path, index=False)
        print(f"[data] saved features → {out_path}")

# combine all the per-ticker files
def mergefeatures(folder="data"):
    
    # here we are just pulling all those per-ticker feature csv's and merging them into one big combined file
    # and then return the dataframe version of that combined file so we can use it straight away
    folder_path = Path(folder)
    dataframes = []

    # walk the folder and pick up only our feature files
    for fname in os.listdir(folder_path):
        if fname.endswith("_features.csv"):
            df = pd.read_csv(folder_path / fname)
            
            # now, after reset_index we should have date in here
            if "Date" in df.columns:
                dataframes.append(df)

    if not dataframes:
        print("[merge] no feature CSVs found. did you run download_and_prepare_data()?")
        return None

    combined = pd.concat(dataframes, ignore_index=True)
    out_file = folder_path / "combined_features.csv"
    combined.to_csv(out_file, index=False)
    print(f"[merge] wrote combined file → {out_file}")
    return combined

# train + save the model
def trainmodel():
    
    #train a simple random forest on the engineered features and stash the model on disk for later use.
    combined_path = Path("data") / "combined_features.csv"
    if not combined_path.exists():
        print("[train] combined_features.csv not found. run merge_all_features() first.")
        return

    # load the data and drop any leftover gaps
    df = pd.read_csv(combined_path).dropna()

    # features we plan to feed into the model and it's fine if some have "Close vs Adj Close", 
    # pandas will just use whichever is present.
    feature_cols = [
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
        "daily_return",
        "ma_7",
        "ma_30",
        "ma_90",
        "volatility_30",
    ]

    # keep only the columns that are actually present (makes this more forgiving)
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["target_30d_return"]

    # standard 80/20 split, random_state just makes the split repeatable
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # random forest is a decent baseline that handles nonlinearity without much fuss
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # save the trained model so we can load it elsewhere
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "random_forest_model.joblib"
    joblib.dump(model, model_path)

    print(f"[train] model trained and saved → {model_path}")


if __name__ == "__main__":
    # a small basket of big names, edit as you like
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA"]

    # 1) pull data + make features
    downloaddata(tickers, "2022-01-01", "2025-07-18")

    # 2) stitch everything into one file
    mergefeatures()

    # 3) train and save the model
    trainmodel()