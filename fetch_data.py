import yfinance as yf
import pandas as pd
import os

def fetch_prices():
    print("Fetching copper prices...")
    copper = yf.download("HG=F", start="2020-01-01", auto_adjust=True)["Close"]

    print("Fetching USD/INR rate...")
    usdinr = yf.download("INR=X", start="2020-01-01", auto_adjust=True)["Close"]

    # Zinc still needed as model feature — but won't show on dashboard
    print("Fetching zinc prices...")
    zinc = yf.download("ZNC=F", start="2020-01-01", auto_adjust=True)["Close"]

    df = pd.DataFrame({
        "copper": copper.squeeze() * 2204,   # convert to USD/tonne
        "zinc":   zinc.squeeze(),
        "usdinr": usdinr.squeeze()
    }).dropna()

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/prices.csv")
    print(f"✅ Saved {len(df)} rows to data/prices.csv")
    return df

if __name__ == "__main__":
    df = fetch_prices()
    print(df.tail())