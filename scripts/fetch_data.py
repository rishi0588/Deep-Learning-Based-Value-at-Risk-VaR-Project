import yfinance as yf
import pandas as pd
import numpy as np
import os

# Folder to save the data
os.makedirs("data", exist_ok=True)

# Define tickers and names
stocks = {
    "RELIANCE.NS": "Reliance",
    "INFY.NS": "Infosys",
    "AAPL": "Apple",
    "TSLA": "Tesla"
}

# Fetch and preprocess each stock
for ticker, name in stocks.items():
    print(f"Fetching data for {name} ({ticker})...")
    df = yf.download(ticker, start="2000-01-01", end="2025-06-30")
    
    # Calculate daily log returns
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    
    # Save cleaned data
    df.to_csv(f"data/{name}_data.csv")
    print(f"✅ Saved: data/{name}_data.csv")

print("All stock data fetched and saved successfully.")
