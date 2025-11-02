import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Folder paths
DATA_FOLDER = "data"
OUTPUT_FOLDER = "data/processed"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Parameters
WINDOW_SIZE = 30  # use last 30 days to predict next day

# Load and process each dataset
for file in os.listdir(DATA_FOLDER):
    if not file.endswith(".csv"):
        continue

    stock_name = file.replace("_data.csv", "")
    print(f"Processing {stock_name}...")

    df = pd.read_csv(os.path.join(DATA_FOLDER, file))
    df.dropna(subset=["LogReturn"], inplace=True)
    
    # Scale the log returns
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(df[["LogReturn"]])

    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled_returns)):
        X.append(scaled_returns[i - WINDOW_SIZE:i])
        y.append(scaled_returns[i])

    X, y = np.array(X), np.array(y)
    
    np.savez_compressed(f"{OUTPUT_FOLDER}/{stock_name}_seq.npz", X=X, y=y)
    print(f"✅ Saved processed file: {OUTPUT_FOLDER}/{stock_name}_seq.npz")

print("All datasets processed successfully.")
