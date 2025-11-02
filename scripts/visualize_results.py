import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_FOLDER = "results"
SUMMARY_FILE = os.path.join(RESULTS_FOLDER, "var_summary.csv")

sns.set_style("whitegrid")

# --- 1️⃣ Plot actual vs predicted returns with VaR lines ---
def plot_var(stock, model):
    data_path = f"{RESULTS_FOLDER}/{stock}_{model}_preds.npz"
    if not os.path.exists(data_path):
        print(f"⚠️ Missing file for {stock}-{model}")
        return

    data = np.load(data_path)
    y_test, preds = data["y_test"].flatten(), data["preds"].flatten()

    VaR95 = np.percentile(preds, 5)
    VaR99 = np.percentile(preds, 1)

    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label="Actual", alpha=0.8)
    plt.plot(preds, label="Predicted", alpha=0.7)
    plt.axhline(VaR95, color='r', linestyle='--', label=f"VaR95 ({VaR95:.4f})")
    plt.axhline(VaR99, color='m', linestyle='--', label=f"VaR99 ({VaR99:.4f})")
    plt.title(f"{stock} - {model} | Actual vs Predicted Returns with VaR Lines")
    plt.xlabel("Days")
    plt.ylabel("Log Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/{stock}_{model}_VaR_plot.png")
    plt.close()
    print(f"✅ Saved plot: {RESULTS_FOLDER}/{stock}_{model}_VaR_plot.png")


# --- 2️⃣ Summary comparison charts (basic bar version) ---
def summary_visuals_basic(df):
    plt.figure(figsize=(12, 6))
    for stock in df["Stock"].unique():
        subset = df[df["Stock"] == stock]
        plt.bar(subset["Model"], subset["ViolRate95"], alpha=0.6, label=stock)
    plt.title("VaR95 Violation Rate Comparison (All Stocks)")
    plt.ylabel("Violation Rate (95%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/VaR95_violation_comparison_basic.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    for stock in df["Stock"].unique():
        subset = df[df["Stock"] == stock]
        plt.bar(subset["Model"], subset["ViolRate99"], alpha=0.6, label=stock)
    plt.title("VaR99 Violation Rate Comparison (All Stocks)")
    plt.ylabel("Violation Rate (99%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/VaR99_violation_comparison_basic.png")
    plt.close()
    print("✅ Basic summary charts saved.")


# --- 3️⃣ Seaborn: Combined barplots across all models/stocks ---
def summary_visuals_advanced(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Stock", y="ViolRate95", hue="Model", palette="Blues")
    plt.title("VaR 95% Violation Rate Comparison Across Models and Stocks")
    plt.ylabel("Violation Rate (95%)")
    plt.xlabel("Stock")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/VaR95_violation_comparison_advanced.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Stock", y="ViolRate99", hue="Model", palette="Reds")
    plt.title("VaR 99% Violation Rate Comparison Across Models and Stocks")
    plt.ylabel("Violation Rate (99%)")
    plt.xlabel("Stock")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/VaR99_violation_comparison_advanced.png")
    plt.close()
    print("✅ Advanced summary charts saved.")


# --- 4️⃣ Seaborn: Per-stock focused comparisons ---
def summary_visuals_per_stock(df):
    stocks = df["Stock"].unique()
    for s in stocks:
        temp = df[df["Stock"] == s]

        plt.figure(figsize=(7, 5))
        sns.barplot(data=temp, x="Model", y="ViolRate95", palette="Blues")
        plt.title(f"VaR 95% Violation Rates — {s}")
        plt.ylabel("Violation Rate (95%)")
        plt.tight_layout()
        plt.savefig(f"{RESULTS_FOLDER}/{s}_VaR95_violation_per_stock.png")
        plt.close()

        plt.figure(figsize=(7, 5))
        sns.barplot(data=temp, x="Model", y="ViolRate99", palette="Reds")
        plt.title(f"VaR 99% Violation Rates — {s}")
        plt.ylabel("Violation Rate (99%)")
        plt.tight_layout()
        plt.savefig(f"{RESULTS_FOLDER}/{s}_VaR99_violation_per_stock.png")
        plt.close()

    print("✅ Per-stock Seaborn charts saved.")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    df = pd.read_csv(SUMMARY_FILE)

    # Plot VaR lines for each stock-model
    for _, row in df.iterrows():
        plot_var(row["Stock"], row["Model"])

    # Create comparison summaries
    summary_visuals_basic(df)
    summary_visuals_advanced(df)
    summary_visuals_per_stock(df)

    print("\n🎨 All visualizations generated successfully!")
