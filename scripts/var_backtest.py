import numpy as np
import os
from scipy.stats import chi2
import pandas as pd

RESULTS_FOLDER = "results"
SUMMARY_FILE = os.path.join(RESULTS_FOLDER, "var_summary.csv")


def kupiec_test(violations, alpha, n):
    if violations == 0 or violations == n:
        return 0, 0
    p_hat = max(min(violations / n, 0.9999), 0.0001)
    LR = -2 * np.log(((1 - alpha) ** (n - violations) * (alpha ** violations)) /
                     ((1 - p_hat) ** (n - violations) * (p_hat ** violations)))
    p_value = 1 - chi2.cdf(LR, 1)
    if np.isnan(p_value) or np.isinf(p_value):
        p_value = 0
    return LR, p_value


summary = []

for file in os.listdir(RESULTS_FOLDER):
    if not file.endswith("_preds.npz"):
        continue

    stock_model = file.replace("_preds.npz", "")
    stock, model = stock_model.split("_", 1)
    data = np.load(os.path.join(RESULTS_FOLDER, file))

    y_test = data["y_test"].flatten()
    preds = data["preds"].flatten()

    # Compute empirical VaR from predicted *distribution*
    VaR95 = np.percentile(preds, 5)
    VaR99 = np.percentile(preds, 1)

    # Compare actual returns vs VaR
    violations95 = np.sum(y_test < VaR95)
    violations99 = np.sum(y_test < VaR99)
    n = len(y_test)

    viol_rate95 = violations95 / n
    viol_rate99 = violations99 / n

    LR95, p95 = kupiec_test(violations95, 0.05, n)
    LR99, p99 = kupiec_test(violations99, 0.01, n)

    summary.append({
        "Stock": stock,
        "Model": model,
        "VaR95": VaR95,
        "VaR99": VaR99,
        "ViolRate95": viol_rate95,
        "ViolRate99": viol_rate99,
        "Kupiec_LR95": LR95,
        "Kupiec_p95": p95,
        "Kupiec_LR99": LR99,
        "Kupiec_p99": p99
    })

df = pd.DataFrame(summary)
df.to_csv(SUMMARY_FILE, index=False)

print(f"✅ VaR summary saved to {SUMMARY_FILE}")
print(df)
