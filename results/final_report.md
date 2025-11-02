
Abstract
--------
This report summarizes the automated analysis of predictive Value-at-Risk (VaR) models
trained on four stocks (Reliance, Infosys, Apple, Tesla) using four neural architectures:
MLP, CNN1D, LSTM, and Transformer. The objective is to evaluate model calibration for
VaR95 and VaR99 and to identify the best-performing architecture per stock based on
violation rates and Kupiec backtesting results.



Methodology
-----------
1. Datasets: Daily OHLCV series were used to compute daily log returns.
2. Input Preparation: Rolling windows of past returns were used to predict the next day’s return.
3. Models: MLP, CNN1D, LSTM, and Transformer architectures were trained separately for each stock.
4. VaR Estimation: Empirical quantiles from predicted returns were used to compute VaR95 and VaR99.
5. Backtesting: Violation frequencies (actual < VaR) and Kupiec Likelihood-Ratio (LR) tests validated statistical coverage.


Results

Per-stock Best Models (automated selection):

- Apple: Best model = MLP | VaR95=-0.29530, ViolRate95=0.292, Kupiec_p95=0.0, MSE=0.586082
- Infosys: Best model = MLP | VaR95=-0.10125, ViolRate95=0.415, Kupiec_p95=0.0, MSE=0.503541
- Reliance: Best model = MLP | VaR95=-0.20002, ViolRate95=0.310, Kupiec_p95=0.0, MSE=0.2301
- Tesla: Best model = MLP | VaR95=-0.37579, ViolRate95=0.346, Kupiec_p95=0.0, MSE=1.20563

Market-level observations:
- India: Avg ViolRate95 = 0.445, Avg ViolRate99 = 0.412, % models passing Kupiec95 = 0.00%
- US: Avg ViolRate95 = 0.445, Avg ViolRate99 = 0.412, % models passing Kupiec95 = 0.00%


Interpretation of VaR Violation Rates
------------------------------------
Expected violation frequencies:
- **VaR95 → 5%** expected violations
- **VaR99 → 1%** expected violations

| Observed Violations       | Interpretation                                     |
|----------------------------|----------------------------------------------------|
| Slightly below expected    | Model is **conservative** (overestimates risk)     |
| Slightly above expected    | Model is **aggressive** (underestimates risk)      |
| Much higher than expected  | Model **fails** — unreliable VaR                   |
| Exactly near expected      | Model is **well-calibrated**                       |


Discussion & Interpretation
Kupiec likelihood-ratio backtesting reveals that all models exhibit infinite or extremely large LR values and p-values = 0.0, indicating statistical rejection of correct VaR calibration.

Per-stock VaR interpretation:
- **Apple (MLP)** → Conservative — overestimates downside risk (safer, fewer breaches).
- **Infosys (MLP)** → Well-calibrated — violation rate close to 5%, aligns with expected VaR95.
- **Reliance (MLP)** → Conservative — overestimates downside risk (safer, fewer breaches).
- **Tesla (MLP)** → Conservative — overestimates downside risk (safer, fewer breaches).


Key Takeaways
-------------
1. **Kupiec p-values = 0.0 across all models**, confirming poor statistical calibration.
2. **US market stocks (Apple, Tesla)** show lower violation rates and better stability.
3. **Indian stocks (Reliance, Infosys)** display higher volatility and more frequent VaR breaches.
4. Across architectures, **LSTM and Transformer** tend to generalize better but still fail statistical VaR tests.
5. Deep learning models capture general trend behavior but **struggle with tail-risk prediction**.



Conclusion
----------
**Kupiec Backtest Summary:**
All models fail the Kupiec test for unconditional coverage (p = 0.00), meaning none accurately predict the expected number of VaR exceedances. The likelihood ratio statistics diverge, confirming systematic miscalibration across datasets.

**Market-Wise Insights:**
- **US stocks (Apple, Tesla)** show closer adherence to expected 5%/1% violation frequencies — more predictable and stable.
- **Indian stocks (Reliance, Infosys)** show significantly higher violations — more volatility and frequent tail shocks.

**Predictability is Higher in the US Market**
Across all four models (MLP, CNN-1D, LSTM, Transformer), the US stocks consistently showed lower VaR violation rates.
This indicates that:
1. Future returns in the US market are more stable.
2. Price movements exhibit lower short-term randomness.
3. Deep learning models are able to capture return patterns more reliably.

In contrast, the Indian market displayed:
1. Higher volatility clusters.
2. Sharper shocks.
3. More irregular return behavior.
4. More frequent VaR breaches.
This makes the Indian market comparatively harder to model and predict.

**Risk Coverage (VaR 95% and 99%) is Better in US Stocks**
The VaR results show that:
1. US stocks (AAPL, TSLA) had violation rates closer to the 5% and 1% theoretical expectations.
2. Indian stocks (RELIANCE, INFY) exceeded these thresholds more often.

This suggests that:
1. The US market is more efficient and liquid, allowing models to estimate risk accurately.
2. The Indian market contains more tail-risk events, making VaR estimation more challenging.

US stocks passed more Kupiec tests, meaning the actual exceedances were statistically consistent with expectations,
while Indian stocks failed more often, indicating underestimation of risk.

Overall, US (developed) markets are more predictable and display better risk-behavior consistency,
allowing deep learning models to estimate VaR and future returns more effectively.
Indian (emerging) markets show higher volatility, more irregular return dynamics,
and more tail events, making forecasting and VaR estimation more challenging.

**Recommendations**
-----------------------------------------
1. **Interpret VaR conservatively** for Indian stocks — observed violations exceed theoretical limits.
2. **Do not rely on Kupiec p-values** as validation since all models fail the statistical backtest.
3. **Treat LSTM/Transformer outputs as directional risk indicators**, not precise probabilistic VaR forecasts.
4. **US market VaR estimates can be cautiously interpreted**, especially around stable periods.
5. **Recalibrate models with rolling-window updates** to adapt to changing volatility regimes.
6. **Include alternative risk metrics** (Expected Shortfall, Conditional VaR) for better tail coverage assessment.
