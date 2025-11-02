## Neural Network–Based Value-at-Risk (VaR) Estimation ## 

This project automates the training, evaluation, and reporting of deep learning–based Value-at-Risk (VaR) models using multiple neural architectures — MLP, CNN-1D, LSTM, and Transformer — across both Indian and US markets.

It provides an end-to-end pipeline for model training, VaR estimation, Kupiec backtesting, automated report generation, and a Streamlit dashboard for interactive visualization.

## Features

Compare MLP, CNN-1D, LSTM, and Transformer architectures

Compute VaR at 95% and 99% confidence levels

Perform Kupiec backtesting for model calibration

Generate Markdown and Word reports automatically

Visualize results in a Streamlit dashboard

Scalable modular structure for easy extension

## Installation
Clone the Repository
git clone https://github.com/rishi0588/Deep-Learning-Based-Value-at-Risk-VaR-Project
cd Deep-Learning-Based-Value-at-Risk-VaR-Project

## Install Dependencies
pip install -r requirements.txt

# Step 1 — Train Models

Train all deep learning architectures for each stock:

python scripts/train_models.py

# Step 2 — Run Backtesting

Run Kupiec tests and compile VaR metrics:

python scripts/var_backtest.py

# Step 3 — Generate Report

Automatically create Word and Markdown reports with results and insights:

python scripts/generate_analysis_report.py

# Step 4 — Launch Dashboard

Run the Streamlit app to visualize metrics interactively:

streamlit run main.py

##  Outputs
File	Description
results/var_summary.csv	Consolidated backtest metrics
results/final_report.docx	Formatted report ready for submission
results/final_report.md	Markdown version of the report
results/*.png	VaR plots and comparison charts

## Technologies Used
Library	Purpose
TensorFlow / Keras	Deep learning model training
Streamlit	Interactive dashboard
Pandas / NumPy / SciPy	Data handling, statistical analysis
Matplotlib / Seaborn	Visualization
python-docx	Report generation (Word)

## Key Insights
US Market Stocks (Apple, Tesla) → Lower VaR violations, higher predictability

Indian Market Stocks (Reliance, Infosys) → Higher volatility, frequent breaches

Deep learning models capture general return patterns but struggle with extreme tail-risk events

## Key Takeaways
Kupiec p-values = 0.0 across all models — poor statistical calibration.

US markets are more predictable and model-friendly.

Indian markets are volatile and prone to risk underestimation.

LSTM and Transformer generalize better but require recalibration.

Future work: integrate Expected Shortfall (ES) and rolling-window retraining.

## Model Summary
Model	Strength	Limitation
MLP	Simple and fast	Fails at capturing temporal patterns
CNN-1D	Detects local trends	Sensitive to noise
LSTM	Handles sequence memory	Overfits small datasets
Transformer	Best generalization	High computational cost

## Example Workflow
# Step 1: Train all models
python scripts/train_models.py

# Step 2: Compute VaR and backtest
python scripts/var_backtest.py

# Step 3: Generate analysis report
python scripts/generate_analysis_report.py

# Step 4: View results interactively
streamlit run main.py

## Author
Rishi Ponda
MBA(Tech) — Data Science, MPSTME, NMIMS

## Notes
Works best with daily OHLCV data (Apple, Tesla, Infosys, Reliance).

All scripts are modular — you can retrain, backtest, or regenerate reports independently.

Ideal for financial risk research, deep learning experimentation, and academic submission.

# Clone, Run, and Explore Deep Learning–Driven Financial Risk Analysis!
