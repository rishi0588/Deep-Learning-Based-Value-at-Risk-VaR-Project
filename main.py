import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# ====================== CONFIG ======================
RESULTS_FOLDER = "results"
st.set_page_config(page_title="Deep Learning VaR Dashboard", layout="wide")
st.markdown(
    """
    <style>
        .main-title {
            font-size:36px !important;
            font-weight:700;
            color:#003366;
            text-align:center;
        }
        .sub-text {
            font-size:18px !important;
            color:#444;
            text-align:center;
            margin-bottom:20px;
        }
        .metric-card {
            background-color:#f0f2f6;
            border-radius:12px;
            padding:15px;
            text-align:center;
        }
        .metric-title {
            color:#444;
            font-size:14px;
        }
        .metric-value {
            font-size:20px;
            font-weight:600;
            color:#003366;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ====================== HEADER ======================
st.markdown('<div class="main-title">📊 Deep Learning–Based Value-at-Risk Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Interactive visualization of model performance, risk coverage, and VaR calibration</div>', unsafe_allow_html=True)

# ====================== LOAD DATA ======================
@st.cache_data
def load_summary():
    path = os.path.join(RESULTS_FOLDER, "var_summary.csv")
    if not os.path.exists(path):
        st.error("❌ var_summary.csv not found. Please run var_backtest.py first.")
        return None
    df = pd.read_csv(path)
    return df

df = load_summary()
if df is None:
    st.stop()

# Add Market column for grouping
india = ["Reliance", "Infosys"]
us = ["Apple", "Tesla"]
df["Market"] = df["Stock"].apply(lambda x: "India" if x in india else "US")

stocks = df["Stock"].unique()
models = df["Model"].unique()

# ====================== SIDEBAR ======================
st.sidebar.header("🔎 Controls")
selected_stock = st.sidebar.selectbox("Select Stock", options=stocks)
selected_model = st.sidebar.selectbox("Select Model", options=models)
st.sidebar.info("Use these filters to view model-specific risk metrics and charts.")

# ====================== TABS ======================
tab1, tab2, tab3 = st.tabs(["📈 Model Insights", "🌍 Market Comparison", "📄 About"])

# ====================== TAB 1: MODEL INSIGHTS ======================
with tab1:
    filtered = df[(df["Stock"] == selected_stock) & (df["Model"] == selected_model)]
    if filtered.empty:
        st.warning("No data found for this combination.")
        st.stop()

    st.subheader(f"Performance Overview — {selected_stock} ({selected_model})")

    val = filtered.iloc[0]
    cols = st.columns(4)
    metric_data = {
        "VaR95": val["VaR95"],
        "VaR99": val["VaR99"],
        "ViolRate95": val["ViolRate95"],
        "Kupiec_p95": val["Kupiec_p95"],
    }
    for i, (label, v) in enumerate(metric_data.items()):
        bg = "#c7f9cc" if "p" in label and v > 0.05 else "#fef9c3" if "ViolRate" in label else "#cce5ff"
        display_val = "N/A" if pd.isna(v) else f"{v:.4f}"
        with cols[i]:
            st.markdown(f"""
                <div class="metric-card" style="background-color:{bg}">
                    <div class="metric-title">{label}</div>
                    <div class="metric-value">{display_val}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Load and display plots (your generated PNGs)
    st.subheader("Prediction and VaR Visualizations")
    plot_files = sorted(glob.glob(os.path.join(RESULTS_FOLDER, f"{selected_stock}_{selected_model}_VaR_plot.png")))
    if plot_files:
        for img_path in plot_files:
            st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
    else:
        st.info("No saved plots available for this stock/model combination.")

# ====================== TAB 2: MARKET COMPARISON ======================
with tab2:
    st.subheader("Market-Level Comparison")

    market_summary = (
        df.groupby("Market")[["ViolRate95", "ViolRate99", "Kupiec_p95"]]
        .mean(numeric_only=True)
        .reset_index()
    )

    # Remove invalid Kupiec values
    market_summary = market_summary.replace([0, float("inf"), -float("inf")], None)
    market_summary = market_summary.dropna(how="all", subset=["Kupiec_p95", "ViolRate95", "ViolRate99"])

    if market_summary.empty:
        st.info("No valid market-level metrics available for visualization.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Average Violation Rate (95%) by Market**")
            fig, ax = plt.subplots(figsize=(5,3))
            sns.barplot(data=market_summary, x="Market", y="ViolRate95", palette="Blues", ax=ax)
            ax.set_ylabel("Average ViolRate95")
            for container in ax.containers:
                ax.bar_label(container, fmt="%.3f", label_type="edge", fontsize=9)
            st.pyplot(fig, use_container_width=True)

        with c2:
            st.markdown("**Average Kupiec p-value by Market**")

            kupiec_data = market_summary.copy()
            kupiec_data["Kupiec_p95"] = kupiec_data["Kupiec_p95"].replace([float("inf"), -float("inf")], 0)
    
            if (kupiec_data["Kupiec_p95"].fillna(0) == 0).all():
                st.warning("All Kupiec p-values are zero — using small reference values for visualization only.")
                kupiec_data["Kupiec_p95"] = [0.005, 0.008]  

                fig, ax = plt.subplots(figsize=(5,3))
                sns.barplot(data=kupiec_data, x="Market", y="Kupiec_p95", palette="Greens", ax=ax)
                ax.set_ylabel("Average Kupiec p95")
                ax.set_ylim(0, 0.05)
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.3f", label_type="edge", fontsize=9)
                st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    # Load the main summary visualizations (the seaborn ones you generated)
    st.subheader("Model Comparison Across Stocks")
    img_paths = [
        "VaR95_violation_comparison_advanced.png",
        "VaR99_violation_comparison_advanced.png"
    ]
    for img in img_paths:
        img_path = os.path.join(RESULTS_FOLDER, img)
        if os.path.exists(img_path):
            st.image(img_path, caption=img.replace("_", " ").replace(".png", ""), use_container_width=True)

    st.markdown("**Key Observations:**")
    st.markdown("""
    - Indian and US markets exhibit comparable violation rates, indicating similar model calibration.  
    - Kupiec p-values remain low overall, highlighting potential underestimation of tail risk.  
    - CNN and LSTM architectures often show slightly better tail capture compared to MLP.  
    """)

# ====================== TAB 3: ABOUT ======================
with tab3:
    st.markdown("### Project Overview")
    st.write("""
    This dashboard presents analytical results from deep learning models trained to estimate daily Value-at-Risk (VaR)
    for selected Indian and US stocks. Models include advanced MLP, CNN1D, BiLSTM, and Transformer architectures.
    """)
    st.markdown("**Key Metrics:**")
    st.markdown("""
    - **VaR95 / VaR99:** Predicted one-day downside risk thresholds at 95% and 99% confidence levels.  
    - **Violation Rate:** Actual proportion of returns below the VaR threshold.  
    - **Kupiec p-value:** Statistical validation of model calibration for VaR coverage.  
    """)
