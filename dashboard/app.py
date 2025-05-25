from core.monte_carlo import run_monte_carlo_simulation, plot_monte_carlo_simulation
from core.efficient_frontier import simulate_random_portfolios, plot_efficient_frontier

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pandas as pd

from utils.ticker_loader import get_sp500_tickers
from core.data_fetcher import fetch_price_data
from core.portfolio_analyzer import (
    calculate_daily_returns,
    equal_weighted_portfolio,
    calculate_sharpe_ratio,
    plot_correlation_heatmap,
    plot_allocation_pie
)
from core.optimizer import optimize_portfolio


# === Page Configuration ===
st.set_page_config(page_title="Smart Portfolio Optimizer", layout="wide")
st.title("📊 Smart Portfolio Optimizer for Retail Investors")

# === Theme Toggle ===
theme_choice = st.sidebar.radio("🖌️ Choose Theme", options=["System Default", "Light", "Dark"])
if theme_choice == "Light":
    st.markdown("""
        <style>
            html, body, [data-testid="stAppViewContainer"] {
                background-color: #ffffff;
                color: #000000;
            }
        </style>
    """, unsafe_allow_html=True)
elif theme_choice == "Dark":
    st.markdown("""
        <style>
            html, body, [data-testid="stAppViewContainer"] {
                background-color: #111827;
                color: #F9FAFB;
            }
        </style>
    """, unsafe_allow_html=True)
# "System Default" uses Streamlit’s default or config.toml setting

# === Sidebar Inputs ===
st.sidebar.markdown(f"🖌️ Current Theme: **{theme_choice}**")

all_tickers = get_sp500_tickers()

tickers = st.sidebar.multiselect(
    "📂 Select up to 15 S&P 500 Tickers",
    all_tickers,
    default=["AAPL", "MSFT", "GOOGL"]
)

start_date = st.sidebar.date_input("📅 Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("📅 End Date", value=pd.to_datetime("today"))

if len(tickers) > 15:
    st.warning("⚠️ Too many tickers selected. Please limit to 15 or fewer.")
    st.stop()

# === Data Processing ===
prices = fetch_price_data(tickers, start_date=str(start_date), end_date=str(end_date))
returns = calculate_daily_returns(prices)
portfolio_returns = equal_weighted_portfolio(returns)
sharpe = calculate_sharpe_ratio(portfolio_returns)

# === Portfolio Metrics ===
st.markdown("---")
st.subheader("📈 Portfolio Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Annualized Return", f"{portfolio_returns.mean() * 252:.2%}")
col2.metric("Volatility", f"{portfolio_returns.std() * np.sqrt(252):.2%}")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

# === Visualizations ===
st.markdown("---")
st.subheader("📅 Price History")
st.line_chart(prices)

st.subheader("📊 Daily Returns")
st.line_chart(portfolio_returns)

st.subheader("📊 Correlation Heatmap")
fig_heatmap = plot_correlation_heatmap(returns)
st.pyplot(fig_heatmap)

# === Optimized Portfolio Allocation ===
st.markdown("---")
st.subheader("📌 Optimized Asset Allocation")
weights = optimize_portfolio(returns)
fig_alloc = plot_allocation_pie(weights, tickers)
st.plotly_chart(fig_alloc)

st.subheader("📋 Optimal Weights Table")
st.dataframe(pd.DataFrame({"Ticker": tickers, "Weight": weights}))

# === Download Portfolio ===
st.markdown("### 📥 Download Optimized Weights as CSV")

weights_df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
csv = weights_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="optimized_portfolio.csv",
    mime="text/csv"
)

# === Efficient Frontier ===
st.markdown("---")
st.subheader("📐 Efficient Frontier")

df_portfolios = simulate_random_portfolios(returns)
fig_frontier = plot_efficient_frontier(df_portfolios, weights, returns)
st.pyplot(fig_frontier)

# === Monte Carlo Simulation ===
st.markdown("---")
st.subheader("🔮 Monte Carlo Simulation (1-Year Forecast)")

simulations = run_monte_carlo_simulation(portfolio_returns)
fig_mc = plot_monte_carlo_simulation(simulations)
st.pyplot(fig_mc)
