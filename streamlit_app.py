def show_empty_state():
    st.markdown(
        """
        <div style='text-align: center; margin-top: 50px;'>
            <h4 style='margin-top: 10px;'>Please select a stock ticker to get started!</h4>
            <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 480px; margin: auto;">
                <iframe src="https://giphy.com/embed/MDJ9IbxxvDUQM" 
                        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" 
                        frameborder="0" allowfullscreen></iframe>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# === SYSTEM PATH FIX ===
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === LIBRARIES ===
import streamlit as st
import numpy as np
import pandas as pd

# === LOCAL MODULES ===
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
from core.efficient_frontier import simulate_random_portfolios, plot_efficient_frontier
from core.monte_carlo import run_monte_carlo_simulation, plot_monte_carlo_simulation

# === PAGE CONFIG ===
st.set_page_config(page_title="Smart Portfolio Optimizer", layout="wide")
st.title("\U0001F4CA Smart Portfolio Optimizer for Retail Investors")

# === THEME TOGGLE ===
theme_choice = st.sidebar.radio("\U0001F5A8️ Choose Theme", options=["System Default", "Light", "Dark"])
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

st.sidebar.markdown(f"\U0001F5A8️ Current Theme: **{theme_choice}**")

# === SIDEBAR INPUTS ===
all_tickers = get_sp500_tickers()
tickers = st.sidebar.multiselect(
    "\U0001F4C2 Select up to 15 S&P 500 Tickers",
    all_tickers,
    help="Choose tickers from the S&P 500 list"
)

# Show fun empty state if no ticker is selected
if not tickers:
    show_empty_state()
    st.stop()
else:
    if "celebrated" not in st.session_state:
        st.session_state.celebrated = True
        st.balloons()

if len(tickers) > 15:
    st.warning("⚠️ Too many tickers selected. Please limit to 15 or fewer.")
    st.stop()

start_date = st.sidebar.date_input("\U0001F4C5 Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("\U0001F4C5 End Date", value=pd.to_datetime("today"))

# === DATA PROCESSING ===
prices = fetch_price_data(tickers, start_date=str(start_date), end_date=str(end_date))
returns = calculate_daily_returns(prices)
portfolio_returns = equal_weighted_portfolio(returns)
sharpe = calculate_sharpe_ratio(portfolio_returns)

# === PORTFOLIO METRICS ===
st.markdown("---")
st.subheader("\U0001F4C8 Portfolio Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Annualized Return", f"{portfolio_returns.mean() * 252:.2%}")
col2.metric("Volatility", f"{portfolio_returns.std() * np.sqrt(252):.2%}")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

# === VISUALIZATIONS ===
st.markdown("---")
st.subheader("\U0001F4C5 Price History")
st.line_chart(prices)

st.subheader("\U0001F4CA Daily Returns")
st.line_chart(portfolio_returns)

st.subheader("\U0001F4CA Correlation Heatmap")
fig_heatmap = plot_correlation_heatmap(returns)
st.pyplot(fig_heatmap)

# === OPTIMIZED PORTFOLIO ===
st.markdown("---")
st.subheader("\U0001F4CC Optimized Asset Allocation")
weights = optimize_portfolio(returns)
fig_alloc = plot_allocation_pie(weights, tickers)
st.plotly_chart(fig_alloc)

st.subheader("\U0001F4CB Optimal Weights Table")
weights_df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
st.dataframe(weights_df)

# === DOWNLOAD BUTTON ===
st.markdown("### \U0001F4E5 Download Optimized Weights as CSV")
csv = weights_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="optimized_portfolio.csv", mime="text/csv")

# === EFFICIENT FRONTIER ===
st.markdown("---")
st.subheader("\U0001F4A0 Efficient Frontier")
df_portfolios = simulate_random_portfolios(returns)
fig_frontier = plot_efficient_frontier(df_portfolios, weights, returns)
st.pyplot(fig_frontier)

# === MONTE CARLO SIMULATION ===
st.markdown("---")
st.subheader("\U0001F52E Monte Carlo Simulation (1-Year Forecast)")
simulations = run_monte_carlo_simulation(portfolio_returns)
fig_mc = plot_monte_carlo_simulation(simulations)
st.pyplot(fig_mc)
