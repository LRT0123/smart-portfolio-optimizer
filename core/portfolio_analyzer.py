import pandas as pd
import numpy as np

def calculate_daily_returns(price_df):
    """
    Calculates daily returns of a price dataframe.
    """
    return price_df.pct_change().dropna()

def equal_weighted_portfolio(returns_df):
    """
    Assumes equal weight allocation for all assets.
    Returns portfolio returns as a Series.
    """
    weights = np.array([1 / returns_df.shape[1]] * returns_df.shape[1])
    portfolio_returns = returns_df.dot(weights)
    return portfolio_returns

def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.02):
    """
    Calculates annualized Sharpe Ratio for portfolio returns.
    Assumes daily returns input.
    """
    excess_returns = portfolio_returns - (risk_free_rate / 252)
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    return sharpe_ratio

import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(returns_df):
    """
    Plots and returns a correlation heatmap.
    """
    corr = returns_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    return fig
def plot_allocation_pie(weights, tickers):
    import plotly.express as px
    df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": weights
    })
    fig = px.pie(df, values='Weight', names='Ticker', title='Portfolio Allocation')
    return fig
