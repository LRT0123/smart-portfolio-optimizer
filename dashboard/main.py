import numpy as np
from core.data_fetcher import fetch_price_data
from core.portfolio_analyzer import (
    calculate_daily_returns,
    equal_weighted_portfolio,
    calculate_sharpe_ratio
)

tickers = ["AAPL", "MSFT", "GOOGL"]
prices = fetch_price_data(tickers)
returns = calculate_daily_returns(prices)
portfolio_returns = equal_weighted_portfolio(returns)
sharpe = calculate_sharpe_ratio(portfolio_returns)

print("=== Portfolio Analysis ===")
print(f"Annualized Mean Return: {portfolio_returns.mean() * 252:.2%}")
print(f"Annualized Volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")