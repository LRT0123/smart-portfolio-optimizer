import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def simulate_random_portfolios(returns, num_portfolios=5000, risk_free_rate=0.02):
    num_assets = returns.shape[1]
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    results = {
        "Returns": [],
        "Volatility": [],
        "Sharpe": [],
        "Weights": []
    }

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return - risk_free_rate) / port_volatility

        results["Returns"].append(port_return)
        results["Volatility"].append(port_volatility)
        results["Sharpe"].append(sharpe_ratio)
        results["Weights"].append(weights)

    return pd.DataFrame(results)

def plot_efficient_frontier(df, optimal_weights, returns):
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    opt_return = np.dot(optimal_weights, mean_returns)
    opt_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df["Volatility"], df["Returns"], c=df["Sharpe"], cmap="viridis", alpha=0.5)
    ax.scatter(opt_vol, opt_return, c="red", s=100, marker="*", label="Optimal Portfolio")
    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    ax.legend()
    fig.colorbar(scatter, label="Sharpe Ratio")
    return fig
