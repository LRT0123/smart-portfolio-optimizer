import numpy as np
import matplotlib.pyplot as plt

def run_monte_carlo_simulation(portfolio_returns, start_value=10000, days=252, num_simulations=500):
    daily_return = portfolio_returns.mean()
    daily_std = portfolio_returns.std()

    simulations = []

    for _ in range(num_simulations):
        prices = [start_value]
        for _ in range(days):
            shock = np.random.normal(loc=daily_return, scale=daily_std)
            prices.append(prices[-1] * (1 + shock))
        simulations.append(prices)

    return np.array(simulations)

def plot_monte_carlo_simulation(simulations):
    fig, ax = plt.subplots(figsize=(8, 6))
    for sim in simulations:
        ax.plot(sim, alpha=0.1, color='purple')
    ax.set_title("Monte Carlo Simulation - Portfolio Value")
    ax.set_xlabel("Days")
    ax.set_ylabel("Portfolio Value")
    return fig
