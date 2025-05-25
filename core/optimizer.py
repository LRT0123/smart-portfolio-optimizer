import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(returns, risk_free_rate=0.02):
    """
    Optimize portfolio weights to maximize Sharpe Ratio.
    """
    num_assets = returns.shape[1]
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    def portfolio_performance(weights):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / vol
        return -sharpe  # negative for minimization

    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    result = minimize(portfolio_performance, initial_weights,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x  # optimal weights
