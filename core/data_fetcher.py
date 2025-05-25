import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_price_data(tickers, start_date="2018-01-01", end_date=None):
    """
    Fetches adjusted close prices for a list of tickers using yfinance.
    
    Args:
        tickers (list): List of ticker symbols (e.g., ['AAPL', 'GOOGL']).
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format. Defaults to today.

    Returns:
        pd.DataFrame: DataFrame containing adjusted close prices.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # Explicitly set auto_adjust=False to preserve 'Adj Close'
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)

    # Handle single or multiple ticker case
    if "Adj Close" in data.columns:
        data = data["Adj Close"]
    elif "Close" in data.columns:
        data = data["Close"]
    else:
        raise ValueError("Expected 'Adj Close' or 'Close' in data")

    if isinstance(data, pd.Series):
        data = data.to_frame()

    return data.dropna()