import pandas as pd

def get_sp500_tickers():
    """
    Scrapes the current list of S&P 500 tickers from Wikipedia.
    Replaces '.' with '-' for yfinance compatibility (e.g., BRK.B → BRK-B).
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    tickers = table[0]['Symbol'].tolist()
    return [ticker.replace('.', '-') for ticker in tickers]
