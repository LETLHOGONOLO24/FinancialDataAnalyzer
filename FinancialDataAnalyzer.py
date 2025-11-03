import requests
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import time
import numpy as np

class MultiSourceFinancialAnalyzer:
    def __init__(self, alpha_vantage_key=None):
        self.alpha_vantage_key = alpha_vantage_key
        self.alpha_vantage_url = "https://www.alphavantage.co/query"

    def fetch_alpha_vantage_data(self, symbol, months=6):
        """
        Fetch data from Alpha Vantage API
        """
        if not self.alpha_vantage_key:
            print("Alpha Vantage API key not provided. Skipping Alpha Vantage...")
            return None
        
        print(f"Fetching {symbol} data from Alpha Vantage...")

        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.alpha_vantage_key,
            'outputsize': 'compact'
        }

        try:
            response = requests.get(self.alpha_vantage_url, params=params)
            data = response.json()

            if 'Error Message' in data:
                print(f"Alpha Vantage Error: {data['Error Message']}")
                return None
            if 'Note' in data:
                print("Alpha vantage rate limit reached. Waiting...")
                time.sleep(60)
                return self.fetch_alpha_vantage_data(symbol, months)
