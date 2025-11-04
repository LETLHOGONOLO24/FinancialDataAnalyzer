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
            
            time_series = data.get('Time Series (Daily)', {})
            if not time_series:
                print("No Alpha Vantage data found.")
                return None
            
            # Time to convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df = df.astype(float)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Limit to requested period
            cutoff_date = df.index.max() - pd.DateOffset(months=months)
            df = df[df.index >= cutoff_date]

            print(f"✔️ Alpha Vantage: {len(df)} days of data")
            return df
        
        except Exception as e:
            print(f"Alpha Vantage fetch failed: {e}")
            return None
        
        def fetch_yahoo_finance_data(self, symbol, months=6):
            """
            Fetch data from Yahoo Finance API
            """
            print(f"Fetching {symbol} data from Yahoo Finance...")

            try:
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date() - timedelta(days=months*30)

                # Download data
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)

                if df.empty:
                    print("No Yahoo Finance data found.")
                    return None
                
                # Rename columns to match our format
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = ['open', 'high', 'low', 'close', 'volume']

                print(f"✔️ Yahoo Finance: {len(df)} days of data")
                return df
            
            except Exception as e:
                print(f"Yahoo Finance fetch failed: {e}")
                return None
            
            def compare_data_sources(self, df_av, df_yf, symbol):
                """
                Compare data from both sources and identify discrepancies
                """
                print(f"\n{'='*60}")
                print("DATA SOURCE COMPARISON")
                print(f"{'='*60}")

                if df_av is None or df_yf is None:
                    print("Cannot compare - one or both data sources failed.")
                    return None
                
                # Find common dates
                common_dates = df_av.index.intersection(df_yf.index)

                if len(common_dates) == 0:
                    print("No common dates between data sources.")
                    return None
                
                print(f"Common trading days: {len(common_dates)}")

                # Compare closing prices
                av_prices = df_av.loc[common_dates, 'close']
                yf_prices = df_yf.loc[common_dates, 'close']

                # Calculate differences
                price_differences = abs(av_prices - yf_prices)
                avg_differences = price_differences.mean()
                max_differences = price_differences.max()

                print(f"Average price difference: ${avg_difference:.4f}")
                print(f"Maximum price difference: ${max_difference:.4f}")

                # Show dates with largest discrepancies
                if max_difference > 0.01: # Only show if there are meaningful differences
                    top_discrepancies = price_difference.nlargest(3)
                    print("\nLargest discrepancies:")

                    for date, diff in top_discrepancies.items():
                        av_price = df_av.loc[date, 'close']
                        yf_price = df_yf.loc[date, 'close']
                        print(f" {date.date()}: AlphaVantage=${av_price:.2f}, Yahoo=${yf_price:.2f}, Diff=${diff:.2f}")

                return common_dates

