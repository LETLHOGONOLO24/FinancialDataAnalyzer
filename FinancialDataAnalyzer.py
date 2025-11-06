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
            start_date = end_date - timedelta(days=months*30)
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                print("No Yahoo Finance data found.")
                return None
                
            # Rename columns to match our format
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            print(f"✓ Yahoo Finance: {len(df)} days of data")
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
        avg_difference = price_differences.mean()
        max_difference = price_differences.max()

        print(f"Average price difference: ${avg_difference:.4f}")
        print(f"Maximum price difference: ${max_difference:.4f}")

        # Show dates with largest discrepancies
        if max_difference > 0.01: # Only show if there are meaningful differences
            top_discrepancies = price_differences.nlargest(3)
            print("\nLargest discrepancies:")

            for date, diff in top_discrepancies.items():
                av_price = df_av.loc[date, 'close']
                yf_price = df_yf.loc[date, 'close']
                print(f" {date.date()}: AlphaVantage=${av_price:.2f}, Yahoo=${yf_price:.2f}, Diff=${diff:.2f}")

        return common_dates

    def merge_data_sources(self, df_av, df_yf):
        """
        Create a merged dataset, preferring one source over another or creating averages
        """
        if df_av is None and df_yf is None:
            print("Both data sources failed. Cannot merge.")
            return None
        elif df_av is None:
            print("Using Yahoo Finance data (Alpha Vantage failed)")
            return df_yf
        elif df_yf is None:
            print("Using Alpha Vantage data (Yahoo Finance failed)")
            return df_av
        else:
            # Both sources available - use Yahoo Finance as primary (more reliable)
            print("Using Yahoo Finance as primary data source")
            primary_df = df_yf.copy()

        return primary_df
                
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        df = df.copy()

        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # Daily returns and volatility
        df['daily_return'] = df['close'].pct_change() * 100
        df['volatility_20'] = df['daily_return'].rolling(window=20).std()

        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df
            
    def generate_signals(self, df):
        """Generate trading signals using multiple indicators"""
        df = df.copy()
        df['signal'] = 0
                
        # SMA Crossover strategy
        sma_signal = np.where(df['sma_20'] > df['sma_50'], 1, -1)
                
        # RSI-based signals (oversold/overbought)
        rsi_signal = np.where(df['rsi'] < 30, 1, np.where(df['rsi'] > 70, -1, 0))
                
        # Combined signal (simple logic - can be enhanced)
        df['signal'] = np.where((sma_signal == 1) & (rsi_signal == 1), 2,  # Strong buy
                       np.where((sma_signal == -1) & (rsi_signal == -1), -2, # Strong sell
                       np.where(sma_signal == 1, 1, # Weak buy
                       np.where(sma_signal == -1, -1, 0)))) # Weak sell
                
        return df
    
    def analyze_performance(self, df, symbol):
        """Comprehensive performance analysis"""
        print(f"\n{'='*50}")
        print(f"PERFORMANCE ANALYSIS: {symbol}")
        print(f"{'='*50}")

        if df is None or len(df) == 0:
            print("No data to analyze.")
            return
        
        print(f"Analysis Period: {df.index.min().date()} to {df.index.max().date()}")
        print(f"Total Trading Days: {len(df)}")
        print(f"Initial Price: ${df['close'].iloc[0]:.2f}")
        print(f"Final Price: ${df['close'].iloc[-1]:.2f}")
        
        # Returns
        total_return = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        print(f"Total Return: {total_return:.2f}%")
        
        # Risk metrics
        print(f"Average Daily Return: {df['daily_return'].mean():.3f}%")
        print(f"Daily Return Volatility: {df['daily_return'].std():.3f}%")
        print(f"Maximum Daily Gain: {df['daily_return'].max():.3f}%")
        print(f"Maximum Daily Loss: {df['daily_return'].min():.3f}%")
        
        # RSI statistics
        if 'rsi' in df.columns:
            print(f"Average RSI: {df['rsi'].mean():.1f}")
            print(f"Oversold Days (RSI < 30): {(df['rsi'] < 30).sum()}")
            print(f"Overbought Days (RSI > 70): {(df['rsi'] > 70).sum()}")

    def plot_comprehensive_analysis(self, df, symbol, df_av=None, df_yf=None):
        """Create comprehensive visualization"""
        if df is None:
            print("No data to plot.")
            return
            
        fig = plt.figure(figsize=(15, 12))
        
        # Create subplots
        gs = plt.GridSpec(4, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])  # Price and indicators
        ax2 = fig.add_subplot(gs[1, :])  # Daily returns
        ax3 = fig.add_subplot(gs[2, 0])  # RSI
        ax4 = fig.add_subplot(gs[2, 1])  # Volume
        ax5 = fig.add_subplot(gs[3, :])  # Data source comparison (if available)
        
        # Plot 1: Price and Moving Averages
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=1, color='black')
        ax1.plot(df.index, df['sma_20'], label='20-day SMA', linewidth=1.5, color='blue', alpha=0.8)
        ax1.plot(df.index, df['sma_50'], label='50-day SMA', linewidth=1.5, color='red', alpha=0.8)
        
        # Plot signals
        if 'signal' in df.columns:
            strong_buy = df[df['signal'] == 2]
            strong_sell = df[df['signal'] == -2]
            weak_buy = df[df['signal'] == 1]
            weak_sell = df[df['signal'] == -1]
            
            ax1.scatter(strong_buy.index, strong_buy['close'], 
                       color='darkgreen', marker='^', s=100, label='Strong Buy', zorder=5)
            ax1.scatter(strong_sell.index, strong_sell['close'], 
                       color='darkred', marker='v', s=100, label='Strong Sell', zorder=5)
            ax1.scatter(weak_buy.index, weak_buy['close'], 
                       color='lightgreen', marker='^', s=60, label='Weak Buy', alpha=0.7, zorder=4)
            ax1.scatter(weak_sell.index, weak_sell['close'], 
                       color='lightcoral', marker='v', s=60, label='Weak Sell', alpha=0.7, zorder=4)
        
        ax1.set_title(f'{symbol} - Technical Analysis', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Daily Returns
        colors = ['red' if x < 0 else 'green' for x in df['daily_return']]
        ax2.bar(df.index, df['daily_return'], color=colors, alpha=0.7)
        ax2.set_title('Daily Returns (%)')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: RSI
        ax3.plot(df.index, df['rsi'], color='purple', linewidth=1.5)
        ax3.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought')
        ax3.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold')
        ax3.set_title('Relative Strength Index (RSI)')
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Volume
        ax4.bar(df.index, df['volume'] / 1e6, color='gray', alpha=0.7)
        ax4.set_title('Trading Volume')
        ax4.set_ylabel('Volume (Millions)')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Data Source Comparison (if both sources available)
        if df_av is not None and df_yf is not None:
            common_dates = df_av.index.intersection(df_yf.index)
            if len(common_dates) > 0:
                ax5.plot(common_dates, df_av.loc[common_dates, 'close'], 
                        label='Alpha Vantage', alpha=0.7)
                ax5.plot(common_dates, df_yf.loc[common_dates, 'close'], 
                        label='Yahoo Finance', alpha=0.7)
                ax5.set_title('Data Source Comparison (Closing Prices)')
                ax5.set_ylabel('Price ($)')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'No common dates for comparison', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Data Source Comparison')
        else:
            ax5.text(0.5, 0.5, 'Only one data source available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Data Source Comparison')
        
        plt.tight_layout()
        plt.show()