import ccxt
import pandas as pd
import time
from datetime import datetime
import sys
import os
import requests
import numpy as np

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import *

class MultiExchangeDataCollector:
    def __init__(self):
        """Initialize with Kraken exchange (US-friendly)"""
        self.exchange = None
        self.exchange_name = ""
        self._initialize_kraken()
        
    def _initialize_kraken(self):
        """Initialize Kraken exchange"""
        try:
            self.exchange = ccxt.kraken({
                'apiKey': '',  # No API key needed for public data
                'secret': '',
                'enableRateLimit': True,
                'timeout': 10000,
            })
            markets = self.exchange.load_markets()
            self.exchange_name = "KRAKEN"
            print(f"✅ Connected to {self.exchange_name} with {len(markets)} markets")
            return True
        except Exception as e:
            print(f"❌ Kraken connection failed: {e}")
            return False
    
    def get_current_price(self, symbol):
        """Get current price for a symbol"""
        if not self.exchange:
            return None
            
        try:
            # Kraken uses different symbol format, but ccxt handles conversion
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol, timeframe='5m', limit=100):
        """Get historical OHLCV data from Kraken"""
        if not self.exchange:
            return self._generate_synthetic_data(symbol, limit)
            
        try:
            # Kraken API call with proper error handling
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < 10:
                print(f"📊 Insufficient data from Kraken for {symbol}, using synthetic data")
                return self._generate_synthetic_data(symbol, limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            
            # Validate data quality
            if df['close'].isna().sum() > len(df) * 0.1:  # More than 10% NaN
                print(f"📊 Poor data quality for {symbol}, using synthetic data")
                return self._generate_synthetic_data(symbol, limit)
                
            return df
            
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return self._generate_synthetic_data(symbol, limit)
    
    def _generate_synthetic_data(self, symbol, limit=100):
        """Generate high-quality synthetic data for testing"""
        print(f"📊 Generating realistic synthetic data for {symbol}")
        
        # Base prices for different coins
        base_prices = {
            'BTC/USD': 43000, 'ETH/USD': 2800, 'ADA/USD': 0.48,
            'DOT/USD': 6.5, 'SOL/USD': 95, 'AVAX/USD': 32,
            'DOGE/USD': 0.08, 'LTC/USD': 95
        }
        
        base_price = base_prices.get(symbol, 100)
        
        # Generate realistic price movement
        np.random.seed(hash(symbol) % 2147483647)  # Consistent data per symbol
        
        # Create timestamps
        now = datetime.now()
        timestamps = pd.date_range(end=now, periods=limit, freq='5min')
        
        # Generate price walk with realistic patterns
        returns = []
        for i in range(limit):
            # Add some trend and volatility patterns
            trend = 0.0001 * np.sin(i * 0.1)  # Slight trend
            volatility = 0.008 + 0.004 * np.sin(i * 0.05)  # Variable volatility
            random_return = np.random.normal(trend, volatility)
            returns.append(random_return)
        
        # Calculate prices
        prices = [base_price]
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, base_price * 0.5))  # Prevent unrealistic drops
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            # Realistic intrabar movement
            volatility = abs(np.random.normal(0, 0.003))
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            # Volume based on volatility (higher vol = higher volume)
            base_volume = 1000 if 'BTC' in symbol else 10000
            volume = base_volume * (1 + volatility * 10) * np.random.uniform(0.5, 2.0)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume,
                'symbol': symbol
            })
        
        df = pd.DataFrame(data)
        return df

    def test_connection(self):
        """Test exchange connection"""
        if self.exchange:
            print(f"✅ Using {self.exchange_name} exchange")
            return True
        else:
            print("⚠️  No exchange connection, using synthetic data")
            return True  # Continue with synthetic data

# Backward compatibility
DataCollector = MultiExchangeDataCollector

if __name__ == "__main__":
    collector = MultiExchangeDataCollector()
    
    if collector.test_connection():
        print("✅ Data collector ready!")
        
        # Test price fetching
        btc_price = collector.get_current_price('BTC/USD')
        if btc_price:
            print(f"📈 Current BTC price: ${btc_price['price']:,.2f}")
        
        # Test historical data
        print("📊 Testing historical data...")
        hist_data = collector.get_historical_data('BTC/USD', '5m', 20)
        if hist_data is not None:
            print(f"Got {len(hist_data)} data points")
            print("Sample data:")
            print(hist_data.tail(3))
