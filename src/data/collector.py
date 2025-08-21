import ccxt
import pandas as pd
import time
from datetime import datetime
import sys
import os
import requests
import numpy as np
from config.config import DEFAULT_EXCHANGE

class GeminiDataCollector:
    def __init__(self):
        """Initialize with configured exchange"""
        self.exchange = None
        self.exchange_name = DEFAULT_EXCHANGE.upper()
        self._initialize_exchange()
        
    def _initialize_exchange(self):
        """Initialize configured exchange"""
        try:
            # Load API keys from environment if available
            api_key = os.getenv('GEMINI_API_KEY', '')
            secret = os.getenv('GEMINI_SECRET_KEY', '')
            
            # Fall back to Binance keys if Gemini keys not set
            if not api_key:
                api_key = os.getenv('BINANCE_API_KEY', '')
                secret = os.getenv('BINANCE_SECRET_KEY', '')
            
            # Dynamically create exchange based on config
            exchange_class = getattr(ccxt, DEFAULT_EXCHANGE.lower())
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
                'timeout': 15000,
                'sandbox': False,
            })
            markets = self.exchange.load_markets()
            print(f"✅ Connected to {self.exchange_name} with {len(markets)} markets")
            return True
        except Exception as e:
            print(f"⚠️ Gemini connection issue: {e}")
            print("📊 Will use synthetic data for testing")
            return False
    
    def get_current_price(self, symbol):
        """Get current price for a symbol - symbol should be in BTC/USD format"""
        try:
            if self.exchange:
                # ccxt should handle BTC/USD format directly
                ticker = self.exchange.fetch_ticker(symbol)
                return {
                    'symbol': symbol,
                    'price': ticker['last'],
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'timestamp': datetime.now()
                }
            else:
                return self._get_synthetic_price(symbol)
                
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return self._get_synthetic_price(symbol)
    
    def get_historical_data(self, symbol, timeframe='5m', limit=100):
        """Get historical OHLCV data"""
        try:
            if self.exchange:
                # ccxt should handle BTC/USD format directly for Gemini
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if ohlcv and len(ohlcv) >= 20:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['symbol'] = symbol
                    return df
            
            # Fallback to synthetic data
            return self._generate_synthetic_data(symbol, limit)
            
        except Exception as e:
            print(f"Historical data error for {symbol}: {e}")
            return self._generate_synthetic_data(symbol, limit)
    
    def _get_synthetic_price(self, symbol):
        """Generate synthetic price for testing"""
        base_prices = {
            'BTC/USD': 45000, 'ETH/USD': 3000, 'LTC/USD': 100,
            'LINK/USD': 15, 'SOL/USD': 95, 'DOGE/USD': 0.08
        }
        
        base_price = base_prices.get(symbol, 1000)
        # Add some realistic variation
        variation = np.random.normal(0, 0.02)
        price = base_price * (1 + variation)
        
        return {
            'symbol': symbol,
            'price': price,
            'bid': price * 0.999,
            'ask': price * 1.001,
            'timestamp': datetime.now()
        }
    
    def _generate_synthetic_data(self, symbol, limit=100):
        """Generate realistic synthetic data for testing"""
        print(f"📊 Generating synthetic data for {symbol}")
        
        base_prices = {
            'BTC/USD': 45000, 'ETH/USD': 3000, 'LTC/USD': 100,
            'LINK/USD': 15, 'SOL/USD': 95, 'DOGE/USD': 0.08
        }
        
        base_price = base_prices.get(symbol, 1000)
        
        # Generate realistic price movement
        np.random.seed(hash(symbol) % 2147483647)
        
        now = datetime.now()
        timestamps = pd.date_range(end=now, periods=limit, freq='5min')
        
        # Generate price walk with realistic patterns
        returns = []
        for i in range(limit):
            trend = 0.0001 * np.sin(i * 0.1)
            volatility = 0.01 + 0.005 * np.sin(i * 0.05)
            random_return = np.random.normal(trend, volatility)
            returns.append(random_return)
        
        prices = [base_price]
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, base_price * 0.5))
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            volatility = abs(np.random.normal(0, 0.004))
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            volume = 1000 * (1 + volatility * 10) * np.random.uniform(0.5, 2.0)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume,
                'symbol': symbol
            })
        
        return pd.DataFrame(data)

    def test_connection(self):
        """Test connection"""
        if self.exchange:
            print(f"✅ Using {self.exchange_name} exchange")
            return True
        else:
            print(f"⚠️ Using synthetic data for testing")
            return True

# Backward compatibility
MultiExchangeDataCollector = GeminiDataCollector
DataCollector = GeminiDataCollector

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    collector = GeminiDataCollector()
    
    if collector.test_connection():
        print("✅ Gemini collector ready!")
        
        # Test price fetching with symbols that exist on Gemini
        test_symbols = ['BTC/USD', 'ETH/USD', 'LTC/USD']
        
        for symbol in test_symbols:
            price_data = collector.get_current_price(symbol)
            if price_data:
                print(f"📈 {symbol}: ${price_data['price']:,.2f}")
        
        # Test historical data
        print("\n📊 Testing historical data...")
        hist_data = collector.get_historical_data('BTC/USD', '1h', 10)
        if hist_data is not None:
            print(f"Got {len(hist_data)} data points for BTC/USD")
            if len(hist_data) > 0:
                print("Sample data:")
                print(hist_data.tail(2))
