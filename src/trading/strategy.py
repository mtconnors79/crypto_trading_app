import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

class AdvancedTradingStrategy:
    def __init__(self):
        """Advanced Multi-Indicator Strategy for Maximum Profit"""
        self.name = "Advanced Profit Maximizer"
        self.short_window = 7   # 7-period EMA
        self.long_window = 21   # 21-period EMA
        self.rsi_period = 14
        self.bb_period = 20
        self.min_confidence = 0.20
        
    def calculate_indicators(self, df):
        """Calculate advanced technical indicators"""
        # Exponential Moving Averages
        df['ema_short'] = df['close'].ewm(span=self.short_window).mean()
        df['ema_long'] = df['close'].ewm(span=self.long_window).mean()
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        bb_std = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility and momentum
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        df['momentum'] = df['close'].pct_change(periods=10)
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signal(self, df, symbol='', current_balance=100):
        """Generate trading signals"""
        try:
            if len(df) < max(self.long_window, self.bb_period):
                return 'hold', 0.0, "Insufficient data"
            
            # Calculate indicators
            df_indicators = self.calculate_indicators(df.copy())
            
            current = df_indicators.iloc[-1]
            previous = df_indicators.iloc[-2]
            
            # Get values
            price = current['close']
            ema_short = current['ema_short']
            ema_long = current['ema_long']
            rsi = current['rsi']
            macd = current['macd']
            macd_signal = current['macd_signal']
            bb_upper = current['bb_upper']
            bb_lower = current['bb_lower']
            bb_width = current['bb_width']
            volume_ratio = current['volume_ratio']
            momentum = current['momentum']
            
            # Previous values
            prev_ema_short = previous['ema_short']
            prev_ema_long = previous['ema_long']
            prev_macd = previous['macd']
            prev_macd_signal = previous['macd_signal']
            
            # Check for NaN values
            if pd.isna([ema_short, ema_long, rsi, macd, macd_signal]).any():
                return 'hold', 0.0, "Invalid indicator values"
            
            confidence = 0.0
            signals = []
            
            # Signal 1: EMA Crossover (30% weight)
            if ema_short > ema_long and prev_ema_short <= prev_ema_long:
                signals.append(('buy', 0.3, 'EMA bullish crossover'))
            elif ema_short < ema_long and prev_ema_short >= prev_ema_long:
                signals.append(('sell', 0.3, 'EMA bearish crossover'))
            
            # Signal 2: RSI (25% weight)
            if 30 < rsi < 40:  # Oversold recovery
                signals.append(('buy', 0.25, f'RSI oversold recovery ({rsi:.1f})'))
            elif 60 < rsi < 70:  # Overbought
                signals.append(('sell', 0.25, f'RSI overbought ({rsi:.1f})'))
            
            # Signal 3: MACD (20% weight)
            if macd > macd_signal and prev_macd <= prev_macd_signal:
                signals.append(('buy', 0.2, 'MACD bullish'))
            elif macd < macd_signal and prev_macd >= prev_macd_signal:
                signals.append(('sell', 0.2, 'MACD bearish'))
            
            # Signal 4: Bollinger Bands (15% weight)
            if not pd.isna(bb_lower) and price <= bb_lower * 1.01 and bb_width > 0.02:
                signals.append(('buy', 0.15, 'BB oversold'))
            elif not pd.isna(bb_upper) and price >= bb_upper * 0.99 and bb_width > 0.02:
                signals.append(('sell', 0.15, 'BB overbought'))
            
            # Signal 5: Volume confirmation (10% weight)
            if not pd.isna(volume_ratio) and volume_ratio > 1.2:
                if any(s[0] == 'buy' for s in signals):
                    signals.append(('buy', 0.1, 'Volume confirmation'))
                elif any(s[0] == 'sell' for s in signals):
                    signals.append(('sell', 0.1, 'Volume confirmation'))
            
            # Aggregate signals
            buy_confidence = sum(s[1] for s in signals if s[0] == 'buy')
            sell_confidence = sum(s[1] for s in signals if s[0] == 'sell')
            
            # Market condition adjustments
            if not pd.isna(momentum):
                if momentum > 0.02:  # Strong upward momentum
                    buy_confidence *= 1.1
                elif momentum < -0.02:  # Strong downward momentum
                    sell_confidence *= 1.1
            
            # Determine final signal
            if buy_confidence > sell_confidence and buy_confidence >= self.min_confidence:
                reasons = [s[2] for s in signals if s[0] == 'buy']
                return 'buy', min(0.95, buy_confidence), ' + '.join(reasons)
            elif sell_confidence > buy_confidence and sell_confidence >= self.min_confidence:
                reasons = [s[2] for s in signals if s[0] == 'sell']
                return 'sell', min(0.95, sell_confidence), ' + '.join(reasons)
            else:
                return 'hold', 0.0, f"Weak signals: Buy({buy_confidence:.2f}) Sell({sell_confidence:.2f})"
                
        except Exception as e:
            return 'hold', 0.0, f"Analysis error: {str(e)}"
    
    def calculate_position_size(self, signal, confidence, current_balance, current_price):
        """Calculate optimal position size"""
        if signal == 'hold':
            return 0
        
        # Base position size (70% of balance max)
        max_position_value = min(70.0, current_balance * 0.8)
        
        # Adjust based on confidence
        confidence_multiplier = 0.5 + (confidence * 1.0)  # 0.5x to 1.5x
        
        position_value = max_position_value * confidence_multiplier
        position_quantity = position_value / current_price
        
        return position_quantity

if __name__ == "__main__":
    print("🚀 Testing Advanced Trading Strategy...")
    
    # Create sample data for testing
    import numpy as np
    dates = pd.date_range(end=datetime.now(), periods=50, freq='5min')
    prices = [45000]
    for _ in range(49):
        change = np.random.normal(0, 0.01)
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.005 for p in prices],
        'low': [p * 0.995 for p in prices],
        'close': prices,
        'volume': [np.random.uniform(100, 1000) for _ in prices],
        'symbol': 'BTC/USD'
    })
    
    strategy = AdvancedTradingStrategy()
    signal, confidence, reason = strategy.generate_signal(df)
    
    print(f"📊 Test Signal: {signal.upper()}")
    print(f"📈 Confidence: {confidence:.2f}")
    print(f"📝 Reason: {reason}")
    print("✅ Strategy test completed!")
