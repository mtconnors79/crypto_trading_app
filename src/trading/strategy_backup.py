import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add config path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import *

class SimpleMovingAverageStrategy:
    def __init__(self):
        """Simple Moving Average Crossover Strategy"""
        self.name = "SMA Crossover"
        self.short_window = 5   # 5-period moving average
        self.long_window = 20   # 20-period moving average
        
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # Simple Moving Averages
        df['sma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_window).mean()
        
        # RSI (Relative Strength Index)
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signal(self, df):
        """Generate buy/sell/hold signals"""
        if len(df) < self.long_window:
            return 'hold', 0.0, "Insufficient data"
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Get current values
        current_price = current['close']
        sma_short = current['sma_short']
        sma_long = current['sma_long']
        rsi = current['rsi']
        
        prev_sma_short = previous['sma_short']
        prev_sma_long = previous['sma_long']
        
        confidence = 0.0
        reason = ""
        
        # Buy conditions
        if (sma_short > sma_long and 
            prev_sma_short <= prev_sma_long and  # Crossover just happened
            rsi < 70 and  # Not overbought
            current_price < current['bb_upper']):  # Not at upper band
            
            confidence = min(0.8, (sma_short - sma_long) / sma_long)
            reason = f"SMA crossover: Short({sma_short:.2f}) > Long({sma_long:.2f}), RSI: {rsi:.1f}"
            return 'buy', confidence, reason
        
        # Sell conditions
        elif (sma_short < sma_long and 
              prev_sma_short >= prev_sma_long and  # Crossover just happened
              rsi > 30):  # Not oversold
            
            confidence = min(0.8, (sma_long - sma_short) / sma_long)
            reason = f"SMA crossover: Short({sma_short:.2f}) < Long({sma_long:.2f}), RSI: {rsi:.1f}"
            return 'sell', confidence, reason
        
        # Hold conditions
        else:
            reason = f"No clear signal. SMA: {sma_short:.2f}/{sma_long:.2f}, RSI: {rsi:.1f}"
            return 'hold', 0.0, reason
     def backtest_strategy(self, df, initial_balance=1000):
        """Simple backtest of the strategy"""
        try:
            df = self.calculate_indicators(df.copy())
            
            if len(df) < self.long_window:
                return {
                    'initial_balance': initial_balance,
                    'final_value': initial_balance,
                    'total_return': 0.0,
                    'trades': [],
                    'num_trades': 0
                }
            
            balance = initial_balance
            position = 0
            trades = []
            
            for i in range(self.long_window, len(df)):
                current_data = df.iloc[:i+1]
                signal, confidence, reason = self.generate_signal(current_data)
                current_price = df.iloc[i]['close']
                
                if pd.isna(current_price):
                    continue
                    
                if signal == 'buy' and position == 0 and confidence > 0.3:
                    # Buy with 90% of available balance
                    position = (balance * 0.9) / current_price
                    balance = balance * 0.1  # Keep 10% cash
                    trades.append({
                        'timestamp': df.iloc[i]['timestamp'],
                        'action': 'buy',
                        'price': current_price,
                        'quantity': position,
                        'reason': reason,
                        'confidence': confidence
                    })
                    
                elif signal == 'sell' and position > 0:
                    # Sell all position
                    balance += position * current_price
                    trades.append({
                        'timestamp': df.iloc[i]['timestamp'],
                        'action': 'sell',
                        'price': current_price,
                        'quantity': position,
                        'reason': reason,
                        'confidence': confidence
                    })
                    position = 0
            
            # Calculate final value
            final_price = df.iloc[-1]['close']
            if pd.isna(final_price):
                final_price = df.dropna(subset=['close']).iloc[-1]['close']
                
            final_value = balance + (position * final_price)
            total_return = ((final_value - initial_balance) / initial_balance) * 100
            
            return {
                'initial_balance': initial_balance,
                'final_value': final_value,
                'total_return': total_return,
                'trades': trades,
                'num_trades': len(trades)
            }
            
        except Exception as e:
            print(f"Backtest error: {e}")
            return {
                'initial_balance': initial_balance,
                'final_value': initial_balance,
                'total_return': 0.0,
                'trades': [],
                'num_trades': 0
            } 
