#!/usr/bin/env python3
"""
Quick Backtest Runner - Fast strategy testing and comparison
Optimized for rapid iteration and strategy development
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.dirname(__file__))

from src.data.collector import GeminiDataCollector
from src.trading.strategy import AdvancedTradingStrategy
from config.trading_symbols import get_symbols_for_market

class QuickBacktester:
    def __init__(self):
        """Initialize quick backtesting framework"""
        self.data_collector = GeminiDataCollector()
        self.strategy = AdvancedTradingStrategy()
        
    def quick_backtest(self, symbols, days=7, initial_balance=1000):
        """
        Quick backtest for rapid strategy testing
        
        Args:
            symbols: List of symbols to test
            days: Number of days to backtest
            initial_balance: Starting balance
            
        Returns:
            Quick results summary
        """
        print(f"🚀 Quick Backtest: {len(symbols)} symbols, {days} days")
        
        # Fetch data quickly
        data = {}
        for symbol in symbols[:5]:  # Limit to 5 symbols for speed
            try:
                df = self.data_collector.get_historical_data(symbol, '5m', days * 288)  # 288 5min candles per day
                if df is not None and len(df) > 50:
                    data[symbol] = df
                    print(f"   ✅ {symbol}: {len(df)} candles")
            except Exception as e:
                print(f"   ❌ {symbol}: {e}")
        
        if not data:
            return {'error': 'No data available'}
        
        # Simple backtest simulation
        portfolio_value = initial_balance
        cash = initial_balance
        positions = {}
        trades = []
        
        # Get all timestamps
        all_times = set()
        for df in data.values():
            all_times.update(df['timestamp'])
        
        timestamps = sorted(list(all_times))
        
        for i, current_time in enumerate(timestamps):
            if i % 100 == 0:  # Progress
                progress = (i / len(timestamps)) * 100
                print(f"   📊 {progress:.0f}%", end='\r')
            
            # Get current prices
            current_prices = {}
            for symbol, df in data.items():
                current_data = df[df['timestamp'] <= current_time]
                if len(current_data) > 0:
                    current_prices[symbol] = current_data.iloc[-1]['close']
            
            # Update portfolio value
            portfolio_value = cash
            for symbol, pos in positions.items():
                if symbol in current_prices:
                    portfolio_value += pos['quantity'] * current_prices[symbol]
            
            # Generate signals and trade
            for symbol, df in data.items():
                if symbol not in current_prices:
                    continue
                
                current_data = df[df['timestamp'] <= current_time]
                if len(current_data) < 30:
                    continue
                
                signal, confidence, reason = self.strategy.generate_signal(current_data.tail(50))
                current_price = current_prices[symbol]
                
                # Simple trading logic
                if signal == 'buy' and confidence > 0.3 and symbol not in positions and len(positions) < 3:
                    investment = min(200, cash * 0.3)  # Max 30% per position
                    if investment > 50:  # Minimum investment
                        quantity = investment / current_price
                        cash -= investment
                        positions[symbol] = {'quantity': quantity, 'entry_price': current_price}
                        trades.append({
                            'time': current_time, 'symbol': symbol, 'action': 'BUY', 
                            'price': current_price, 'quantity': quantity, 'confidence': confidence
                        })
                
                elif signal == 'sell' and confidence > 0.3 and symbol in positions:
                    pos = positions[symbol]
                    proceeds = pos['quantity'] * current_price
                    pnl = proceeds - (pos['quantity'] * pos['entry_price'])
                    cash += proceeds
                    del positions[symbol]
                    trades.append({
                        'time': current_time, 'symbol': symbol, 'action': 'SELL',
                        'price': current_price, 'quantity': pos['quantity'], 'pnl': pnl, 'confidence': confidence
                    })
        
        # Calculate final results
        final_value = cash
        for symbol, pos in positions.items():
            if symbol in current_prices:
                final_value += pos['quantity'] * current_prices[symbol]
        
        total_return = ((final_value - initial_balance) / initial_balance) * 100
        
        # Quick analysis
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0
        
        print(f"\n✅ Quick backtest completed!")
        
        return {
            'initial_balance': initial_balance,
            'final_value': final_value,
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'symbols_tested': list(data.keys()),
            'trades': trades,
            'final_positions': positions
        }
    
    def compare_strategies(self, symbols, strategies_config):
        """
        Compare multiple strategy configurations
        
        Args:
            symbols: List of symbols to test
            strategies_config: Dict of strategy parameters to test
            
        Returns:
            Comparison results
        """
        print(f"⚖️  Comparing {len(strategies_config)} strategy configurations...")
        
        results = {}
        
        for config_name, config_params in strategies_config.items():
            print(f"\n🧪 Testing: {config_name}")
            
            # Apply configuration to strategy
            original_params = {}
            for param, value in config_params.items():
                if hasattr(self.strategy, param):
                    original_params[param] = getattr(self.strategy, param)
                    setattr(self.strategy, param, value)
            
            # Run quick backtest
            result = self.quick_backtest(symbols, days=10, initial_balance=1000)
            
            # Restore original parameters
            for param, value in original_params.items():
                setattr(self.strategy, param, value)
            
            if 'error' not in result:
                results[config_name] = {
                    'config': config_params,
                    'total_return': result['total_return'],
                    'win_rate': result['win_rate'],
                    'num_trades': result['num_trades'],
                    'final_value': result['final_value']
                }
                
                print(f"   📊 Return: {result['total_return']:+.2f}% | Win Rate: {result['win_rate']:.1%} | Trades: {result['num_trades']}")
        
        if results:
            # Find best configuration
            best_config = max(results.items(), key=lambda x: x[1]['total_return'])
            
            print(f"\n🏆 BEST CONFIGURATION: {best_config[0]}")
            print(f"📈 Return: {best_config[1]['total_return']:+.2f}%")
            print(f"🎯 Win Rate: {best_config[1]['win_rate']:.1%}")
            print(f"⚙️  Parameters: {best_config[1]['config']}")
        
        return results
    
    def test_symbol_groups(self):
        """Test different symbol groups for performance"""
        print(f"🎯 Testing different symbol group strategies...")
        
        symbol_groups = {
            'conservative': get_symbols_for_market('conservative')[:6],
            'balanced': get_symbols_for_market('balanced')[:6], 
            'aggressive': get_symbols_for_market('aggressive')[:6],
            'small_cap': get_symbols_for_market('small_cap')[:6],
            'meme': get_symbols_for_market('meme')[:6]
        }
        
        results = {}
        
        for group_name, symbols in symbol_groups.items():
            print(f"\n📊 Testing {group_name.upper()} symbols: {symbols}")
            result = self.quick_backtest(symbols, days=10, initial_balance=1000)
            
            if 'error' not in result:
                results[group_name] = result
                print(f"   💰 Return: {result['total_return']:+.2f}%")
                print(f"   🎯 Win Rate: {result['win_rate']:.1%}")
                print(f"   🔄 Trades: {result['num_trades']}")
        
        # Compare results
        if results:
            print(f"\n🏆 SYMBOL GROUP COMPARISON")
            print(f"{'='*50}")
            
            sorted_results = sorted(results.items(), key=lambda x: x[1]['total_return'], reverse=True)
            
            for group_name, result in sorted_results:
                print(f"{group_name.upper():12} | Return: {result['total_return']:+6.2f}% | "
                      f"Win Rate: {result['win_rate']:5.1%} | Trades: {result['num_trades']:3d}")
        
        return results
    
    def parameter_sensitivity_test(self, symbols):
        """Test parameter sensitivity"""
        print(f"🔧 Testing parameter sensitivity...")
        
        base_results = self.quick_backtest(symbols, days=7)
        
        if 'error' in base_results:
            print("❌ Cannot run sensitivity test - no baseline")
            return
        
        base_return = base_results['total_return']
        print(f"📊 Baseline return: {base_return:+.2f}%")
        
        # Test different confidence thresholds
        confidence_tests = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        print(f"\n🎯 Testing confidence thresholds...")
        
        for conf in confidence_tests:
            original_conf = self.strategy.min_confidence
            self.strategy.min_confidence = conf
            
            result = self.quick_backtest(symbols, days=7)
            if 'error' not in result:
                change = result['total_return'] - base_return
                print(f"   Confidence {conf:.2f}: {result['total_return']:+6.2f}% ({change:+.2f}%)")
            
            self.strategy.min_confidence = original_conf
        
        # Test different window sizes
        print(f"\n📈 Testing EMA windows...")
        window_tests = [(5, 15), (7, 21), (10, 30), (12, 26)]
        
        for short, long in window_tests:
            original_short = self.strategy.short_window
            original_long = self.strategy.long_window
            
            self.strategy.short_window = short
            self.strategy.long_window = long
            
            result = self.quick_backtest(symbols, days=7)
            if 'error' not in result:
                change = result['total_return'] - base_return
                print(f"   EMA {short}/{long}: {result['total_return']:+6.2f}% ({change:+.2f}%)")
            
            self.strategy.short_window = original_short
            self.strategy.long_window = original_long

def main():
    """Main quick backtesting execution"""
    print("⚡ QUICK CRYPTO BACKTEST RUNNER")
    print("="*40)
    
    backtester = QuickBacktester()
    
    # Test 1: Quick backtest with small-cap symbols
    print(f"\n🧪 TEST 1: Small-cap high-volatility backtest")
    small_cap_symbols = get_symbols_for_market('small_cap')[:6]
    result = backtester.quick_backtest(small_cap_symbols, days=10)
    
    if 'error' not in result:
        print(f"✅ Small-cap test completed!")
        print(f"   💰 Return: {result['total_return']:+.2f}%")
        print(f"   🎯 Win Rate: {result['win_rate']:.1%}")
        print(f"   🔄 Trades: {result['num_trades']}")
    
    # Test 2: Compare strategy configurations
    print(f"\n🧪 TEST 2: Strategy configuration comparison")
    strategies = {
        'conservative': {'min_confidence': 0.35, 'short_window': 7, 'long_window': 21},
        'aggressive': {'min_confidence': 0.20, 'short_window': 5, 'long_window': 15},
        'balanced': {'min_confidence': 0.25, 'short_window': 7, 'long_window': 20},
        'high_confidence': {'min_confidence': 0.40, 'short_window': 10, 'long_window': 25}
    }
    
    comparison = backtester.compare_strategies(small_cap_symbols, strategies)
    
    # Test 3: Symbol group comparison
    print(f"\n🧪 TEST 3: Symbol group performance")
    group_results = backtester.test_symbol_groups()
    
    # Test 4: Parameter sensitivity
    print(f"\n🧪 TEST 4: Parameter sensitivity analysis")
    backtester.parameter_sensitivity_test(small_cap_symbols)
    
    print(f"\n✅ Quick backtesting completed!")
    print(f"💡 Use the full backtesting framework for detailed analysis")

if __name__ == "__main__":
    main()
