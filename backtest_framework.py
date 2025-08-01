#!/usr/bin/env python3
"""
Advanced Backtesting Framework for Crypto Trading Bot
Comprehensive strategy testing with realistic market conditions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import sys
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.dirname(__file__))

from src.data.collector import GeminiDataCollector
from src.trading.strategy import AdvancedTradingStrategy
from config.trading_symbols import SYMBOLS, get_symbols_for_market

warnings.filterwarnings('ignore')

class AdvancedBacktester:
    def __init__(self, initial_balance=1000, max_positions=4, trading_fee=0.0025):
        """
        Initialize comprehensive backtesting framework
        
        Args:
            initial_balance: Starting capital
            max_positions: Maximum concurrent positions
            trading_fee: Trading fee percentage (0.0025 = 0.25%)
        """
        self.initial_balance = initial_balance
        self.max_positions = max_positions
        self.trading_fee = trading_fee
        
        # Initialize components
        self.data_collector = GeminiDataCollector()
        self.strategy = AdvancedTradingStrategy()
        
        # Backtest results storage
        self.results = {}
        self.detailed_trades = []
        self.portfolio_history = []
        
        print("🚀 Advanced Crypto Backtesting Framework Initialized")
        print(f"💰 Initial Balance: ${initial_balance:,}")
        print(f"📊 Max Positions: {max_positions}")
        print(f"💸 Trading Fee: {trading_fee:.4f} ({trading_fee*100:.2f}%)")
    
    def fetch_historical_data(self, symbols: List[str], timeframe='5m', days=30) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols with parallel processing
        
        Args:
            symbols: List of trading symbols
            timeframe: Data timeframe ('1m', '5m', '1h', '1d')
            days: Number of days of history
            
        Returns:
            Dictionary of DataFrames keyed by symbol
        """
        print(f"📊 Fetching {days} days of {timeframe} data for {len(symbols)} symbols...")
        
        # Calculate required data points
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30, 
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        minutes_per_day = 1440
        limit = int((days * minutes_per_day) / timeframe_minutes.get(timeframe, 5))
        limit = min(limit, 1000)  # API limits
        
        data = {}
        successful_fetches = 0
        
        def fetch_symbol_data(symbol):
            try:
                df = self.data_collector.get_historical_data(symbol, timeframe, limit)
                if df is not None and len(df) > 100:  # Minimum data requirement
                    # Ensure data quality
                    df = df.dropna()
                    df = df.reset_index(drop=True)
                    return symbol, df
                else:
                    print(f"   ⚠️  {symbol}: Insufficient data ({len(df) if df is not None else 0} points)")
                    return symbol, None
            except Exception as e:
                print(f"   ❌ {symbol}: Data fetch error - {e}")
                return symbol, None
        
        # Parallel data fetching
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {executor.submit(fetch_symbol_data, symbol): symbol for symbol in symbols}
            
            for future in as_completed(future_to_symbol):
                symbol, df = future.result()
                if df is not None:
                    data[symbol] = df
                    successful_fetches += 1
                    print(f"   ✅ {symbol}: {len(df)} data points")
        
        print(f"📈 Successfully fetched data for {successful_fetches}/{len(symbols)} symbols")
        return data
    
    def simulate_realistic_execution(self, signal_price: float, signal_time: pd.Timestamp, 
                                   next_candles: pd.DataFrame, order_type='market') -> Tuple[float, float]:
        """
        Simulate realistic order execution with slippage and timing delays
        
        Args:
            signal_price: Price when signal was generated
            signal_time: Time when signal was generated
            next_candles: Next few candles for execution simulation
            order_type: 'market' or 'limit'
            
        Returns:
            Tuple of (execution_price, slippage_cost)
        """
        if len(next_candles) == 0:
            return signal_price, 0.0
        
        if order_type == 'market':
            # Market order: executed at next candle's open with slippage
            execution_price = next_candles.iloc[0]['open']
            slippage = abs(execution_price - signal_price) / signal_price
            
            # Add realistic slippage for crypto (higher for small caps)
            additional_slippage = np.random.normal(0.001, 0.0005)  # 0.1% avg slippage
            execution_price *= (1 + additional_slippage)
            
        else:  # limit order
            # Limit order: may not fill immediately
            next_candle = next_candles.iloc[0]
            
            # Check if limit order would fill
            if signal_price >= next_candle['low'] and signal_price <= next_candle['high']:
                execution_price = signal_price  # Limit order filled
                slippage = 0.0
            else:
                # Order didn't fill, use next candle's open
                execution_price = next_candle['open']
                slippage = abs(execution_price - signal_price) / signal_price
        
        return execution_price, slippage
    
    def run_backtest(self, data: Dict[str, pd.DataFrame], start_date=None, end_date=None,
                    position_sizing='dynamic', max_position_pct=0.2) -> Dict:
        """
        Run comprehensive backtest simulation
        
        Args:
            data: Historical data dictionary
            start_date: Backtest start date
            end_date: Backtest end date  
            position_sizing: 'fixed', 'dynamic', or 'kelly'
            max_position_pct: Maximum position size as % of portfolio
            
        Returns:
            Backtest results dictionary
        """
        print(f"\n🔄 Running backtest simulation...")
        print(f"📅 Period: {start_date or 'All available'} to {end_date or 'All available'}")
        print(f"💼 Position Sizing: {position_sizing}")
        print(f"📊 Max Position: {max_position_pct*100:.1f}% of portfolio")
        
        # Initialize portfolio state
        cash_balance = self.initial_balance
        positions = {}  # {symbol: {'quantity': float, 'entry_price': float, 'entry_date': timestamp}}
        portfolio_value_history = []
        trade_history = []
        
        # Get unified timeindex from all data
        all_timestamps = set()
        for df in data.values():
            if start_date:
                df = df[df['timestamp'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['timestamp'] <= pd.to_datetime(end_date)]
            all_timestamps.update(df['timestamp'].tolist())
        
        timestamps = sorted(list(all_timestamps))
        
        print(f"⏰ Simulating {len(timestamps)} time periods...")
        
        # Main simulation loop
        for i, current_time in enumerate(timestamps):
            # Calculate current portfolio value
            portfolio_value = cash_balance
            current_prices = {}
            
            # Get current prices and calculate portfolio value
            for symbol in data.keys():
                symbol_data = data[symbol]
                current_row = symbol_data[symbol_data['timestamp'] <= current_time]
                
                if len(current_row) > 0:
                    current_price = current_row.iloc[-1]['close']
                    current_prices[symbol] = current_price
                    
                    if symbol in positions:
                        position_value = positions[symbol]['quantity'] * current_price
                        portfolio_value += position_value
            
            # Record portfolio value
            portfolio_value_history.append({
                'timestamp': current_time,
                'portfolio_value': portfolio_value,
                'cash_balance': cash_balance,
                'num_positions': len(positions)
            })
            
            # Generate signals and execute trades
            for symbol in data.keys():
                if symbol not in current_prices:
                    continue
                    
                symbol_data = data[symbol]
                historical_data = symbol_data[symbol_data['timestamp'] <= current_time]
                
                if len(historical_data) < 50:  # Need minimum history for indicators
                    continue
                
                # Generate signal
                signal, confidence, reason = self.strategy.generate_signal(
                    historical_data.tail(100), symbol, cash_balance
                )
                
                current_price = current_prices[symbol]
                
                # Execute BUY signals
                if (signal == 'buy' and confidence >= 0.25 and 
                    len(positions) < self.max_positions and symbol not in positions):
                    
                    # Calculate position size
                    if position_sizing == 'dynamic':
                        position_pct = min(max_position_pct, 0.1 + (confidence * 0.3))
                    elif position_sizing == 'kelly':
                        # Simplified Kelly criterion
                        win_rate = 0.55  # Assume based on strategy
                        avg_win = 0.03   # 3% average win
                        avg_loss = 0.02  # 2% average loss
                        kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                        position_pct = min(max_position_pct, kelly_f * confidence)
                    else:  # fixed
                        position_pct = max_position_pct
                    
                    position_value = portfolio_value * position_pct
                    
                    # Simulate execution
                    future_data = symbol_data[symbol_data['timestamp'] > current_time].head(3)
                    execution_price, slippage = self.simulate_realistic_execution(
                        current_price, current_time, future_data, 'limit'
                    )
                    
                    # Calculate actual quantities
                    fee_cost = position_value * self.trading_fee
                    net_investment = position_value - fee_cost
                    quantity = net_investment / execution_price
                    total_cost = position_value
                    
                    if total_cost <= cash_balance:
                        # Execute trade
                        cash_balance -= total_cost
                        positions[symbol] = {
                            'quantity': quantity,
                            'entry_price': execution_price,
                            'entry_date': current_time,
                            'entry_confidence': confidence
                        }
                        
                        trade_history.append({
                            'timestamp': current_time,
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': quantity,
                            'price': execution_price,
                            'signal_price': current_price,
                            'slippage': slippage,
                            'fee': fee_cost,
                            'confidence': confidence,
                            'reason': reason,
                            'portfolio_value': portfolio_value
                        })
                
                # Execute SELL signals
                elif (signal == 'sell' and confidence >= 0.25 and symbol in positions):
                    position = positions[symbol]
                    
                    # Simulate execution
                    future_data = symbol_data[symbol_data['timestamp'] > current_time].head(3)
                    execution_price, slippage = self.simulate_realistic_execution(
                        current_price, current_time, future_data, 'limit'
                    )
                    
                    # Calculate proceeds
                    gross_proceeds = position['quantity'] * execution_price
                    fee_cost = gross_proceeds * self.trading_fee
                    net_proceeds = gross_proceeds - fee_cost
                    
                    # Calculate P&L
                    cost_basis = position['quantity'] * position['entry_price']
                    realized_pnl = net_proceeds - cost_basis
                    pnl_pct = (realized_pnl / cost_basis) * 100
                    
                    # Execute trade
                    cash_balance += net_proceeds
                    del positions[symbol]
                    
                    trade_history.append({
                        'timestamp': current_time,
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': position['quantity'],
                        'price': execution_price,
                        'signal_price': current_price,
                        'slippage': slippage,
                        'fee': fee_cost,
                        'confidence': confidence,
                        'reason': reason,
                        'entry_price': position['entry_price'],
                        'realized_pnl': realized_pnl,
                        'pnl_pct': pnl_pct,
                        'hold_days': (current_time - position['entry_date']).days,
                        'portfolio_value': portfolio_value
                    })
            
            # Progress reporting
            if i % max(1, len(timestamps) // 20) == 0:
                progress = (i / len(timestamps)) * 100
                print(f"   📊 Progress: {progress:.1f}% | Portfolio: ${portfolio_value:,.2f} | Positions: {len(positions)}")
        
        # Calculate final results
        final_portfolio_value = portfolio_value_history[-1]['portfolio_value'] if portfolio_value_history else self.initial_balance
        total_return = ((final_portfolio_value - self.initial_balance) / self.initial_balance) * 100
        
        # Store results
        self.portfolio_history = portfolio_value_history
        self.detailed_trades = trade_history
        
        results = {
            'initial_balance': self.initial_balance,
            'final_value': final_portfolio_value,
            'total_return': total_return,
            'num_trades': len(trade_history),
            'num_symbols_traded': len(set(trade['symbol'] for trade in trade_history)),
            'portfolio_history': portfolio_value_history,
            'trades': trade_history,
            'final_positions': positions
        }
        
        print(f"✅ Backtest completed!")
        print(f"📈 Total Return: {total_return:+.2f}%")
        print(f"🔄 Total Trades: {len(trade_history)}")
        
        return results
    
    def analyze_results(self, results: Dict) -> Dict:
        """
        Comprehensive analysis of backtest results
        
        Args:
            results: Backtest results dictionary
            
        Returns:
            Analysis metrics dictionary
        """
        print(f"\n📊 Analyzing backtest results...")
        
        if not results['trades']:
            return {'error': 'No trades to analyze'}
        
        trades_df = pd.DataFrame(results['trades'])
        portfolio_df = pd.DataFrame(results['portfolio_history'])
        
        # Basic metrics
        total_return = results['total_return']
        final_value = results['final_value']
        num_trades = len(trades_df)
        
        # Trade analysis
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        # P&L analysis
        winning_trades = sell_trades[sell_trades['realized_pnl'] > 0]
        losing_trades = sell_trades[sell_trades['realized_pnl'] < 0]
        
        win_rate = len(winning_trades) / len(sell_trades) if len(sell_trades) > 0 else 0
        avg_win = winning_trades['realized_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['realized_pnl'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['realized_pnl'].sum() / losing_trades['realized_pnl'].sum()) if len(losing_trades) > 0 and losing_trades['realized_pnl'].sum() != 0 else float('inf')
        
        # Risk metrics
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['cumulative_returns'] = (1 + portfolio_df['returns']).cumprod() - 1
        
        volatility = portfolio_df['returns'].std() * np.sqrt(252 * 24 * 12) if len(portfolio_df) > 1 else 0  # Annualized for 5min data
        sharpe_ratio = (total_return / 100) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        portfolio_df['peak'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
        max_drawdown = portfolio_df['drawdown'].min() * 100
        
        # Symbol performance
        symbol_performance = {}
        for symbol in trades_df['symbol'].unique():
            symbol_trades = sell_trades[sell_trades['symbol'] == symbol]
            if len(symbol_trades) > 0:
                symbol_pnl = symbol_trades['realized_pnl'].sum()
                symbol_trades_count = len(symbol_trades)
                symbol_win_rate = len(symbol_trades[symbol_trades['realized_pnl'] > 0]) / len(symbol_trades)
                symbol_performance[symbol] = {
                    'total_pnl': symbol_pnl,
                    'num_trades': symbol_trades_count,
                    'win_rate': symbol_win_rate,
                    'avg_pnl': symbol_pnl / symbol_trades_count
                }
        
        # Time-based analysis
        trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
        hourly_performance = trades_df.groupby('hour')['realized_pnl'].mean().dropna()
        
        analysis = {
            'performance_metrics': {
                'total_return_pct': total_return,
                'final_value': final_value,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'volatility_annualized': volatility * 100,
                'profit_factor': profit_factor
            },
            'trade_metrics': {
                'total_trades': num_trades,
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'best_trade': sell_trades['realized_pnl'].max() if len(sell_trades) > 0 else 0,
                'worst_trade': sell_trades['realized_pnl'].min() if len(sell_trades) > 0 else 0
            },
            'symbol_performance': symbol_performance,
            'execution_quality': {
                'avg_slippage': trades_df['slippage'].mean(),
                'total_fees': trades_df['fee'].sum(),
                'avg_confidence': trades_df['confidence'].mean()
            },
            'time_analysis': {
                'avg_hold_days': sell_trades['hold_days'].mean() if len(sell_trades) > 0 else 0,
                'hourly_performance': hourly_performance.to_dict()
            }
        }
        
        return analysis
    
    def create_performance_report(self, results: Dict, analysis: Dict, save_path='backtest_report.html'):
        """
        Create comprehensive HTML performance report
        
        Args:
            results: Backtest results
            analysis: Analysis metrics
            save_path: Path to save HTML report
        """
        print(f"📄 Creating performance report: {save_path}")
        
        # Generate visualizations
        portfolio_df = pd.DataFrame(results['portfolio_history'])
        trades_df = pd.DataFrame(results['trades'])
        
        # Create plots
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Crypto Trading Bot Backtest Results', fontsize=16, color='white')
        
        # Portfolio value over time
        axes[0,0].plot(pd.to_datetime(portfolio_df['timestamp']), portfolio_df['portfolio_value'], 
                      color='#00ff41', linewidth=2)
        axes[0,0].axhline(y=self.initial_balance, color='red', linestyle='--', alpha=0.7)
        axes[0,0].set_title('Portfolio Value Over Time', color='white')
        axes[0,0].set_ylabel('Value ($)', color='white')
        axes[0,0].tick_params(colors='white')
        
        # Drawdown chart
        portfolio_df['peak'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak'] * 100
        axes[0,1].fill_between(pd.to_datetime(portfolio_df['timestamp']), portfolio_df['drawdown'], 
                              color='red', alpha=0.7)
        axes[0,1].set_title('Drawdown (%)', color='white')
        axes[0,1].set_ylabel('Drawdown (%)', color='white')
        axes[0,1].tick_params(colors='white')
        
        # Trade P&L distribution
        if len(trades_df[trades_df['action'] == 'SELL']) > 0:
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            axes[1,0].hist(sell_trades['realized_pnl'], bins=20, color='cyan', alpha=0.7, edgecolor='white')
            axes[1,0].axvline(x=0, color='red', linestyle='--')
            axes[1,0].set_title('Trade P&L Distribution', color='white')
            axes[1,0].set_xlabel('P&L ($)', color='white')
            axes[1,0].tick_params(colors='white')
        
        # Symbol performance
        if analysis['symbol_performance']:
            symbols = list(analysis['symbol_performance'].keys())[:10]  # Top 10
            pnls = [analysis['symbol_performance'][s]['total_pnl'] for s in symbols]
            colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
            axes[1,1].barh(symbols, pnls, color=colors, alpha=0.7)
            axes[1,1].set_title('P&L by Symbol', color='white')
            axes[1,1].set_xlabel('Total P&L ($)', color='white')
            axes[1,1].tick_params(colors='white')
        
        plt.tight_layout()
        plt.savefig('backtest_charts.png', facecolor='black', dpi=100)
        plt.close()
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto Trading Bot Backtest Report</title>
            <style>
                body {{ background: #1a1a1a; color: #fff; font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .metric-card {{ background: #2d2d2d; padding: 15px; margin: 10px; border-radius: 8px; display: inline-block; min-width: 200px; }}
                .positive {{ color: #00ff41; }}
                .negative {{ color: #ff4757; }}
                .neutral {{ color: #ffa502; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #444; padding: 8px; text-align: left; }}
                th {{ background: #333; }}
                .charts {{ text-align: center; margin: 30px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Crypto Trading Bot Backtest Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>📊 Performance Summary</h2>
            <div class="metric-card">
                <h3>Total Return</h3>
                <h2 class="{'positive' if analysis['performance_metrics']['total_return_pct'] > 0 else 'negative'}">
                    {analysis['performance_metrics']['total_return_pct']:+.2f}%
                </h2>
            </div>
            <div class="metric-card">
                <h3>Final Value</h3>
                <h2>${analysis['performance_metrics']['final_value']:,.2f}</h2>
            </div>
            <div class="metric-card">
                <h3>Sharpe Ratio</h3>
                <h2 class="{'positive' if analysis['performance_metrics']['sharpe_ratio'] > 1 else 'neutral'}">
                    {analysis['performance_metrics']['sharpe_ratio']:.2f}
                </h2>
            </div>
            <div class="metric-card">
                <h3>Max Drawdown</h3>
                <h2 class="negative">{analysis['performance_metrics']['max_drawdown_pct']:.2f}%</h2>
            </div>
            
            <h2>🔄 Trading Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Trades</td><td>{analysis['trade_metrics']['total_trades']}</td></tr>
                <tr><td>Win Rate</td><td>{analysis['trade_metrics']['win_rate']:.1%}</td></tr>
                <tr><td>Average Win</td><td class="positive">${analysis['trade_metrics']['avg_win']:.2f}</td></tr>
                <tr><td>Average Loss</td><td class="negative">${analysis['trade_metrics']['avg_loss']:.2f}</td></tr>
                <tr><td>Profit Factor</td><td>{analysis['performance_metrics']['profit_factor']:.2f}</td></tr>
                <tr><td>Best Trade</td><td class="positive">${analysis['trade_metrics']['best_trade']:.2f}</td></tr>
                <tr><td>Worst Trade</td><td class="negative">${analysis['trade_metrics']['worst_trade']:.2f}</td></tr>
            </table>
            
            <div class="charts">
                <img src="backtest_charts.png" alt="Performance Charts" style="max-width: 100%;">
            </div>
            
            <h2>🎯 Top Performing Symbols</h2>
            <table>
                <tr><th>Symbol</th><th>Total P&L</th><th>Trades</th><th>Win Rate</th><th>Avg P&L</th></tr>
        """
        
        # Add top symbols to report
        sorted_symbols = sorted(analysis['symbol_performance'].items(), 
                               key=lambda x: x[1]['total_pnl'], reverse=True)[:10]
        
        for symbol, metrics in sorted_symbols:
            pnl_class = 'positive' if metrics['total_pnl'] > 0 else 'negative'
            html_content += f"""
                <tr>
                    <td>{symbol}</td>
                    <td class="{pnl_class}">${metrics['total_pnl']:.2f}</td>
                    <td>{metrics['num_trades']}</td>
                    <td>{metrics['win_rate']:.1%}</td>
                    <td class="{pnl_class}">${metrics['avg_pnl']:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Report saved to {save_path}")
    
    def run_parameter_optimization(self, data: Dict[str, pd.DataFrame], 
                                 param_ranges: Dict, optimization_metric='sharpe_ratio') -> Dict:
        """
        Optimize strategy parameters using backtest results
        
        Args:
            data: Historical data
            param_ranges: Dictionary of parameter ranges to test
            optimization_metric: Metric to optimize ('total_return', 'sharpe_ratio', 'profit_factor')
            
        Returns:
            Optimization results
        """
        print(f"\n🔧 Running parameter optimization...")
        print(f"📊 Optimizing for: {optimization_metric}")
        print(f"🎯 Parameter ranges: {param_ranges}")
        
        optimization_results = []
        total_combinations = np.prod([len(v) for v in param_ranges.values()])
        current_combination = 0
        
        # Generate all parameter combinations
        import itertools
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        for combination in itertools.product(*param_values):
            current_combination += 1
            params = dict(zip(param_names, combination))
            
            print(f"   Testing {current_combination}/{total_combinations}: {params}")
            
            # Apply parameters to strategy
            original_params = {}
            for param_name, param_value in params.items():
                if hasattr(self.strategy, param_name):
                    original_params[param_name] = getattr(self.strategy, param_name)
                    setattr(self.strategy, param_name, param_value)
            
            try:
                # Run backtest with current parameters
                results = self.run_backtest(data, position_sizing='dynamic', max_position_pct=0.25)
                analysis = self.analyze_results(results)
                
                if 'error' not in analysis:
                    optimization_score = analysis['performance_metrics'].get(optimization_metric, 0)
                    
                    optimization_results.append({
                        'parameters': params.copy(),
                        'score': optimization_score,
                        'total_return': analysis['performance_metrics']['total_return_pct'],
                        'sharpe_ratio': analysis['performance_metrics']['sharpe_ratio'],
                        'max_drawdown': analysis['performance_metrics']['max_drawdown_pct'],
                        'win_rate': analysis['trade_metrics']['win_rate'],
                        'num_trades': analysis['trade_metrics']['total_trades']
                    })
            
            except Exception as e:
                print(f"      ❌ Error with parameters {params}: {e}")
            
            # Restore original parameters
            for param_name, original_value in original_params.items():
                setattr(self.strategy, param_name, original_value)
        
        # Find best parameters
        if optimization_results:
            best_result = max(optimization_results, key=lambda x: x['score'])
            
            print(f"\n🏆 Optimization completed!")
            print(f"📊 Best {optimization_metric}: {best_result['score']:.4f}")
            print(f"🎯 Best parameters: {best_result['parameters']}")
            print(f"📈 Total return: {best_result['total_return']:+.2f}%")
            print(f"⚡ Sharpe ratio: {best_result['sharpe_ratio']:.2f}")
            
            return {
                'best_parameters': best_result['parameters'],
                'best_score': best_result['score'],
                'all_results': optimization_results,
                'optimization_metric': optimization_metric
            }
        else:
            print("❌ No valid optimization results")
            return {'error': 'No valid results'}

def main():
    """Main backtesting execution"""
    print("🚀 ADVANCED CRYPTO BACKTESTING FRAMEWORK")
    print("="*60)
    
    # Initialize backtester
    backtester = AdvancedBacktester(
        initial_balance=1000,
        max_positions=4,
        trading_fee=0.0025
    )
    
    # Test with small-cap high-volatility symbols
    test_symbols = get_symbols_for_market('small_cap')[:8]  # Test with 8 symbols
    print(f"🎯 Testing symbols: {test_symbols}")
    
    # Fetch historical data
    data = backtester.fetch_historical_data(test_symbols, timeframe='5m', days=14)
    
    if len(data) < 3:
        print("❌ Insufficient data for backtesting")
        return
    
    print(f"\n📊 Data loaded for {len(data)} symbols")
    
    # Run comprehensive backtest
    results = backtester.run_backtest(
        data, 
        position_sizing='dynamic',
        max_position_pct=0.25
    )
    
    # Analyze results
    analysis = backtester.analyze_results(results)
    
    if 'error' not in analysis:
        # Print summary
        print(f"\n🏆 BACKTEST SUMMARY")
        print(f"="*40)
        print(f"💰 Total Return: {analysis['performance_metrics']['total_return_pct']:+.2f}%")
        print(f"📊 Sharpe Ratio: {analysis['performance_metrics']['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {analysis['performance_metrics']['max_drawdown_pct']:.2f}%")
        print(f"🎯 Win Rate: {analysis['trade_metrics']['win_rate']:.1%}")
        print(f"🔄 Total Trades: {analysis['trade_metrics']['total_trades']}")
        print(f"💸 Total Fees: ${analysis['execution_quality']['total_fees']:.2f}")
        
        # Create performance report
        backtester.create_performance_report(results, analysis)
        
        # Run parameter optimization (optional)
        print(f"\n🔧 Running parameter optimization...")
        param_ranges = {
            'min_confidence': [0.2, 0.25, 0.3, 0.35],
            'short_window': [5, 7, 10],
            'long_window': [15, 20, 25]
        }
        
        optimization = backtester.run_parameter_optimization(
            data, param_ranges, 'sharpe_ratio'
        )
        
        if 'error' not in optimization:
            print(f"\n🏆 OPTIMIZATION RESULTS")
            print(f"🎯 Best parameters: {optimization['best_parameters']}")
            print(f"📊 Best Sharpe ratio: {optimization['best_score']:.4f}")
    
    print(f"\n✅ Backtesting completed! Check backtest_report.html for detailed results.")

if __name__ == "__main__":
    main()
