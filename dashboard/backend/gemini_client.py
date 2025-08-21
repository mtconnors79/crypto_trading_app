import ccxt
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import csv
import json
from config.config import DEFAULT_EXCHANGE

load_dotenv()

class GeminiDashboardClient:
    def __init__(self):
        """Initialize dashboard client with configured exchange"""
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
            'sandbox': False
        })
        
    async def get_portfolio_summary(self):
        """Get complete portfolio summary"""
        try:
            balance = self.exchange.fetch_balance()
            
            # Get current prices for positions
            positions = []
            total_portfolio_value = float(balance['USD']['free']) if 'USD' in balance else 0.0
            
            # Check each currency in balance
            for currency in balance:
                if currency == 'USD':
                    continue
                    
                # Skip metadata fields
                if currency in ['info', 'free', 'used', 'total']:
                    continue
                    
                # Get balance info
                currency_balance = balance[currency]
                if isinstance(currency_balance, dict) and currency_balance.get('total', 0) > 0:
                    try:
                        symbol = f"{currency}/USD"
                        ticker = self.exchange.fetch_ticker(symbol)
                        current_price = float(ticker['last'])
                        total_amount = float(currency_balance['total'])
                        position_value = total_amount * current_price
                        total_portfolio_value += position_value
                        
                        positions.append({
                            'symbol': symbol,
                            'quantity': total_amount,
                            'free': float(currency_balance.get('free', 0)),
                            'used': float(currency_balance.get('used', 0)),
                            'current_price': current_price,
                            'position_value': position_value
                        })
                    except Exception as e:
                        print(f"Error processing {currency}: {e}")
                        continue
            
            return {
                'cash_balance': float(balance['USD']['free']) if 'USD' in balance else 0.0,
                'total_portfolio_value': total_portfolio_value,
                'positions': positions,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Portfolio error: {e}")
            return {'error': str(e)}
    
    async def get_trade_history(self, limit=50):
        """Get recent trade history from Gemini"""
        try:
            trades = []
            
            # Get trades for symbols we know exist
            symbols = ['BTC/USD', 'ETH/USD', 'DOGE/USD', 'SHIB/USD', 'PEPE/USD']
            
            for symbol in symbols:
                try:
                    symbol_trades = self.exchange.fetch_my_trades(symbol, limit=10)
                    trades.extend(symbol_trades)
                except:
                    continue
            
            # Sort by timestamp
            trades.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Format for frontend
            formatted_trades = []
            for trade in trades[:limit]:
                formatted_trades.append({
                    'id': trade['id'],
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'amount': float(trade['amount']),
                    'price': float(trade['price']),
                    'cost': float(trade['cost']),
                    'fee': float(trade['fee']['cost']) if trade['fee'] else 0,
                    'timestamp': datetime.fromtimestamp(trade['timestamp']/1000).isoformat(),
                    'datetime': trade['datetime']
                })
            
            return formatted_trades
        except Exception as e:
            print(f"Trade history error: {e}")
            return []
    
    async def get_bot_analytics(self):
        """Get bot performance analytics from logs"""
        try:
            # Try both possible locations for CSV
            csv_paths = [
                'logs/trades.csv',
                '../logs/trades.csv',
                '../../logs/trades.csv'
            ]
            
            csv_file = None
            for path in csv_paths:
                if os.path.exists(path):
                    csv_file = path
                    break
            
            if not csv_file:
                return {
                    'no_trades': True,
                    'message': 'No bot trade log found. Start trading bot to generate data.',
                    'total_trades': 0,
                    'buy_trades': 0,
                    'sell_trades': 0,
                    'win_rate': 0,
                    'recent_trades': [],
                    'strategy_stats': {}
                }
            
            df = pd.read_csv(csv_file)
            
            if df.empty:
                return {'no_trades': True, 'message': 'No trades in log file'}
            
            # Calculate analytics
            total_trades = len(df)
            buy_trades = len(df[df['action'] == 'BUY'])
            sell_trades = len(df[df['action'] == 'SELL'])
            
            # Win rate calculation (simplified)
            win_rate = 50  # Placeholder - would need proper P&L calculation
            
            # Recent trades
            recent_trades = df.tail(10).to_dict('records')
            
            # Strategy performance
            strategy_stats = {}
            if 'strategy' in df.columns:
                strategy_stats = df['strategy'].value_counts().to_dict()
            
            return {
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'win_rate': win_rate,
                'recent_trades': recent_trades,
                'strategy_stats': strategy_stats,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Analytics error: {e}")
            return {'error': str(e)}
    
    async def get_market_data(self, symbols=None):
        """Get current market data for tracked symbols"""
        if symbols is None:
            symbols = ['BTC/USD', 'ETH/USD', 'DOGE/USD', 'SHIB/USD', 'PEPE/USD']
        
        market_data = []
        
        for symbol in symbols:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                market_data.append({
                    'symbol': symbol,
                    'price': float(ticker['last']),
                    'change_24h': float(ticker['percentage']) if ticker['percentage'] else 0,
                    'volume_24h': float(ticker['quoteVolume']) if ticker['quoteVolume'] else 0,
                    'high_24h': float(ticker['high']) if ticker['high'] else 0,
                    'low_24h': float(ticker['low']) if ticker['low'] else 0
                })
            except Exception as e:
                print(f"Market data error for {symbol}: {e}")
                continue
        
        return market_data
