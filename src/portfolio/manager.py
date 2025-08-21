import pandas as pd
from datetime import datetime
from typing import Dict, List
import sys
import os
import ccxt
from dotenv import load_dotenv

load_dotenv()

class RealGeminiPortfolioManager:
    def __init__(self):
        """Initialize portfolio manager with real Gemini account"""
        # Initialize Gemini connection
        api_key = os.getenv('GEMINI_API_KEY', '')
        secret = os.getenv('GEMINI_SECRET_KEY', '')
        
        # Fall back to Binance keys if Gemini keys not set
        if not api_key:
            api_key = os.getenv('BINANCE_API_KEY', '')
            secret = os.getenv('BINANCE_SECRET_KEY', '')
        
        if not api_key or not secret:
            raise Exception("API keys not found in .env file (checked GEMINI and BINANCE)")
        
        self.exchange = ccxt.gemini({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'sandbox': False
        })
        
        # Get real account balance
        try:
            balance = self.exchange.fetch_balance()
            self.cash_balance = balance['USD']['free']  # Available USD for trading
            self.initial_balance = self.cash_balance  # Track starting amount
            print(f"💰 Real Gemini Balance: ${self.cash_balance:.2f}")
        except Exception as e:
            raise Exception(f"Cannot access Gemini account: {e}")
        
        # Track simulated positions (real trades would be tracked by Gemini)
        self.positions = {}  # For display purposes
        self.trades = []  # Track our trading activity
        self.total_fees = 0.0
        
    def get_real_balance(self):
        """Fetch current real balance from Gemini"""
        try:
            balance = self.exchange.fetch_balance()
            self.cash_balance = balance['USD']['free']
            return balance
        except Exception as e:
            print(f"Error fetching real balance: {e}")
            return None
    
    def buy(self, symbol: str, quantity: float, price: float, fee: float = 0.0):
        """Execute a REAL limit buy order on Gemini"""
        try:
            # Calculate order amount
            order_amount = quantity * price
            
            if order_amount > self.cash_balance:
                return False, "Insufficient funds"
            
            # Place REAL limit buy order at current market price (will fill immediately if price is fair)
            order = self.exchange.create_limit_buy_order(symbol, quantity, price)
            
            # For limit orders, we need to check if it was filled
            if order['status'] in ['closed', 'filled'] or order['filled'] > 0:
                # Order executed successfully
                filled_qty = order['filled'] if order['filled'] > 0 else quantity
                avg_price = order['average'] if order['average'] else price
                actual_cost = filled_qty * avg_price
                
                # Update local tracking
                if symbol in self.positions:
                    current_qty = self.positions[symbol]['quantity']
                    current_avg = self.positions[symbol]['avg_price']
                    new_avg = ((current_qty * current_avg) + (filled_qty * avg_price)) / (current_qty + filled_qty)
                    
                    self.positions[symbol] = {
                        'quantity': current_qty + filled_qty,
                        'avg_price': new_avg
                    }
                else:
                    self.positions[symbol] = {
                        'quantity': filled_qty,
                        'avg_price': avg_price
                    }
                
                # Update cash balance
                self.cash_balance -= actual_cost
                self.total_fees += (actual_cost * 0.0025)  # 0.25% maker fee for limit orders
                
                # Record trade
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': 'buy',
                    'quantity': filled_qty,
                    'price': avg_price,
                    'fee': actual_cost * 0.0025,
                    'total': actual_cost,
                    'order_id': order['id'],
                    'status': order['status']
                }
                self.trades.append(trade)
                
                return True, f"REAL LIMIT BUY: {filled_qty:.6f} {symbol} at ${avg_price:.6f}"
            else:
                return False, f"Limit order placed but not filled yet: {order['id']}"
                
        except Exception as e:
            return False, f"Order failed: {str(e)}"
    
    def sell(self, symbol: str, quantity: float, price: float, fee: float = 0.0):
        """Execute a REAL limit sell order on Gemini"""
        try:
            if symbol not in self.positions:
                return False, "No position in this symbol"
            
            if self.positions[symbol]['quantity'] < quantity:
                return False, "Insufficient quantity to sell"
            
            # Place REAL limit sell order at current market price
            order = self.exchange.create_limit_sell_order(symbol, quantity, price)
            
            if order['status'] in ['closed', 'filled'] or order['filled'] > 0:
                # Order executed successfully
                filled_qty = order['filled'] if order['filled'] > 0 else quantity
                avg_price = order['average'] if order['average'] else price
                revenue = filled_qty * avg_price
                
                # Update position
                self.positions[symbol]['quantity'] -= filled_qty
                
                # Remove position if quantity is zero
                if self.positions[symbol]['quantity'] <= 0:
                    del self.positions[symbol]
                
                # Update cash balance
                fee_amount = revenue * 0.0025  # 0.25% maker fee
                self.cash_balance += revenue - fee_amount
                self.total_fees += fee_amount
                
                # Record trade
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': filled_qty,
                    'price': avg_price,
                    'fee': fee_amount,
                    'total': revenue,
                    'order_id': order['id'],
                    'status': order['status']
                }
                self.trades.append(trade)
                
                return True, f"REAL LIMIT SELL: {filled_qty:.6f} {symbol} at ${avg_price:.6f}"
            else:
                return False, f"Limit order placed but not filled yet: {order['id']}"
                
        except Exception as e:
            return False, f"Order failed: {str(e)}"
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value using real balance + positions"""
        # Refresh real balance
        self.get_real_balance()
        
        total_value = self.cash_balance
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value = position['quantity'] * current_prices[symbol]
                total_value += position_value
        
        return total_value
    
    def get_portfolio_summary(self, current_prices: Dict[str, float] = None):
        """Get portfolio summary with real data"""
        # Refresh real balance
        self.get_real_balance()
        
        summary = {
            'cash_balance': self.cash_balance,
            'positions': self.positions.copy(),
            'total_trades': len(self.trades),
            'total_fees': self.total_fees
        }
        
        if current_prices:
            summary['portfolio_value'] = self.get_portfolio_value(current_prices)
            summary['total_pnl'] = summary['portfolio_value'] - self.initial_balance
            summary['pnl_percentage'] = (summary['total_pnl'] / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        return summary
    
    def print_portfolio(self, current_prices: Dict[str, float] = None):
        """Print formatted portfolio summary with real data"""
        summary = self.get_portfolio_summary(current_prices)
        
        print("\n" + "="*50)
        print("🏦 REAL GEMINI PORTFOLIO")
        print("="*50)
        print(f"💵 Cash Balance: ${summary['cash_balance']:.2f}")
        
        if summary['positions']:
            print(f"\n📊 Positions:")
            for symbol, pos in summary['positions'].items():
                current_price = current_prices.get(symbol, 0) if current_prices else 0
                current_value = pos['quantity'] * current_price if current_price else 0
                cost_basis = pos['quantity'] * pos['avg_price']
                pnl = current_value - cost_basis if current_price else 0
                
                print(f"  {symbol}: {pos['quantity']:.6f} @ ${pos['avg_price']:.6f}")
                if current_price:
                    print(f"    Current: ${current_price:.6f} | Value: ${current_value:.2f} | P&L: ${pnl:.2f}")
        
        if current_prices and 'portfolio_value' in summary:
            print(f"\n💰 Total Portfolio Value: ${summary['portfolio_value']:,.2f}")
            print(f"📈 Total P&L: ${summary['total_pnl']:,.2f} ({summary['pnl_percentage']:.2f}%)")
        
        print(f"🔄 Total Trades: {summary['total_trades']}")
        print(f"💸 Total Fees: ${summary['total_fees']:.2f} (0.25% limit orders)")
        print("="*50)

# Backward compatibility
PortfolioManager = RealGeminiPortfolioManager

if __name__ == "__main__":
    # Test the real portfolio manager
    try:
        portfolio = RealGeminiPortfolioManager()
        print("✅ Real portfolio manager with LIMIT ORDERS initialized!")
        portfolio.print_portfolio()
    except Exception as e:
        print(f"❌ Error: {e}")
