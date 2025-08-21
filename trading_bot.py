#!/usr/bin/env python3
"""
High-Performance AI Crypto Trading Bot - Enhanced Version
Advanced Position Sizing & Partial Selling Logic
"""

import time
import csv
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.dirname(__file__))

from src.data.collector import GeminiDataCollector
from src.portfolio.manager import RealGeminiPortfolioManager
from src.trading.strategy import AdvancedTradingStrategy
from config.trading_symbols import SYMBOLS, MARKET_CONDITION
from config.config import (
    BASE_POSITION_SIZE, MAX_POSITION_SIZE, MIN_POSITION_SIZE,
    MIN_CONFIDENCE_THRESHOLD, MAX_DAILY_TRADES, MAX_POSITIONS,
    TRADE_COOLDOWN, MAX_DAILY_LOSS, SAVE_TRADES_TO_CSV,
    PARTIAL_SELL_ENABLED, FIRST_SELL_PERCENTAGE, SECOND_SELL_PERCENTAGE,
    DEFAULT_TRADING_FEE as TRADING_FEE
)
from config.risk_management import (
    STOP_LOSS_ENABLED, STOP_LOSS_PERCENTAGE,
    TAKE_PROFIT_ENABLED, TAKE_PROFIT_PERCENTAGE
)

class EnhancedTradingBot:
    def __init__(self):
        """Initialize the enhanced trading bot"""
        self.data_collector = GeminiDataCollector()
        self.portfolio = RealGeminiPortfolioManager()
        self.strategy = AdvancedTradingStrategy()
        self.is_running = False
        self.cycle_count = 0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.start_time = datetime.now()
        self.last_trade_times = {}
        self.performance_log = []
        
        # Enhanced tracking
        self.sell_signals = {}  # Track sell signals per symbol
        self.position_entries = {}  # Track entry details for partial selling
        self.confidence_history = {}  # Track confidence trends
        self.stop_losses = {}  # Track stop-loss levels per position
        self.take_profits = {}  # Track take-profit levels per position
        
        # Performance tracking for dynamic sizing
        self.recent_trades_performance = []
        self.win_rate = 0.5  # Start neutral
        
        # Get initial balance from real account
        self.initial_balance = self.portfolio.initial_balance
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        print("🚀 ENHANCED AI CRYPTO TRADING BOT")
        print("="*60)
        print(f"💰 Real Account Balance: ${self.initial_balance:.2f}")
        print(f"🎯 Strategy: {MARKET_CONDITION.upper()} ({len(SYMBOLS)} symbols)")
        print(f"⚡ Dynamic Position Sizing: ${MIN_POSITION_SIZE}-${MAX_POSITION_SIZE}")
        print(f"🔄 Partial Selling: {PARTIAL_SELL_ENABLED}")
        print(f"📊 Confidence-Based Logic: ENABLED")
        print(f"🛡️ Stop-Loss: {'ENABLED' if STOP_LOSS_ENABLED else 'DISABLED'} ({STOP_LOSS_PERCENTAGE*100:.1f}%)")
        print(f"🎯 Take-Profit: {'ENABLED' if TAKE_PROFIT_ENABLED else 'DISABLED'} ({TAKE_PROFIT_PERCENTAGE*100:.1f}%)")
        print(f"🚀 Small Cap Focus: HIGH RISK, HIGH REWARD!")
        
    def calculate_dynamic_position_size(self, confidence, symbol, current_price):
        """Calculate position size based on confidence and performance"""
        
        # Base calculation using confidence
        confidence_multiplier = self._get_confidence_multiplier(confidence)
        base_amount = BASE_POSITION_SIZE * confidence_multiplier
        
        # Adjust based on recent performance
        performance_multiplier = self._get_performance_multiplier()
        adjusted_amount = base_amount * performance_multiplier
        
        # Apply min/max limits
        final_amount = max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, adjusted_amount))
        
        # Ensure we don't exceed available cash
        available_cash = self.portfolio.cash_balance * 0.9  # Keep 10% reserve
        final_amount = min(final_amount, available_cash)
        
        # Calculate quantity
        position_quantity = final_amount / current_price
        
        print(f"   💡 Position Sizing for {symbol}:")
        print(f"      Confidence: {confidence:.2f} → Multiplier: {confidence_multiplier:.2f}")
        print(f"      Performance: {self.win_rate:.2f} → Multiplier: {performance_multiplier:.2f}")
        print(f"      Final Amount: ${final_amount:.2f}")
        
        return position_quantity, final_amount
    
    def _get_confidence_multiplier(self, confidence):
        """Convert confidence to position size multiplier"""
        # Linear scaling: 0.25 conf = 0.6x, 0.50 conf = 1.0x, 0.75+ conf = 1.4x
        if confidence <= 0.25:
            return 0.6
        elif confidence <= 0.35:
            return 0.8
        elif confidence <= 0.50:
            return 1.0
        elif confidence <= 0.65:
            return 1.2
        else:
            return 1.4  # High confidence = bigger positions
    
    def _get_performance_multiplier(self):
        """Adjust position size based on recent performance"""
        if self.win_rate >= 0.7:
            return 1.2  # Winning streak = bigger positions
        elif self.win_rate >= 0.6:
            return 1.1
        elif self.win_rate >= 0.4:
            return 1.0  # Neutral
        elif self.win_rate >= 0.3:
            return 0.9
        else:
            return 0.8  # Losing streak = smaller positions
    
    def determine_sell_quantity(self, symbol, confidence):
        """Determine how much to sell based on partial selling logic"""
        if not PARTIAL_SELL_ENABLED:
            # Original logic: sell everything
            return self.portfolio.positions[symbol]['quantity'], "Complete exit"
        
        # Track sell signals
        if symbol not in self.sell_signals:
            self.sell_signals[symbol] = 0
        
        self.sell_signals[symbol] += 1
        total_quantity = self.portfolio.positions[symbol]['quantity']
        
        if self.sell_signals[symbol] == 1:
            # First sell signal: sell percentage based on confidence
            if confidence >= 0.6:
                sell_percentage = 0.7  # High confidence = sell more
            elif confidence >= 0.4:
                sell_percentage = 0.5  # Medium confidence = sell half
            else:
                sell_percentage = 0.3  # Low confidence = sell less
            
            sell_quantity = total_quantity * sell_percentage
            reason = f"Partial exit #{self.sell_signals[symbol]} ({sell_percentage*100:.0f}%)"
            
        else:
            # Second+ sell signal: sell remaining position
            sell_quantity = total_quantity
            reason = f"Complete exit (signal #{self.sell_signals[symbol]})"
            # Reset sell signal counter
            self.sell_signals[symbol] = 0
        
        return sell_quantity, reason
    
    def update_performance_tracking(self, trade_result):
        """Update performance metrics for dynamic sizing"""
        self.recent_trades_performance.append(trade_result)
        
        # Keep only last 20 trades for calculation
        if len(self.recent_trades_performance) > 20:
            self.recent_trades_performance.pop(0)
        
        # Calculate win rate
        if len(self.recent_trades_performance) >= 5:
            wins = sum(1 for result in self.recent_trades_performance if result > 0)
            self.win_rate = wins / len(self.recent_trades_performance)
        
    def test_connections(self):
        """Test all connections before starting"""
        print("\n🔍 Testing connections...")
        return self.data_collector.test_connection()
    
    def get_current_prices(self):
        """Get current prices for all symbols"""
        prices = {}
        
        for symbol in SYMBOLS:
            try:
                price_data = self.data_collector.get_current_price(symbol)
                if price_data:
                    prices[symbol] = price_data['price']
            except Exception as e:
                print(f"Price error for {symbol}: {e}")
                continue
                
        return prices
    
    def analyze_symbol(self, symbol):
        """Analyze a single symbol and generate trading signal"""
        try:
            # Check cooldown
            if symbol in self.last_trade_times:
                time_since_last = (datetime.now() - self.last_trade_times[symbol]).seconds
                if time_since_last < TRADE_COOLDOWN:
                    return 'hold', 0.0, f'Cooldown: {TRADE_COOLDOWN - time_since_last}s remaining'
            
            # Get historical data
            df = self.data_collector.get_historical_data(symbol, '5m', 100)
            if df is None:
                return None, None, "No data available"
            
            # Generate signal
            signal, confidence, reason = self.strategy.generate_signal(df, symbol, self.portfolio.cash_balance)
            
            # Enhanced fee-aware trading with dynamic thresholds
            if signal in ['buy', 'sell'] and confidence > 0:
                # Lower threshold for high-performing bot
                performance_bonus = (self.win_rate - 0.5) * 0.005  # Up to 0.0025 bonus
                min_profit_needed = (TRADING_FEE * 2) + 0.002 - performance_bonus
                
                if confidence < min_profit_needed:
                    return 'hold', confidence, f"Profit potential ({confidence:.3f}) < fees needed ({min_profit_needed:.3f})"
            
            # Track confidence history
            if symbol not in self.confidence_history:
                self.confidence_history[symbol] = []
            self.confidence_history[symbol].append(confidence)
            if len(self.confidence_history[symbol]) > 10:
                self.confidence_history[symbol].pop(0)
            
            return signal, confidence, reason
            
        except Exception as e:
            return None, None, f"Analysis error: {str(e)}"
    
    def execute_trade(self, symbol, signal, confidence, current_price):
        """Execute enhanced trades with dynamic sizing"""
        if self.daily_trades >= MAX_DAILY_TRADES:
            print(f"   ⛔ Daily trade limit reached ({MAX_DAILY_TRADES})")
            return False
            
        if signal == 'buy' and confidence >= MIN_CONFIDENCE_THRESHOLD:
            # Check if we already have a position in this symbol
            if symbol in self.portfolio.positions:
                print(f"   ⚠️ Already holding position in {symbol}")
                return False
            
            # Check position limits
            if len(self.portfolio.positions) >= MAX_POSITIONS:
                print(f"   ⛔ Max positions reached ({MAX_POSITIONS})")
                return False
            
            # Enhanced position sizing
            position_size, investment_amount = self.calculate_dynamic_position_size(
                confidence, symbol, current_price
            )
            
            cost = position_size * current_price
            fee = cost * TRADING_FEE
            
            if cost + fee <= self.portfolio.cash_balance:
                success, msg = self.portfolio.buy(symbol, position_size, current_price, fee)
                if success:
                    self.daily_trades += 1
                    self.last_trade_times[symbol] = datetime.now()
                    
                    # Store entry details for partial selling
                    self.position_entries[symbol] = {
                        'entry_price': current_price,
                        'entry_confidence': confidence,
                        'entry_time': datetime.now()
                    }
                    
                    # Set stop-loss and take-profit levels
                    if STOP_LOSS_ENABLED:
                        self.stop_losses[symbol] = current_price * (1 - STOP_LOSS_PERCENTAGE)
                        print(f"   🛡️ Stop-Loss: ${self.stop_losses[symbol]:.6f} (-{STOP_LOSS_PERCENTAGE*100:.1f}%)")
                    
                    if TAKE_PROFIT_ENABLED:
                        self.take_profits[symbol] = current_price * (1 + TAKE_PROFIT_PERCENTAGE)
                        print(f"   🎯 Take-Profit: ${self.take_profits[symbol]:.6f} (+{TAKE_PROFIT_PERCENTAGE*100:.1f}%)")
                    
                    # Reset sell signals for this symbol
                    self.sell_signals[symbol] = 0
                    
                    print(f"🟢 ENHANCED BUY: {symbol}")
                    print(f"   💰 Amount: ${cost:.2f} (confidence-based)")
                    print(f"   📊 Confidence: {confidence:.2f}")
                    print(f"   🎯 Quantity: {position_size:.6f}")
                    
                    self._log_trade('BUY', symbol, current_price, position_size, confidence, msg, investment_amount)
                    return True
                else:
                    print(f"❌ BUY FAILED: {msg}")
                    
        elif signal == 'sell' and symbol in self.portfolio.positions:
            # Enhanced partial selling
            sell_quantity, sell_reason = self.determine_sell_quantity(symbol, confidence)
            
            revenue = sell_quantity * current_price
            fee = revenue * TRADING_FEE
            
            success, msg = self.portfolio.sell(symbol, sell_quantity, current_price, fee)
            if success:
                self.daily_trades += 1
                self.last_trade_times[symbol] = datetime.now()
                
                # Calculate P&L for this partial sale
                entry_details = self.position_entries.get(symbol, {})
                entry_price = entry_details.get('entry_price', current_price)
                pnl = (current_price - entry_price) * sell_quantity - fee
                
                self.daily_pnl += pnl
                self.update_performance_tracking(pnl)
                
                print(f"🔴 ENHANCED SELL: {symbol}")
                print(f"   💰 Revenue: ${revenue:.2f}")
                print(f"   📊 Reason: {sell_reason}")
                print(f"   💹 P&L: ${pnl:+.2f}")
                
                # If complete exit, clean up tracking
                if sell_quantity == self.portfolio.positions.get(symbol, {}).get('quantity', 0):
                    if symbol in self.position_entries:
                        del self.position_entries[symbol]
                    if symbol in self.stop_losses:
                        del self.stop_losses[symbol]
                    if symbol in self.take_profits:
                        del self.take_profits[symbol]
                
                self._log_trade('SELL', symbol, current_price, sell_quantity, confidence, msg, revenue)
                return True
            else:
                print(f"❌ SELL FAILED: {msg}")
        
        return False
    
    def _log_trade(self, action, symbol, price, quantity, confidence, message, amount):
        """Enhanced trade logging"""
        if not SAVE_TRADES_TO_CSV:
            return
            
        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'amount': amount,
            'confidence': confidence,
            'fee_rate': TRADING_FEE,
            'order_type': 'LIMIT',
            'strategy': MARKET_CONDITION,
            'win_rate': self.win_rate,
            'position_sizing': 'DYNAMIC',
            'message': message
        }
        
        csv_file = 'logs/trades.csv'
        file_exists = os.path.isfile(csv_file)
        
        try:
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trade_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(trade_data)
        except Exception as e:
            print(f"Logging error: {e}")
    
    def check_stop_loss_take_profit(self, current_prices):
        """Check and execute stop-loss or take-profit orders"""
        positions_to_sell = []
        
        for symbol in list(self.portfolio.positions.keys()):
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            # Check stop-loss
            if symbol in self.stop_losses and current_price <= self.stop_losses[symbol]:
                positions_to_sell.append((symbol, 'stop_loss', current_price))
                print(f"\n🛑 STOP-LOSS TRIGGERED: {symbol}")
                print(f"   Current: ${current_price:.6f} | Stop: ${self.stop_losses[symbol]:.6f}")
            
            # Check take-profit
            elif symbol in self.take_profits and current_price >= self.take_profits[symbol]:
                positions_to_sell.append((symbol, 'take_profit', current_price))
                print(f"\n🎯 TAKE-PROFIT TRIGGERED: {symbol}")
                print(f"   Current: ${current_price:.6f} | Target: ${self.take_profits[symbol]:.6f}")
        
        # Execute stop-loss/take-profit sells
        for symbol, reason, price in positions_to_sell:
            quantity = self.portfolio.positions[symbol]['quantity']
            revenue = quantity * price
            fee = revenue * TRADING_FEE
            
            success, msg = self.portfolio.sell(symbol, quantity, price, fee)
            if success:
                self.daily_trades += 1
                
                # Calculate P&L
                entry_price = self.position_entries.get(symbol, {}).get('entry_price', price)
                pnl = (price - entry_price) * quantity - fee
                self.daily_pnl += pnl
                
                print(f"   💰 Sold: {quantity:.6f} @ ${price:.6f}")
                print(f"   💹 P&L: ${pnl:+.2f}")
                
                # Clean up tracking
                if symbol in self.position_entries:
                    del self.position_entries[symbol]
                if symbol in self.stop_losses:
                    del self.stop_losses[symbol]
                if symbol in self.take_profits:
                    del self.take_profits[symbol]
                if symbol in self.sell_signals:
                    del self.sell_signals[symbol]
                
                self._log_trade('SELL', symbol, price, quantity, 0, f"{reason.upper()}: {msg}", revenue)
    
    def run_trading_cycle(self):
        """Run enhanced trading cycle"""
        self.cycle_count += 1
        
        print(f"\n{'='*70}")
        print(f"⚡ ENHANCED CYCLE #{self.cycle_count} - {datetime.now().strftime('%H:%M:%S')} - Day {(datetime.now() - self.start_time).days + 1}")
        print(f"{'='*70}")
        
        # Get current prices
        current_prices = self.get_current_prices()
        if not current_prices:
            print("❌ Could not get current prices")
            return
        
        # Check stop-loss and take-profit levels first
        self.check_stop_loss_take_profit(current_prices)
        
        # Show enhanced performance metrics
        current_portfolio_value = self.portfolio.get_portfolio_value(current_prices)
        total_return = ((current_portfolio_value - self.initial_balance) / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        print(f"💰 Portfolio: ${current_portfolio_value:.2f} ({total_return:+.2f}%)")
        print(f"💵 Cash: ${self.portfolio.cash_balance:.2f}")
        print(f"📊 Positions: {len(self.portfolio.positions)}")
        print(f"🔄 Daily Trades: {self.daily_trades}/{MAX_DAILY_TRADES}")
        print(f"📈 Daily P&L: ${self.daily_pnl:+.2f}")
        print(f"🎯 Win Rate: {self.win_rate:.1%} (last {len(self.recent_trades_performance)} trades)")
        print(f"🎪 Strategy: {MARKET_CONDITION.upper()} - Enhanced Logic")
        
        # Analyze and trade
        trades_executed = 0
        signals_found = []
        
        print(f"\n📊 ANALYZING {len(SYMBOLS)} SYMBOLS (Enhanced Logic)...")
        
        for symbol in SYMBOLS:
            if symbol not in current_prices:
                print(f"   {symbol}: No price data")
                continue
                
            current_price = current_prices[symbol]
            signal, confidence, reason = self.analyze_symbol(symbol)
            
            if signal and confidence is not None:
                if signal in ['buy', 'sell'] and confidence >= MIN_CONFIDENCE_THRESHOLD:
                    signals_found.append((symbol, signal, confidence, reason, current_price))
                    print(f"   🎯 {symbol}: {signal.upper()} | Conf: {confidence:.2f} | ${current_price:,.6f}")
                else:
                    print(f"   📊 {symbol}: {signal} | Conf: {confidence:.2f} | ${current_price:,.6f}")
            else:
                print(f"   ❌ {symbol}: Analysis failed | ${current_price:,.6f}")
        
        # Execute trades (sort by confidence)
        signals_found.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\n🚀 EXECUTING ENHANCED TRADES...")
        for symbol, signal, confidence, reason, current_price in signals_found[:5]:
            print(f"\n🎯 {symbol}: {signal.upper()} (Conf: {confidence:.2f})")
            print(f"   💲 Price: ${current_price:,.6f}")
            print(f"   📝 Reason: {reason}")
            
            if self.execute_trade(symbol, signal, confidence, current_price):
                trades_executed += 1
        
        print(f"\n⚡ Enhanced cycle completed | Trades executed: {trades_executed}")
    
    def run(self, duration_hours=2, cycle_delay=120, max_retries=3):
        """Run the enhanced trading bot with error recovery"""
        if not self.test_connections():
            print("❌ Cannot start without exchange connection")
            return
        
        print(f"\n🚀 STARTING ENHANCED SMALL CAP TRADING...")
        print(f"⏰ Duration: {duration_hours} hours")
        print(f"🕐 Cycle frequency: {cycle_delay} seconds")
        print(f"🎯 Target: Maximum profit with ${self.initial_balance}")
        print(f"💸 Dynamic position sizing: ${MIN_POSITION_SIZE}-${MAX_POSITION_SIZE}")
        print(f"🔄 Partial selling: {'ENABLED' if PARTIAL_SELL_ENABLED else 'DISABLED'}")
        print(f"📈 Strategy: {MARKET_CONDITION.upper()} with enhanced logic")
        print(f"🛡️ Error recovery: ENABLED (max {max_retries} retries)")
        print(f"\n💡 Press Ctrl+C anytime to stop and see results")
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        self.is_running = True
        consecutive_errors = 0
        
        try:
            while self.is_running and datetime.now() < end_time:
                try:
                    cycle_start = time.time()
                    self.run_trading_cycle()
                    cycle_duration = time.time() - cycle_start
                    
                    # Reset error counter on successful cycle
                    consecutive_errors = 0
                    
                    sleep_time = max(20, cycle_delay - int(cycle_duration))
                    
                    for i in range(sleep_time, 0, -10):
                        if not self.is_running:
                            break
                        print(f"💤 Next cycle in {i}s... (Press Ctrl+C to stop)", end='\r')
                        time.sleep(min(10, i))
                    
                    print(" " * 50, end='\r')
                    
                except KeyboardInterrupt:
                    raise  # Re-raise to handle at outer level
                    
                except Exception as e:
                    consecutive_errors += 1
                    print(f"\n⚠️ Error in trading cycle (attempt {consecutive_errors}/{max_retries}): {e}")
                    
                    if consecutive_errors >= max_retries:
                        print(f"\n❌ Max retries ({max_retries}) exceeded. Stopping bot.")
                        self.is_running = False
                        break
                    
                    # Exponential backoff for retry
                    retry_delay = min(300, cycle_delay * (2 ** (consecutive_errors - 1)))
                    print(f"🔄 Retrying in {retry_delay} seconds...")
                    
                    # Try to recover state
                    try:
                        print("🔧 Attempting to recover state...")
                        if not self.test_connections():
                            print("📡 Reconnecting to exchange...")
                            self.data_collector = GeminiDataCollector()
                            self.portfolio = RealGeminiPortfolioManager()
                        print("✅ State recovery successful")
                    except Exception as recovery_error:
                        print(f"❌ State recovery failed: {recovery_error}")
                    
                    time.sleep(retry_delay)
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 Trading stopped by user")
        except Exception as e:
            print(f"\n❌ Critical trading error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            self._print_final_summary()
    
    def _print_final_summary(self):
        """Print enhanced final summary"""
        try:
            final_prices = self.get_current_prices()
            final_value = self.portfolio.get_portfolio_value(final_prices) if final_prices else self.portfolio.cash_balance
            total_return = ((final_value - self.initial_balance) / self.initial_balance) * 100 if self.initial_balance > 0 else 0
            duration = datetime.now() - self.start_time
            
            print(f"\n{'='*70}")
            print(f"🏁 ENHANCED TRADING RESULTS")
            print(f"{'='*70}")
            print(f"💰 Starting Capital: ${self.initial_balance:,.2f}")
            print(f"💰 Final Value: ${final_value:,.2f}")
            print(f"📈 Total Return: {total_return:+.2f}%")
            print(f"💵 Cash Balance: ${self.portfolio.cash_balance:.2f}")
            print(f"🔄 Total Trades: {len(self.portfolio.trades)}")
            print(f"💸 Total Fees: ${self.portfolio.total_fees:.2f}")
            print(f"🎯 Win Rate: {self.win_rate:.1%}")
            print(f"⏰ Trading Duration: {duration}")
            
            print(f"\n🚀 ENHANCED FEATURES PERFORMANCE:")
            print(f"   📊 Dynamic Position Sizing: ${MIN_POSITION_SIZE}-${MAX_POSITION_SIZE}")
            print(f"   🔄 Partial Selling: {'ACTIVE' if PARTIAL_SELL_ENABLED else 'DISABLED'}")
            print(f"   🎯 Performance Adaptation: {self.win_rate:.1%} win rate")
            print(f"   📈 Confidence-Based Sizing: ACTIVE")
            
            if self.portfolio.positions:
                print(f"\n📊 OPEN POSITIONS:")
                for symbol, pos in self.portfolio.positions.items():
                    current_price = final_prices.get(symbol, 0) if final_prices else 0
                    value = pos['quantity'] * current_price if current_price else 0
                    cost_basis = pos['quantity'] * pos['avg_price']
                    pnl = value - cost_basis if current_price else 0
                    
                    # Show partial sell status
                    sell_signals = self.sell_signals.get(symbol, 0)
                    status = f" ({sell_signals} sell signals)" if sell_signals > 0 else ""
                    
                    print(f"   {symbol}: {pos['quantity']:.6f} @ ${pos['avg_price']:.6f} | "
                          f"Current: ${current_price:.6f} | P&L: ${pnl:+.2f}{status}")
            
            print(f"\n📄 Enhanced trade log: logs/trades.csv")
            
        except Exception as e:
            print(f"Error in final summary: {e}")

def main():
    """Main function"""
    print("🚀 ENHANCED AI CRYPTO TRADING BOT")
    print("💡 Dynamic Position Sizing & Partial Selling")
    print("🎯 Confidence-Based Intelligence")
    print("="*80)
    
    bot = EnhancedTradingBot()
    bot.run(duration_hours=2, cycle_delay=120)

if __name__ == "__main__":
    main()
