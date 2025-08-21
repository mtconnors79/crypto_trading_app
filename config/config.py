"""
Trading Bot Configuration Settings
Central configuration file for all trading bot parameters
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for YAML loader
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

try:
    from src.config.yaml_loader import get_yaml_loader
    _yaml_available = True
except ImportError:
    _yaml_available = False

# API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')

# Trading Parameters
BASE_POSITION_SIZE = 50.0  # Base amount per trade in USD
MAX_POSITION_SIZE = 80.0   # Maximum per trade
MIN_POSITION_SIZE = 20.0   # Minimum per trade
MIN_CONFIDENCE_THRESHOLD = 0.25
MAX_DAILY_TRADES = 50
MAX_POSITIONS = 4
TRADE_COOLDOWN = 90  # seconds between trades on same symbol
MAX_DAILY_LOSS = 15.0  # Maximum daily loss in USD

# Partial Selling Configuration
PARTIAL_SELL_ENABLED = True
FIRST_SELL_PERCENTAGE = 0.5   # Sell 50% on first sell signal
SECOND_SELL_PERCENTAGE = 1.0  # Sell remaining on second signal

# Exchange Fee Structure
GEMINI_MAKER_FEE = 0.0025
GEMINI_TAKER_FEE = 0.0035
BINANCE_MAKER_FEE = 0.001
BINANCE_TAKER_FEE = 0.001
DEFAULT_TRADING_FEE = GEMINI_MAKER_FEE

# Risk Management
STOP_LOSS_PERCENTAGE = 0.05  # 5% stop loss
TAKE_PROFIT_PERCENTAGE = 0.10  # 10% take profit
MAX_PORTFOLIO_RISK = 0.02  # Max 2% portfolio risk per trade

# Data Collection Settings
DEFAULT_TIMEFRAME = '5m'
DEFAULT_LIMIT = 100
ENABLE_RATE_LIMIT = True
REQUEST_TIMEOUT = 10000  # milliseconds

# Logging Configuration
SAVE_TRADES_TO_CSV = True
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Dashboard Settings
DASHBOARD_UPDATE_INTERVAL = 10  # seconds
WEBSOCKET_PING_INTERVAL = 30  # seconds

# Environment
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
IS_PRODUCTION = ENVIRONMENT == 'production'
IS_DEVELOPMENT = ENVIRONMENT == 'development'

# Exchange Selection
DEFAULT_EXCHANGE = 'gemini'  # Options: 'gemini', 'binance', 'kraken'
USE_SANDBOX = not IS_PRODUCTION

# Performance Tracking
TRACK_PERFORMANCE = True
PERFORMANCE_WINDOW = 20  # Number of trades to track for performance metrics

# Strategy Parameters
SMA_SHORT_WINDOW = 5
SMA_LONG_WINDOW = 20
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Backtest Configuration
BACKTEST_INITIAL_BALANCE = 10000
BACKTEST_TRADING_FEE = 0.001

# Notification Settings (for future implementation)
ENABLE_NOTIFICATIONS = False
NOTIFICATION_WEBHOOK_URL = os.getenv('NOTIFICATION_WEBHOOK_URL', '')

# Debug Settings
DEBUG_MODE = IS_DEVELOPMENT
VERBOSE_LOGGING = DEBUG_MODE

# Apply YAML overrides if available
if _yaml_available:
    try:
        yaml_loader = get_yaml_loader()
        updated_count = yaml_loader.update_module_config(globals(), 'trading')
        if updated_count > 0:
            print(f"✅ Applied {updated_count} YAML configuration overrides")
    except Exception as e:
        print(f"⚠️ Warning: Failed to load YAML config overrides: {e}")
        print("⚠️ Continuing with default configuration values")