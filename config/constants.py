"""
Constants configuration - Remove magic numbers
"""

import sys
import os

# Add src to path for YAML loader
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Timing Constants (seconds)
DEFAULT_CYCLE_DELAY = 120  # 2 minutes between trading cycles
MIN_SLEEP_TIME = 20  # Minimum sleep between cycles
SLEEP_COUNTDOWN_INTERVAL = 10  # Update countdown every 10 seconds
DEFAULT_TRADING_DURATION_HOURS = 2  # Default bot run duration

# Retry Constants
DEFAULT_MAX_RETRIES = 3
MAX_RETRY_DELAY = 300  # 5 minutes max retry delay
RETRY_BACKOFF_BASE = 2  # Exponential backoff base

# Display Constants
TERMINAL_WIDTH = 70  # Width for separator lines
CLEAR_LINE_SPACES = 50  # Spaces to clear terminal line

# Performance Constants
MIN_DATA_POINTS = 10  # Minimum data points for analysis
PERFORMANCE_WINDOW_SIZE = 20  # Trades to track for performance
MIN_TRADES_FOR_WIN_RATE = 5  # Minimum trades to calculate win rate

# File Constants
CSV_BUFFER_SIZE = 1  # Line buffering for CSV
MAX_SYMBOL_ANALYSIS = 5  # Max symbols to trade per cycle

# Percentage Constants
CASH_RESERVE_PERCENTAGE = 0.1  # Keep 10% cash reserve
FEE_PROFIT_BUFFER = 0.002  # Additional profit needed above fees
PERFORMANCE_BONUS_FACTOR = 0.005  # Performance adjustment factor

# Validation Constants
MIN_CONFIDENCE_FOR_TRADE = 0.25
MAX_POSITION_LIMIT = 4
MAX_DAILY_TRADE_LIMIT = 50

# Apply YAML overrides if available
try:
    from src.config.yaml_loader import get_yaml_loader
    yaml_loader = get_yaml_loader()
    updated_count = yaml_loader.update_module_config(globals(), 'constants')
    if updated_count > 0:
        print(f"✅ Applied {updated_count} YAML constants overrides")
except ImportError:
    pass  # YAML not available
except Exception as e:
    print(f"⚠️ Warning: Failed to load YAML constants overrides: {e}")