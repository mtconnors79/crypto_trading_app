"""
Centralized logging configuration for the trading bot
"""

import logging
import logging.handlers
import os
from datetime import datetime
from config.config import LOG_LEVEL, LOG_FORMAT, IS_DEVELOPMENT

# Create logs directory if it doesn't exist
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name, log_file=None, level=None):
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name (usually __name__ of the module)
        log_file: Optional specific log file name
        level: Optional log level override
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Use provided level or default from config
    log_level = level or getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = f"{name.replace('.', '_')}.log"
    
    file_path = os.path.join(LOG_DIR, log_file)
    
    # Use rotating file handler to prevent huge log files
    file_handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add timestamp to log file daily
    if not IS_DEVELOPMENT:
        daily_file = f"{name.replace('.', '_')}_{datetime.now().strftime('%Y%m%d')}.log"
        daily_path = os.path.join(LOG_DIR, daily_file)
        daily_handler = logging.handlers.TimedRotatingFileHandler(
            daily_path,
            when='midnight',
            interval=1,
            backupCount=30  # Keep 30 days of logs
        )
        daily_handler.setLevel(log_level)
        daily_handler.setFormatter(formatter)
        logger.addHandler(daily_handler)
    
    return logger

# Pre-configured loggers for main components
def get_trading_logger():
    """Get logger for trading operations"""
    return setup_logger('trading', 'trading.log')

def get_portfolio_logger():
    """Get logger for portfolio management"""
    return setup_logger('portfolio', 'portfolio.log')

def get_data_logger():
    """Get logger for data collection"""
    return setup_logger('data', 'data_collection.log')

def get_risk_logger():
    """Get logger for risk management"""
    return setup_logger('risk', 'risk_management.log')

def get_error_logger():
    """Get logger for errors only"""
    logger = setup_logger('errors', 'errors.log', level=logging.ERROR)
    return logger

# Decorator for logging function calls
def log_function_call(logger=None):
    """
    Decorator to log function entry and exit
    
    Args:
        logger: Logger instance to use (optional)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger(func.__module__)
            log.debug(f"Entering {func.__name__}")
            try:
                result = func(*args, **kwargs)
                log.debug(f"Exiting {func.__name__} successfully")
                return result
            except Exception as e:
                log.error(f"Error in {func.__name__}: {e}", exc_info=True)
                raise
        return wrapper
    return decorator

# Trade logging specifically
class TradeLogger:
    """Specialized logger for trade operations"""
    
    def __init__(self):
        self.logger = setup_logger('trades', 'trades.log')
    
    def log_buy(self, symbol, quantity, price, confidence, amount):
        """Log a buy trade"""
        self.logger.info(
            f"BUY | {symbol} | Qty: {quantity:.6f} | "
            f"Price: ${price:.6f} | Conf: {confidence:.2f} | "
            f"Amount: ${amount:.2f}"
        )
    
    def log_sell(self, symbol, quantity, price, reason, pnl):
        """Log a sell trade"""
        self.logger.info(
            f"SELL | {symbol} | Qty: {quantity:.6f} | "
            f"Price: ${price:.6f} | Reason: {reason} | "
            f"P&L: ${pnl:+.2f}"
        )
    
    def log_stop_loss(self, symbol, price, loss):
        """Log a stop-loss trigger"""
        self.logger.warning(
            f"STOP-LOSS | {symbol} | Price: ${price:.6f} | "
            f"Loss: ${loss:.2f}"
        )
    
    def log_take_profit(self, symbol, price, profit):
        """Log a take-profit trigger"""
        self.logger.info(
            f"TAKE-PROFIT | {symbol} | Price: ${price:.6f} | "
            f"Profit: ${profit:.2f}"
        )

# Performance logger
class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self):
        self.logger = setup_logger('performance', 'performance.log')
    
    def log_cycle(self, cycle_num, portfolio_value, positions, daily_pnl):
        """Log trading cycle metrics"""
        self.logger.info(
            f"Cycle #{cycle_num} | Portfolio: ${portfolio_value:.2f} | "
            f"Positions: {positions} | Daily P&L: ${daily_pnl:+.2f}"
        )
    
    def log_summary(self, initial, final, total_return, trades, duration):
        """Log final summary"""
        self.logger.info(
            f"SUMMARY | Initial: ${initial:.2f} | Final: ${final:.2f} | "
            f"Return: {total_return:+.2f}% | Trades: {trades} | Duration: {duration}"
        )