"""
Risk Management Configuration
Centralized settings for stop-loss, take-profit, and risk controls
"""

class RiskParameterError(Exception):
    """Exception raised for invalid risk parameters"""
    pass

def validate_percentage(value, name, min_val=0.0, max_val=1.0):
    """Validate a percentage value is within acceptable range"""
    if not isinstance(value, (int, float)):
        raise RiskParameterError(f"{name} must be a number, got {type(value)}")
    if not min_val <= value <= max_val:
        raise RiskParameterError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return value

def validate_risk_parameters():
    """Validate all risk parameters on module load"""
    errors = []
    
    # Validate stop-loss
    try:
        validate_percentage(STOP_LOSS_PERCENTAGE, "STOP_LOSS_PERCENTAGE", 0.001, 0.5)
    except RiskParameterError as e:
        errors.append(str(e))
    
    # Validate take-profit
    try:
        validate_percentage(TAKE_PROFIT_PERCENTAGE, "TAKE_PROFIT_PERCENTAGE", 0.001, 2.0)
    except RiskParameterError as e:
        errors.append(str(e))
    
    # Validate stop-loss < take-profit when both enabled
    if STOP_LOSS_ENABLED and TAKE_PROFIT_ENABLED:
        if STOP_LOSS_PERCENTAGE >= TAKE_PROFIT_PERCENTAGE:
            errors.append(f"STOP_LOSS_PERCENTAGE ({STOP_LOSS_PERCENTAGE}) must be less than TAKE_PROFIT_PERCENTAGE ({TAKE_PROFIT_PERCENTAGE})")
    
    # Validate position risk
    try:
        validate_percentage(MAX_POSITION_RISK, "MAX_POSITION_RISK", 0.001, 0.1)
        validate_percentage(MAX_PORTFOLIO_RISK, "MAX_PORTFOLIO_RISK", 0.01, 0.2)
    except RiskParameterError as e:
        errors.append(str(e))
    
    # Validate daily loss
    try:
        validate_percentage(MAX_DAILY_LOSS_PERCENTAGE, "MAX_DAILY_LOSS_PERCENTAGE", 0.001, 0.1)
    except RiskParameterError as e:
        errors.append(str(e))
    
    if MAX_DAILY_LOSS_USD <= 0:
        errors.append(f"MAX_DAILY_LOSS_USD must be positive, got {MAX_DAILY_LOSS_USD}")
    
    # Validate drawdown
    try:
        validate_percentage(MAX_DRAWDOWN_PERCENTAGE, "MAX_DRAWDOWN_PERCENTAGE", 0.01, 0.5)
    except RiskParameterError as e:
        errors.append(str(e))
    
    if errors:
        error_msg = "Risk parameter validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise RiskParameterError(error_msg)

# Stop-Loss Settings
STOP_LOSS_ENABLED = True
STOP_LOSS_PERCENTAGE = 0.05  # 5% stop-loss
TRAILING_STOP_LOSS = False  # Future feature
TRAILING_STOP_PERCENTAGE = 0.03  # 3% trailing stop

# Take-Profit Settings
TAKE_PROFIT_ENABLED = True
TAKE_PROFIT_PERCENTAGE = 0.10  # 10% take-profit
PARTIAL_TAKE_PROFIT = True  # Take partial profits at multiple levels
TAKE_PROFIT_LEVELS = [
    (0.05, 0.3),  # At 5% profit, sell 30%
    (0.10, 0.5),  # At 10% profit, sell 50%
    (0.15, 1.0),  # At 15% profit, sell remaining
]

# Position Risk Management
MAX_POSITION_RISK = 0.02  # Max 2% portfolio risk per position
MAX_PORTFOLIO_RISK = 0.06  # Max 6% total portfolio risk
POSITION_SIZE_BY_RISK = True  # Size positions based on risk

# Daily Risk Limits
MAX_DAILY_LOSS_PERCENTAGE = 0.03  # 3% max daily loss
MAX_DAILY_LOSS_USD = 100.0  # $100 max daily loss
STOP_TRADING_ON_DAILY_LIMIT = True  # Stop trading if daily limit hit

# Risk-Based Position Sizing
def calculate_position_size_by_risk(portfolio_value, entry_price, stop_loss_price, max_risk=MAX_POSITION_RISK):
    """
    Calculate position size based on risk management rules
    
    Args:
        portfolio_value: Total portfolio value
        entry_price: Entry price for the position
        stop_loss_price: Stop-loss price
        max_risk: Maximum risk as percentage of portfolio
    
    Returns:
        Quantity to buy based on risk
    """
    risk_amount = portfolio_value * max_risk
    price_risk = abs(entry_price - stop_loss_price)
    
    if price_risk == 0:
        return 0
    
    quantity = risk_amount / price_risk
    return quantity

# Risk Monitoring
MONITOR_DRAWDOWN = True
MAX_DRAWDOWN_PERCENTAGE = 0.10  # 10% max drawdown
PAUSE_ON_DRAWDOWN = True  # Pause trading on max drawdown

# Volatility-Based Adjustments
ADJUST_FOR_VOLATILITY = True
HIGH_VOLATILITY_THRESHOLD = 0.03  # 3% price movement
LOW_VOLATILITY_THRESHOLD = 0.01  # 1% price movement

# Risk Scoring (for future ML integration)
RISK_FACTORS = {
    'market_volatility': 0.25,
    'position_concentration': 0.20,
    'correlation_risk': 0.15,
    'time_of_day': 0.10,
    'news_sentiment': 0.15,
    'technical_indicators': 0.15
}

# Emergency Controls
EMERGENCY_STOP_ALL = False  # Master kill switch
EMERGENCY_LIQUIDATE = False  # Liquidate all positions
MAX_CONSECUTIVE_LOSSES = 5  # Stop after 5 consecutive losses

# Risk Reporting
LOG_RISK_METRICS = True
RISK_REPORT_INTERVAL = 3600  # Report every hour (seconds)

# Position Exit Priority (when reducing risk)
EXIT_PRIORITY = [
    'lowest_profit',  # Exit least profitable first
    'highest_loss',   # Exit biggest losers first
    'oldest_position', # Exit oldest positions first
    'smallest_position' # Exit smallest positions first
]

# Validate all parameters on module load
try:
    validate_risk_parameters()
    print("✅ Risk parameters validated successfully")
except RiskParameterError as e:
    print(f"⚠️ Warning: {e}")
    print("⚠️ Using default values, but parameters may be misconfigured")