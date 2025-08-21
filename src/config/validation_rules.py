"""
YAML Configuration Validation Rules
Define validation rules for all configurable parameters
"""

from typing import Dict, Any

# Trading Configuration Validation Rules
TRADING_VALIDATION_RULES: Dict[str, Dict[str, Any]] = {
    'trading.base_position_size': {
        'type': (int, float),
        'min': 1.0,
        'max': 1000.0,
        'required': False
    },
    'trading.max_position_size': {
        'type': (int, float),
        'min': 1.0,
        'max': 10000.0,
        'required': False
    },
    'trading.min_position_size': {
        'type': (int, float),
        'min': 0.1,
        'max': 100.0,
        'required': False
    },
    'trading.min_confidence_threshold': {
        'type': float,
        'min': 0.01,
        'max': 1.0,
        'required': False
    },
    'trading.max_daily_trades': {
        'type': int,
        'min': 1,
        'max': 1000,
        'required': False
    },
    'trading.max_positions': {
        'type': int,
        'min': 1,
        'max': 50,
        'required': False
    },
    'trading.trade_cooldown': {
        'type': int,
        'min': 1,
        'max': 3600,
        'required': False
    },
    'trading.max_daily_loss': {
        'type': (int, float),
        'min': 0.1,
        'max': 10000.0,
        'required': False
    },
    'trading.default_exchange': {
        'type': str,
        'choices': ['binance', 'gemini', 'kraken', 'coinbase'],
        'required': False
    },
    'trading.save_trades_to_csv': {
        'type': bool,
        'required': False
    }
}

# Risk Management Validation Rules
RISK_MANAGEMENT_VALIDATION_RULES: Dict[str, Dict[str, Any]] = {
    'risk_management.stop_loss_enabled': {
        'type': bool,
        'required': False
    },
    'risk_management.stop_loss_percentage': {
        'type': float,
        'min': 0.001,
        'max': 0.5,
        'required': False
    },
    'risk_management.take_profit_enabled': {
        'type': bool,
        'required': False
    },
    'risk_management.take_profit_percentage': {
        'type': float,
        'min': 0.001,
        'max': 2.0,
        'required': False
    },
    'risk_management.max_position_risk': {
        'type': float,
        'min': 0.001,
        'max': 0.1,
        'required': False
    },
    'risk_management.max_portfolio_risk': {
        'type': float,
        'min': 0.01,
        'max': 0.2,
        'required': False
    },
    'risk_management.max_daily_loss_percentage': {
        'type': float,
        'min': 0.001,
        'max': 0.1,
        'required': False
    },
    'risk_management.max_daily_loss_usd': {
        'type': (int, float),
        'min': 0.01,
        'max': 10000.0,
        'required': False
    },
    'risk_management.max_drawdown_percentage': {
        'type': float,
        'min': 0.01,
        'max': 0.5,
        'required': False
    }
}

# Constants Validation Rules
CONSTANTS_VALIDATION_RULES: Dict[str, Dict[str, Any]] = {
    'constants.default_cycle_delay': {
        'type': int,
        'min': 10,
        'max': 3600,
        'required': False
    },
    'constants.min_sleep_time': {
        'type': int,
        'min': 1,
        'max': 300,
        'required': False
    },
    'constants.default_max_retries': {
        'type': int,
        'min': 1,
        'max': 10,
        'required': False
    },
    'constants.performance_window_size': {
        'type': int,
        'min': 5,
        'max': 100,
        'required': False
    },
    'constants.min_trades_for_win_rate': {
        'type': int,
        'min': 1,
        'max': 50,
        'required': False
    },
    'constants.cash_reserve_percentage': {
        'type': float,
        'min': 0.01,
        'max': 0.5,
        'required': False
    }
}

# Combined validation rules
ALL_VALIDATION_RULES: Dict[str, Dict[str, Any]] = {
    **TRADING_VALIDATION_RULES,
    **RISK_MANAGEMENT_VALIDATION_RULES,
    **CONSTANTS_VALIDATION_RULES
}

def get_validation_rules() -> Dict[str, Dict[str, Any]]:
    """Get all validation rules"""
    return ALL_VALIDATION_RULES

def validate_yaml_config(yaml_loader) -> bool:
    """
    Validate YAML configuration using defined rules
    
    Args:
        yaml_loader: YAMLConfigLoader instance
        
    Returns:
        True if valid, raises exception if invalid
    """
    return yaml_loader.validate_config(ALL_VALIDATION_RULES)