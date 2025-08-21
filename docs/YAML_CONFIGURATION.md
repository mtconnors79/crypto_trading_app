# YAML Configuration System

The trading bot now supports flexible YAML-based configuration that allows you to override any setting without modifying Python code.

## Overview

The YAML configuration system provides:
- **Environment-specific configurations** (development, production)
- **Local overrides** for personal customization
- **Validation** to ensure safe parameter values
- **Hot-reloading** of configuration changes
- **Type conversion** and error handling

## Configuration Files Priority

Configuration files are loaded in order of priority (lowest to highest):

1. `config/config.yaml` - Base configuration
2. `config/config.{ENVIRONMENT}.yaml` - Environment-specific (e.g., `config.development.yaml`)
3. `config/config.local.yaml` - Local overrides (highest priority)

Higher priority files override settings from lower priority files.

## File Structure

### Base Configuration (`config/config.yaml`)
Contains all default settings organized by category:

```yaml
# Trading Configuration
trading:
  base_position_size: 50.0
  max_position_size: 80.0
  max_daily_trades: 50

# Risk Management
risk_management:
  stop_loss_enabled: true
  stop_loss_percentage: 0.05
  take_profit_percentage: 0.10

# Constants
constants:
  default_cycle_delay: 120
  min_sleep_time: 20
```

### Environment-Specific Configuration
- `config/config.development.yaml` - Development settings (smaller positions, debug mode)
- `config/config.production.yaml` - Production settings (larger positions, monitoring)

### Local Configuration (`config/config.local.yaml`)
Create this file for personal overrides. **This file is gitignored** so your personal settings won't be committed.

Copy `config/config.local.yaml.example` to `config/config.local.yaml` and customize.

## Available Configuration Sections

### Trading Settings (`trading:`)
- `base_position_size`: Base amount per trade in USD
- `max_position_size`: Maximum per trade
- `min_position_size`: Minimum per trade
- `max_daily_trades`: Maximum trades per day
- `max_positions`: Maximum concurrent positions
- `trade_cooldown`: Seconds between trades on same symbol
- `min_confidence_threshold`: Minimum confidence to execute trade
- `default_exchange`: Exchange to use (gemini, binance, kraken)
- `save_trades_to_csv`: Save trades to CSV file

### Risk Management (`risk_management:`)
- `stop_loss_enabled`: Enable/disable stop-loss orders
- `stop_loss_percentage`: Stop-loss percentage (0.05 = 5%)
- `take_profit_enabled`: Enable/disable take-profit orders
- `take_profit_percentage`: Take-profit percentage (0.10 = 10%)
- `max_position_risk`: Max risk per position as % of portfolio
- `max_portfolio_risk`: Max total portfolio risk
- `max_daily_loss_usd`: Maximum daily loss in USD
- `max_drawdown_percentage`: Maximum drawdown before pause

### Constants (`constants:`)
- `default_cycle_delay`: Trading cycle delay in seconds
- `min_sleep_time`: Minimum sleep between cycles
- `default_max_retries`: Maximum retry attempts
- `performance_window_size`: Number of trades for performance tracking
- `cash_reserve_percentage`: Cash reserve to maintain

### Logging (`logging:`)
- `level`: Log level (DEBUG, INFO, WARNING, ERROR)
- `log_trades`: Enable trade logging
- `log_performance`: Enable performance logging
- `max_log_size_mb`: Maximum log file size

## Usage Examples

### Example 1: Conservative Trading
```yaml
# config/config.local.yaml
trading:
  base_position_size: 25.0        # Smaller positions
  max_daily_trades: 10            # Fewer trades
  min_confidence_threshold: 0.4   # Higher confidence required

risk_management:
  stop_loss_percentage: 0.03      # Tighter stop-loss
  max_position_risk: 0.01         # Lower risk per position
```

### Example 2: Development Testing
```yaml
# config/config.local.yaml
trading:
  base_position_size: 10.0        # Small test positions
  max_positions: 1                # Only 1 position

constants:
  default_cycle_delay: 30         # Faster cycles for testing

development:
  debug_mode: true                # Enable debug output
```

### Example 3: Different Exchange
```yaml
# config/config.local.yaml
trading:
  default_exchange: "binance"     # Use Binance instead of Gemini
```

## Environment Variables

Set the `ENVIRONMENT` variable to automatically load environment-specific configs:

```bash
export ENVIRONMENT=development    # Loads config.development.yaml
export ENVIRONMENT=production     # Loads config.production.yaml
```

## Validation

The system validates all configuration values:
- **Type checking**: Ensures values are correct type (int, float, bool, str)
- **Range validation**: Checks min/max values for numeric parameters
- **Choice validation**: Validates values against allowed options
- **Required validation**: Ensures critical parameters are set

Invalid configurations will show warnings but won't crash the bot.

## How It Works

1. On startup, the bot loads YAML files in priority order
2. Each config module checks for YAML overrides
3. Valid overrides replace default Python values
4. Configuration is validated against rules
5. Bot runs with the merged configuration

## Troubleshooting

### Configuration Not Loading
- Check file syntax with a YAML validator
- Verify file names match exactly
- Check console output for loading messages

### Invalid Values
- Check validation error messages
- Refer to validation rules in the code
- Use example files as reference

### Missing PyYAML
Install the required dependency:
```bash
pip install PyYAML
```

## Best Practices

1. **Start with examples**: Copy from `.example` files
2. **Use local configs**: Keep personal settings in `config.local.yaml`
3. **Test configurations**: Start with conservative settings
4. **Version control**: Only commit base and environment configs
5. **Validate syntax**: Use a YAML linter before running

## Advanced Usage

### Multiple Local Configs
You can create environment-specific local configs:
- `config.development.local.yaml`
- `config.production.local.yaml`

### Programmatic Access
```python
from src.config.yaml_loader import get_yaml_loader

loader = get_yaml_loader()
value = loader.get_config_value('trading.base_position_size', default=50.0)
```

### Custom Validation
Add custom validation rules in `src/config/validation_rules.py`.