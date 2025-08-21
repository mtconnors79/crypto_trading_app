"""
Type hints and type definitions for the trading bot
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
from pandas import DataFrame

# Basic types
Price = float
Quantity = float
Symbol = str
Confidence = float
Fee = float

# Trading types
Signal = Union['buy', 'sell', 'hold', None]
TradeResult = Tuple[bool, str]  # (success, message)
PriceData = Dict[str, Union[Price, datetime]]

# Portfolio types
Position = Dict[str, Union[Quantity, Price]]
Portfolio = Dict[Symbol, Position]

# Analysis types
AnalysisResult = Tuple[Signal, Confidence, str]  # (signal, confidence, reason)
MarketData = Dict[Symbol, PriceData]

# Configuration types
ConfigValue = Union[str, int, float, bool]
Config = Dict[str, ConfigValue]