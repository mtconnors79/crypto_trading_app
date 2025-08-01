"""
Trading Symbol Configuration
Easily modify which cryptocurrencies the bot trades
"""

# SYMBOL SETS FOR DIFFERENT STRATEGIES

# Conservative - Top 6 established coins
CONSERVATIVE_SYMBOLS = [
    'BTC/USD', 'ETH/USD', 'LTC/USD', 
    'LINK/USD', 'SOL/USD', 'DOGE/USD'
]

# Balanced - Mix of established + emerging
BALANCED_SYMBOLS = [
    'BTC/USD', 'ETH/USD', 'LTC/USD', 'LINK/USD', 'SOL/USD', 'DOGE/USD',
    'ADA/USD', 'DOT/USD', 'AVAX/USD', 'MATIC/USD'
]

# Aggressive - Maximum coverage for opportunities  
AGGRESSIVE_SYMBOLS = [
    # Major cryptocurrencies
    'BTC/USD', 'ETH/USD', 'LTC/USD',
    # DeFi tokens
    'LINK/USD', 'UNI/USD', 'AAVE/USD', 'COMP/USD', 'MKR/USD',
    # Layer 1 blockchains
    'SOL/USD', 'ADA/USD', 'DOT/USD', 'AVAX/USD',
    # Popular/trending
    'DOGE/USD', 'MATIC/USD', 'XTZ/USD'
]

# DeFi Focus - DeFi tokens only
DEFI_SYMBOLS = [
    'ETH/USD', 'LINK/USD', 'UNI/USD', 'AAVE/USD', 
    'COMP/USD', 'MKR/USD', 'CRV/USD', 'SUSHI/USD'
]

# Major Only - Just BTC and ETH
MAJOR_ONLY = ['BTC/USD', 'ETH/USD']

# SMALL CAP HIGH-VOLATILITY STRATEGIES

# Pure Meme Coins - Maximum volatility
MEME_COINS = [
    'DOGE/USD', 'SHIB/USD', 'PEPE/USD', 'BONK/USD', 'FLOKI/USD',
    'WIF/USD', 'BOME/USD', 'PNUT/USD', 'GOAT/USD', 'MEW/USD',
    'POPCAT/USD', 'TRUMP/USD'
]

# High Volatility Small Caps - Your custom selection (18 coins)
HIGH_VOLATILITY_SMALL_CAPS = [
    'DOGE/USD', 'SHIB/USD', 'PEPE/USD', 'BONK/USD', 'WIF/USD',
    'AAVE/USD', 'COMP/USD', 'CRV/USD', 'UNI/USD', 'SUSHI/USD',
    'TRUMP/USD', 'GOAT/USD', 'PNUT/USD', 'POPCAT/USD',
    'GALA/USD', 'SAND/USD', 'MANA/USD', 'IMX/USD'
]

def get_symbols_for_market(market_condition='balanced'):
    """
    Get symbol list based on market condition
    Options: 'conservative', 'balanced', 'aggressive', 'defi', 'major', 
             'meme', 'small_cap'
    """
    symbol_sets = {
        'conservative': CONSERVATIVE_SYMBOLS,
        'balanced': BALANCED_SYMBOLS, 
        'aggressive': AGGRESSIVE_SYMBOLS,
        'defi': DEFI_SYMBOLS,
        'major': MAJOR_ONLY,
        'meme': MEME_COINS,
        'small_cap': HIGH_VOLATILITY_SMALL_CAPS
    }
    return symbol_sets.get(market_condition, BALANCED_SYMBOLS)

# ACTIVE CONFIGURATION
# Change this line to switch strategies:
MARKET_CONDITION = 'small_cap'  # Using your custom 18-coin high-volatility selection

# Get active symbols
SYMBOLS = get_symbols_for_market(MARKET_CONDITION)

# Display current configuration
if __name__ == "__main__":
    print(f"🎯 Active Strategy: {MARKET_CONDITION.upper()}")
    print(f"📊 Number of Symbols: {len(SYMBOLS)}")
    print(f"💰 Trading Symbols:")
    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"  {i:2d}. {symbol}")
    
    print(f"\n🚀 Strategy Breakdown:")
    print(f"   • Meme Coins: DOGE, SHIB, PEPE, BONK, WIF, TRUMP, GOAT, PNUT, POPCAT")
    print(f"   • DeFi Tokens: AAVE, COMP, CRV, UNI, SUSHI") 
    print(f"   • Gaming/NFT: GALA, SAND, MANA, IMX")
    print(f"\n⚠️  HIGH RISK, HIGH REWARD STRATEGY!")
    print(f"💡 Perfect for small accounts seeking maximum volatility!")
