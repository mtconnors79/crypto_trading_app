from src.data.collector import DataCollector
from src.trading.strategy import SimpleMovingAverageStrategy

def test_strategy():
    print("🚀 Testing Simple Moving Average Strategy...")
    
    try:
        # Get data
        print("📊 Fetching historical data...")
        collector = DataCollector()
        df = collector.get_historical_data('BTC/USDT', '1h', 50)
        
        if df is None:
            print("❌ Could not fetch data")
            return
            
        print(f"✅ Got {len(df)} data points")
        
        # Initialize strategy
        strategy = SimpleMovingAverageStrategy()
        
        # Calculate indicators
        print("🔧 Calculating indicators...")
        df_with_indicators = strategy.calculate_indicators(df.copy())
        
        # Check if we have enough data
        if len(df_with_indicators) < strategy.long_window:
            print(f"❌ Need at least {strategy.long_window} data points, got {len(df_with_indicators)}")
            return
            
        print("✅ Indicators calculated")
        
        # Generate signal
        print("🎯 Generating trading signal...")
        signal, confidence, reason = strategy.generate_signal(df_with_indicators)
        
        print(f"\n📈 CURRENT SIGNAL:")
        print(f"   Action: {signal.upper()}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Reason: {reason}")
        
        # Show some technical data
        latest = df_with_indicators.iloc[-1]
        print(f"\n📊 LATEST TECHNICAL DATA:")
        print(f"   Price: ${latest['close']:.2f}")
        print(f"   SMA Short (5): ${latest['sma_short']:.2f}")
        print(f"   SMA Long (20): ${latest['sma_long']:.2f}")
        print(f"   RSI: {latest['rsi']:.1f}")
        
        # Simple backtest
        print(f"\n🔄 Running backtest...")
        results = strategy.backtest_strategy(df_with_indicators)
        
        print(f"\n📊 BACKTEST RESULTS:")
        print(f"   Initial: ${results['initial_balance']:,.2f}")
        print(f"   Final: ${results['final_value']:,.2f}")
        print(f"   Return: {results['total_return']:+.2f}%")
        print(f"   Trades: {results['num_trades']}")
        
        print("\n✅ Strategy test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_strategy()
