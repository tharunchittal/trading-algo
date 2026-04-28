"""Testing utilities for HFT RL system"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def test_imports():
    """Test all imports work"""
    print("Testing imports...")
    
    try:
        import yfinance
        print("  ✓ yfinance")
    except ImportError as e:
        print(f"  ✗ yfinance: {e}")
        return False
    
    try:
        import gym
        print("  ✓ gym")
    except ImportError as e:
        print(f"  ✗ gym: {e}")
        return False
    
    try:
        from stable_baselines3 import PPO
        print("  ✓ stable-baselines3")
    except ImportError as e:
        print(f"  ✗ stable-baselines3: {e}")
        return False
    
    try:
        from data_provider import HFTDataProvider
        print("  ✓ data_provider")
    except ImportError as e:
        print(f"  ✗ data_provider: {e}")
        return False
    
    try:
        from pattern_recognizer import PatternRecognizer
        print("  ✓ pattern_recognizer")
    except ImportError as e:
        print(f"  ✗ pattern_recognizer: {e}")
        return False
    
    try:
        from market_regime import MarketRegimeDetector
        print("  ✓ market_regime")
    except ImportError as e:
        print(f"  ✗ market_regime: {e}")
        return False
    
    try:
        from hft_trading_env import HFTTradingEnv
        print("  ✓ hft_trading_env")
    except ImportError as e:
        print(f"  ✗ hft_trading_env: {e}")
        return False
    
    print("✓ All imports successful\n")
    return True


def test_data_provider():
    """Test data provider"""
    print("Testing data provider...")
    
    try:
        from data_provider import HFTDataProvider
        
        provider = HFTDataProvider()
        
        # Try to get data
        df = provider.get_historical_data('SPY', days=5, interval='1m')
        
        if df.empty:
            print("  ✗ No data returned")
            return False
        
        print(f"  ✓ Fetched {len(df)} candles")
        
        # Check columns
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in expected_cols):
            print(f"  ✗ Missing columns. Got: {df.columns.tolist()}")
            return False
        
        print(f"  ✓ All required columns present")
        
        # Check data quality
        if df.isnull().any().any():
            print("  ✗ Data contains NaN values")
            return False
        
        print("  ✓ No missing values")
        
        # Test normalization
        normalized = provider.normalize_ohlcv(df)
        if np.isnan(normalized).any() or normalized.min() < -1e-6 or normalized.max() > 1 + 1e-6:
            print("  ✗ Normalization failed")
            return False
        
        print("  ✓ Normalization works")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    print("✓ Data provider test passed\n")
    return True


def test_pattern_recognizer():
    """Test pattern recognition"""
    print("Testing pattern recognizer...")
    
    try:
        from data_provider import HFTDataProvider
        from pattern_recognizer import PatternRecognizer
        
        provider = HFTDataProvider()
        df = provider.get_historical_data('SPY', days=10, interval='1m')
        
        if df.empty:
            print("  ✗ No data for testing")
            return False
        
        recognizer = PatternRecognizer()
        patterns = recognizer.detect_patterns(df)
        
        print(f"  ✓ Detected {len(patterns)} patterns")
        
        if patterns:
            p = patterns[0]
            print(f"    First pattern: {p.name} ({p.confidence:.2f} confidence)")
            
            # Test feature extraction
            features = recognizer.extract_pattern_features(p, df)
            if len(features) < 8:
                print(f"  ✗ Feature vector too small: {len(features)} < 8")
                return False
            
            print(f"  ✓ Feature extraction works ({len(features)} dims)")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ Pattern recognizer test passed\n")
    return True


def test_market_regime():
    """Test market regime detection"""
    print("Testing market regime detector...")
    
    try:
        from data_provider import HFTDataProvider
        from market_regime import MarketRegimeDetector
        
        provider = HFTDataProvider()
        df = provider.get_historical_data('SPY', days=30, interval='1m')
        
        if df.empty:
            print("  ✗ No data for testing")
            return False
        
        detector = MarketRegimeDetector()
        df = detector.add_indicators(df)
        
        print(f"  ✓ Added indicators")
        
        regime = detector.detect_regime(df)
        if regime not in ['bull', 'bear', 'neutral']:
            print(f"  ✗ Invalid regime: {regime}")
            return False
        
        print(f"  ✓ Regime detection: {regime}")
        
        strength = detector.detect_regime_strength(df)
        if not (0 <= strength <= 1):
            print(f"  ✗ Invalid strength: {strength}")
            return False
        
        print(f"  ✓ Regime strength: {strength:.2f}")
        
        vol_regime = detector.detect_volatility_regime(df)
        print(f"  ✓ Volatility regime: {vol_regime}")
        
        features = detector.get_regime_features(df)
        if len(features) < 10:
            print(f"  ✗ Feature vector too small: {len(features)} < 10")
            return False
        
        print(f"  ✓ Feature extraction works ({len(features)} dims)")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ Market regime test passed\n")
    return True


def test_trading_env():
    """Test RL trading environment"""
    print("Testing trading environment...")
    
    try:
        from data_provider import HFTDataProvider
        from hft_trading_env import HFTTradingEnv
        
        provider = HFTDataProvider()
        df = provider.get_historical_data('SPY', days=10, interval='1m')
        
        if df.empty or len(df) < 100:
            print("  ✗ Insufficient data for testing")
            return False
        
        env = HFTTradingEnv(df)
        print(f"  ✓ Environment created")
        
        obs = env.reset()
        if obs.shape != (42,):
            print(f"  ✗ Observation shape wrong: {obs.shape} != (42,)")
            return False
        
        print(f"  ✓ Observation shape correct")
        
        # Test random actions
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            if obs.shape != (42,):
                print(f"  ✗ Observation shape inconsistent")
                return False
            
            if done:
                break
        
        print(f"  ✓ 10 steps executed successfully")
        
        stats = env.get_stats()
        print(f"  ✓ Stats collected: {len(stats)} metrics")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ Trading environment test passed\n")
    return True


def test_live_trader():
    """Test live trader"""
    print("Testing live trader...")
    
    try:
        from live_hft_trader import LiveHFTTrader
        from data_provider import HFTDataProvider
        
        trader = LiveHFTTrader(symbols=['SPY'])
        print(f"  ✓ Trader initialized")
        
        provider = HFTDataProvider()
        df = provider.get_historical_data('SPY', days=5, interval='1m')
        
        if not df.empty:
            signal = trader.generate_signal('SPY', df)
            print(f"  ✓ Signal generated: {['HOLD', 'BUY', 'SELL'][signal.get('action', 0)]}")
        
        # Test trade execution
        trader.execute_trade('SPY', 1, 450.0)
        print(f"  ✓ Trade executed")
        
        stats = trader.get_performance_stats()
        print(f"  ✓ Performance stats: {stats['total_trades']} trades")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ Live trader test passed\n")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("HFT RL SYSTEM TEST SUITE")
    print("="*60 + "\n")
    
    results = {}
    
    # Test imports first
    if not test_imports():
        print("✗ Import tests failed. Install dependencies with:")
        print("  pip install -r hft_requirements.txt")
        return False
    
    # Run component tests
    results['data_provider'] = test_data_provider()
    results['pattern_recognizer'] = test_pattern_recognizer()
    results['market_regime'] = test_market_regime()
    results['trading_env'] = test_trading_env()
    results['live_trader'] = test_live_trader()
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print("="*60)
    print(f"Result: {passed}/{total} tests passed")
    print("="*60 + "\n")
    
    if passed == total:
        print("✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Review QUICKSTART.md")
        print("  2. Run: python hft_main.py train --symbols SPY --timesteps 100000")
        print("  3. Test: python hft_main.py paper --symbol SPY --days 7")
        return True
    else:
        print("✗ Some tests failed. Check errors above.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
