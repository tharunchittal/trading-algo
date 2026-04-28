"""Debug backtest to understand why no trades execute"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from hft_trading_env import HFTTradingEnv
from config import TRADING_CONFIG

try:
    from stable_baselines3 import PPO
except ImportError:
    print("Error: stable-baselines3 not installed")
    exit(1)

def debug_backtest():
    """Run backtest with debug output"""
    
    print("\n" + "="*60)
    print("DEBUG RL AGENT BACKTEST")
    print("="*60)
    
    # Fetch recent data
    symbol = "SPY"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"\nFetching {symbol} data...")
    df = yf.download(symbol, start=start_date, end=end_date, interval='5m', progress=False)
    
    # Normalize columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns = [col.lower() for col in df.columns]
    
    print(f"✓ Fetched {len(df)} candles")
    
    # Create environment
    env = HFTTradingEnv(
        df=df,
        initial_capital=TRADING_CONFIG['initial_capital']
    )
    
    # Load model
    model_path = "models/hft_agent/hft_agent.zip"
    try:
        model = PPO.load(model_path)
        print(f"✓ Loaded model")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return
    
    # Run backtest with debugging
    print(f"\nRunning debug backtest on {len(df)} candles...")
    print("-" * 60)
    
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    for step in range(min(100, len(df))):  # Just first 100 steps
        try:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Get signal strength for debugging
            from config import PATTERN_CONFIG
            window = env.df.iloc[max(0, env.current_step - PATTERN_CONFIG['lookback_window']) : env.current_step + 1]
            signal_strength = env._signal_strength(window)
            
            # Print every action=1 (buy) or action=2 (sell)
            if action in [1, 2]:
                action_name = "BUY" if action == 1 else "SELL"
                price = env.df.iloc[env.current_step]['close']
                print(f"Step {step:3d}: Action={action_name:4s}, signal_strength={signal_strength:.4f}, price=${price:.2f}, holdings={len(env.holdings)}, cash=${env.cash:.0f}")
            
            # Execute step
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            
            if done:
                break
                
        except Exception as e:
            print(f"Error at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("-" * 60)
    print(f"\nFinal: Trades={len(env.trades_executed)}, NAV=${env.capital:.2f}, Return={(env.capital - env.initial_capital) / env.initial_capital * 100:.2f}%")
    print("\n" + "="*60)

if __name__ == "__main__":
    debug_backtest()
