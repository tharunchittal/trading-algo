"""Quick backtest of trained HFT model"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from hft_trading_env import HFTTradingEnv
from market_regime import MarketRegimeDetector
from config import DATA_CONFIG, RL_CONFIG, TRADING_CONFIG

try:
    from stable_baselines3 import PPO
except ImportError:
    print("Error: stable-baselines3 not installed")
    exit(1)

def quick_backtest():
    """Run a quick backtest on recent data"""
    
    print("\n" + "="*60)
    print("QUICK RL AGENT BACKTEST")
    print("="*60)
    
    # Fetch recent data
    symbol = "SPY"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"\nFetching {symbol} data from {start_date.date()} to {end_date.date()}...")
    df = yf.download(symbol, start=start_date, end=end_date, interval='5m', progress=False)
    
    # Normalize: flatten MultiIndex columns if present, then to lowercase
    if isinstance(df.columns, pd.MultiIndex):
        # Drop the symbol level (level 1), keep the OHLCV level (level 0)
        df.columns = df.columns.droplevel(1)
    df.columns = [col.lower() for col in df.columns]
    
    if df.empty or len(df) < 50:
        print(f"ERROR: Not enough data fetched (got {len(df)} candles)")
        return
    
    print(f"✓ Fetched {len(df)} 5-min candles")
    
    # Create environment
    env = HFTTradingEnv(
        df=df,
        initial_capital=TRADING_CONFIG['initial_capital']
    )
    
    # Try to load model
    model_path = "models/hft_agent/hft_agent.zip"
    try:
        model = PPO.load(model_path)
        print(f"✓ Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model not found at {model_path}")
        return
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return
    
    # Run backtest
    print(f"\nRunning backtest on {len(df)} candles...")
    print("-" * 60)
    
    obs = env.reset()
    if isinstance(obs, tuple):  # gymnasium returns (obs, info)
        obs = obs[0]
    
    trades = []
    trade_count = 0
    entry_price = None
    
    for step in range(len(df)):
        try:
            action, _ = model.predict(obs, deterministic=True)
            result = env.step(action)
            
            # Handle both old gym (4 values) and new gymnasium (5 values) formats
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            
            # Track trades
            if action == 1 and len(env.holdings) == 0:  # Entry signal
                trade_count += 1
                entry_price = env.df.iloc[env.current_step]['close']
                trades.append({
                    'entry_step': step,
                    'entry_price': entry_price,
                    'entry_time': df.index[step],
                })
                print(f"  Step {step:4d}: BUY  @ ${entry_price:.2f}")
            
            elif action == 2 and len(env.holdings) > 0:  # Exit signal
                exit_price = env.df.iloc[env.current_step]['close']
                entry = trades[-1] if trades else None
                if entry:
                    # Calculate actual shares from the environment
                    first_holding = list(env.holdings.values())[0] if env.holdings else {}
                    shares = first_holding.get('qty', 1)
                    pnl = (exit_price - entry['entry_price']) * shares
                    trades[-1]['exit_step'] = step
                    trades[-1]['exit_price'] = exit_price
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit_time'] = df.index[step]
                print(f"  Step {step:4d}: SELL @ ${exit_price:.2f}")
            
            if done:
                break
                
        except Exception as e:
            print(f"Error during backtest step {step}: {e}")
            break
    
    print("-" * 60)
    
    # Results
    print(f"\nBACKTEST RESULTS:")
    print(f"  Trades executed: {trade_count}")
    print(f"  Final NAV: ${env.capital:.2f}")
    print(f"  Total Return: {(env.capital - env.initial_capital) / env.initial_capital * 100:.2f}%")
    
    # Calculate max drawdown
    if env.nav_history:
        peak = max(env.nav_history)
        max_dd = min((nav - peak) / peak for nav in env.nav_history if nav <= peak) if peak > 0 else 0
        print(f"  Max Drawdown: {max_dd * 100:.2f}%")
    
    if trades:
        print(f"\nTrade Details:")
        for i, trade in enumerate(trades, 1):
            entry_time = trade['entry_time'].strftime('%H:%M')
            exit_time = trade.get('exit_time', 'OPEN').strftime('%H:%M') if isinstance(trade.get('exit_time'), pd.Timestamp) else 'OPEN'
            pnl = trade.get('pnl', 0)
            print(f"  Trade {i}: {entry_time} @ ${trade['entry_price']:.2f} -> {exit_time} @ ${trade.get('exit_price', '?'):.2f} | PnL: ${pnl:.2f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    quick_backtest()
