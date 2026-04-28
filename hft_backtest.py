"""Backtesting framework for HFT RL agent"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

from hft_trading_env import HFTTradingEnv
from data_provider import HFTDataProvider
from pattern_recognizer import PatternRecognizer
from market_regime import MarketRegimeDetector
from config import TRADING_CONFIG, PATTERN_CONFIG


class HFTBacktester:
    """Backtest HFT RL agent on historical data"""
    
    def __init__(self, model_path: str = 'models/hft_agent.zip'):
        """
        Initialize backtester.
        
        Args:
            model_path: Path to trained model
        """
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
            self.has_model = True
        except:
            print(f"Warning: Could not load model. Using random actions.")
            self.model = None
            self.has_model = False
        
        self.data_provider = HFTDataProvider()
        self.results = None
    
    def run_backtest(self, symbol: str, days: int = 60, interval: str = '1m') -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            symbol: Stock ticker
            days: Number of days of data
            interval: Candle interval
            
        Returns:
            Backtest results dictionary
        """
        # Fetch data
        print(f"Fetching {symbol} data ({days} days, {interval} interval)...")
        df = self.data_provider.get_historical_data(symbol, days=days, interval=interval)
        
        if df.empty:
            print(f"Error: No data for {symbol}")
            return None
        
        print(f"Loaded {len(df)} candles")
        
        # Create environment
        env = HFTTradingEnv(
            df,
            learned_model_path=PATTERN_CONFIG.get('learned_model_path'),
            auto_train_pattern_model=False,
        )
        
        # Run backtest
        print("Running backtest...")
        obs = env.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            # Get action from model or random
            if self.has_model:
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if done:
                break
            
            if step_count % 10000 == 0:
                print(f"  Step {step_count}: Capital=${info['capital']:.2f}")
        
        # Collect results
        stats = env.get_stats()
        
        initial_cap = env.initial_capital
        final_cap = stats['final_value']
        pnl = final_cap - initial_cap
        return_pct = (pnl / initial_cap) * 100
        
        results = {
            'symbol': symbol,
            'period': f"{df.index[0]} to {df.index[-1]}" if hasattr(df.index[0], 'date') else f"{days} days",
            'candles_tested': len(df),
            'total_episode_reward': episode_reward,
            'final_capital': final_cap,
            'initial_capital': initial_cap,
            'pnl': pnl,
            'return_pct': return_pct,
            'total_return': stats['total_return'],
            'sharpe_ratio': stats['sharpe_ratio'],
            'max_drawdown': stats['max_drawdown'],
            'win_rate': stats['win_rate'],
            'trades': stats['total_trades'],
            'nav_history': env.nav_history,
            'trades_list': env.trades_executed
        }
        
        self.results = results
        return results
    
    def run_multi_symbol_backtest(self, symbols: List[str] = None, days: int = 60) -> Dict:
        """
        Run backtest across multiple symbols.
        
        Args:
            symbols: List of symbols
            days: Number of days
            
        Returns:
            Dictionary of backtest results
        """
        symbols = symbols or ['SPY', 'QQQ', 'AAPL']
        
        all_results = {}
        
        for symbol in symbols:
            print(f"\n{'='*60}")
            print(f"Testing {symbol}")
            print('='*60)
            
            results = self.run_backtest(symbol, days=days)
            
            if results:
                all_results[symbol] = results
                self.print_results(results)
        
        return all_results
    
    def print_results(self, results: Dict = None):
        """Print backtest results"""
        if results is None:
            results = self.results
        
        if results is None:
            print("No results to display")
            return
        
        print("\n" + "="*60)
        print(f"BACKTEST RESULTS: {results['symbol']}")
        print("="*60)
        print(f"Period: {results['period']}")
        print(f"Candles Tested: {results['candles_tested']:,}")
        print()
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print()
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.1%}")
        print()
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Episode Reward: {results['total_episode_reward']:.2f}")
        print("="*60 + "\n")
    
    def plot_results(self, results: Dict = None, save_path: str = 'backtest_results.png'):
        """
        Plot backtest results.
        
        Args:
            results: Results dictionary
            save_path: Path to save plot
        """
        if results is None:
            results = self.results
        
        if results is None:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f"HFT Agent Backtest: {results['symbol']}", fontsize=16, fontweight='bold')
        
        # Portfolio value over time
        ax = axes[0, 0]
        nav = np.array(results['nav_history'])
        ax.plot(nav, linewidth=2, color='blue')
        ax.axhline(y=results['initial_capital'], color='red', linestyle='--', label='Initial Capital')
        ax.set_title('Portfolio Value Over Time')
        ax.set_ylabel('Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cumulative returns
        ax = axes[0, 1]
        returns = (nav - results['initial_capital']) / results['initial_capital'] * 100
        ax.plot(returns, linewidth=2, color='green')
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_title('Cumulative Return (%)')
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3)
        
        # Drawdown
        ax = axes[1, 0]
        cummax = np.maximum.accumulate(nav)
        drawdown = (nav - cummax) / cummax * 100
        ax.plot(drawdown, linewidth=2, color='red')
        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax.set_title('Drawdown (%)')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # Trade distribution
        ax = axes[1, 1]
        trades = results['trades']
        if trades:
            buy_count = sum(1 for t in trades if t['type'] == 'buy')
            sell_count = sum(1 for t in trades if t['type'] == 'sell')
            
            ax.bar(['Buy', 'Sell'], [buy_count, sell_count], color=['green', 'red'])
            ax.set_title(f'Trade Distribution (Total: {len(trades)})')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close()
    
    def create_comparison_report(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create comparison report across symbols.
        
        Args:
            results_dict: Dictionary of results by symbol
            
        Returns:
            DataFrame with comparison metrics
        """
        data = []
        
        for symbol, results in results_dict.items():
            data.append({
                'Symbol': symbol,
                'Candles': results['candles_tested'],
                'Return': f"{results['total_return']:.2%}",
                'Sharpe': f"{results['sharpe_ratio']:.2f}",
                'Max DD': f"{results['max_drawdown']:.2%}",
                'Win Rate': f"{results['win_rate']:.1%}",
                'Trades': results['total_trades'],
                'Final Value': f"${results['final_capital']:,.0f}"
            })
        
        df = pd.DataFrame(data)
        
        print("\n" + "="*100)
        print("MULTI-SYMBOL COMPARISON")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100 + "\n")
        
        return df


def run_full_backtest():
    """Run complete backtest suite"""
    
    print("\n" + "="*60)
    print("HFT RL AGENT BACKTEST SUITE")
    print("="*60 + "\n")
    
    backtester = HFTBacktester()
    
    # Test on multiple symbols and timeframes
    symbols = ['SPY', 'QQQ', 'AAPL']
    
    all_results = {}
    for symbol in symbols:
        results = backtester.run_backtest(symbol, days=60)
        if results:
            all_results[symbol] = results
            backtester.print_results(results)
            backtester.plot_results(results, f'backtest_{symbol}_results.png')
    
    # Create comparison
    if all_results:
        backtester.create_comparison_report(all_results)


if __name__ == '__main__':
    run_full_backtest()
