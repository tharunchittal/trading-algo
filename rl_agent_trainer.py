"""RL Agent Trainer using Stable-Baselines3"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path

# Try to import stable-baselines3, warn if not available
try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable-baselines3 not installed. Install with: pip install stable-baselines3")

from hft_trading_env import HFTTradingEnv
from data_provider import HFTDataProvider
from pattern_recognizer import PatternRecognizer
from config import RL_CONFIG, DATA_CONFIG, PATTERN_CONFIG, EXTENDED_RL_CONFIG, TRADING_CONFIG


class TradingCallback(BaseCallback):
    """Custom callback to monitor training progress"""
    
    def __init__(self, eval_env, eval_frequency: int = 10000):
        super().__init__()
        self.eval_env = eval_env
        self.eval_frequency = eval_frequency
        self.eval_rewards = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_frequency == 0:
            obs = self.eval_env.reset()
            episode_reward = 0
            
            for _ in range(self.eval_env.max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.eval_env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            self.eval_rewards.append(episode_reward)
            print(f"Step {self.n_calls}: Eval Reward = {episode_reward:.2f}")
        
        return True


class HFTRLAgent:
    """RL Trading Agent Trainer"""
    
    def __init__(self, model_path: str = 'models/hft_agent', learned_model_path: str = None):
        """
        Initialize agent.
        
        Args:
            model_path: Path to save/load models
        """
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.learned_model_path = learned_model_path or PATTERN_CONFIG.get('learned_model_path')
        
        self.model = None
        self.train_env = None
        self.eval_env = None
        self.training_history = []

    def train_pattern_model(self, df: pd.DataFrame, max_samples: int = 3000) -> bool:
        """Train and persist learned pattern detector before RL training."""
        recognizer = PatternRecognizer(learned_model_path=self.learned_model_path)
        if recognizer.learned_enabled:
            print(f"Loaded existing learned pattern model from {self.learned_model_path}")
            return True

        trained = recognizer.train_learned_detector(df, max_samples=max_samples, save_after_train=True)
        if trained:
            print(f"Trained and saved learned pattern model to {self.learned_model_path}")
        else:
            print("Warning: Could not train learned pattern model; RL will fallback to rule-based features")
        return trained
    
    def prepare_training_data(self, symbols: List[str] = None) -> Dict:
        """
        Fetch and prepare training data.
        
        Args:
            symbols: List of symbols to fetch
            
        Returns:
            Dictionary of DataFrames
        """
        provider = HFTDataProvider()
        symbols = symbols or DATA_CONFIG['symbols']
        
        print(f"Preparing training data for {len(symbols)} symbols...")
        data = {}
        
        for symbol in symbols:
            df = provider.get_historical_data(
                symbol,
                days=DATA_CONFIG['lookback_days'],
                interval=DATA_CONFIG['interval']
            )
            
            if not df.empty:
                data[symbol] = df
                print(f"  {symbol}: {len(df)} candles")
        
        return data
    
    def create_environments(self, df: pd.DataFrame):
        """Create training and evaluation environments"""
        
        # Split data: 70% train, 30% eval
        split_idx = int(len(df) * 0.7)
        
        train_df = df.iloc[:split_idx].copy()
        eval_df = df.iloc[split_idx:].copy()
        
        self.train_env = HFTTradingEnv(
            train_df,
            learned_model_path=self.learned_model_path,
            auto_train_pattern_model=False,
        )
        self.eval_env = HFTTradingEnv(
            eval_df,
            learned_model_path=self.learned_model_path,
            auto_train_pattern_model=False,
        )
        
        print(f"Created environments: {len(train_df)} train, {len(eval_df)} eval candles")
    
    def build_model(self, agent_type: str = None):
        """
        Build RL model.
        
        Args:
            agent_type: 'PPO' or 'A2C'
        """
        if not HAS_SB3:
            raise RuntimeError("stable-baselines3 not installed")
        
        agent_type = agent_type or RL_CONFIG['agent_type']
        
        policy_kwargs = {
            'net_arch': RL_CONFIG['model_architecture'],
        }
        
        if agent_type == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                self.train_env,
                learning_rate=RL_CONFIG['learning_rate'],
                batch_size=RL_CONFIG['batch_size'],
                gamma=RL_CONFIG['gamma'],
                gae_lambda=RL_CONFIG['gae_lambda'],
                clip_range=RL_CONFIG['clip_ratio'],
                policy_kwargs=policy_kwargs,
                verbose=1,
                device='cpu'  # Use CPU (or 'cuda' if available)
            )
        else:
            self.model = A2C(
                'MlpPolicy',
                self.train_env,
                learning_rate=RL_CONFIG['learning_rate'],
                policy_kwargs=policy_kwargs,
                verbose=1,
                device='cpu'
            )
        
        print(f"Built {agent_type} model with architecture {RL_CONFIG['model_architecture']}")
    
    def train(self, total_timesteps: int = None, eval_frequency: int = 10000, mode: str = 'quick'):
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training timesteps (None = auto from mode)
            eval_frequency: Evaluation frequency
            mode: 'quick' (60k steps) or 'extended' (300k steps)
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        # Auto-determine timesteps from mode if not specified
        if total_timesteps is None:
            if mode == 'extended':
                total_timesteps = 300000
            else:
                total_timesteps = 60000
        
        print(f"Starting {mode} training for {total_timesteps:,} timesteps...")
        print(f"  Position size: {TRADING_CONFIG['position_size']:.0%}")
        print(f"  Cooldown: {TRADING_CONFIG['entry_cooldown_steps']} steps (~{TRADING_CONFIG['entry_cooldown_steps']*5} min on 5m candles)")
        
        callback = TradingCallback(self.eval_env, eval_frequency=eval_frequency)
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        print("Training completed!")
        
        # Save model
        model_file = self.model_path / f'hft_agent_{mode}.zip'
        self.model.save(str(model_file))
        print(f"Model saved to {model_file}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Evaluate trained model.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Performance statistics
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        
        if self.eval_env is None:
            raise RuntimeError("Evaluation environment not created.")
        
        print(f"Evaluating model for {num_episodes} episodes...")
        
        episode_rewards = []
        episode_stats = []
        
        for ep in range(num_episodes):
            obs = self.eval_env.reset()
            episode_reward = 0
            
            for step in range(self.eval_env.max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            stats = self.eval_env.get_stats()
            episode_rewards.append(episode_reward)
            episode_stats.append(stats)
            
            print(f"  Episode {ep+1}: Reward={episode_reward:.2f}, Return={stats['total_return']:.2%}, Win Rate={stats['win_rate']:.1%}")
        
        # Aggregate statistics
        avg_reward = np.mean(episode_rewards)
        avg_return = np.mean([s['total_return'] for s in episode_stats])
        avg_sharpe = np.mean([s['sharpe_ratio'] for s in episode_stats])
        avg_win_rate = np.mean([s['win_rate'] for s in episode_stats])
        
        results = {
            'avg_reward': avg_reward,
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_win_rate': avg_win_rate,
            'episode_stats': episode_stats
        }
        
        return results
    
    def load(self, model_file: str = None):
        """
        Load trained model.
        
        Args:
            model_file: Path to model file (default: models/hft_agent.zip)
        """
        if not HAS_SB3:
            raise RuntimeError("stable-baselines3 not installed")
        
        model_file = model_file or self.model_path / 'hft_agent.zip'
        
        self.model = PPO.load(str(model_file))
        print(f"Loaded model from {model_file}")
    
    def predict_action(self, observation: np.ndarray) -> int:
        """
        Predict trading action for given observation.
        
        Args:
            observation: Current market observation
            
        Returns:
            Action (0=hold, 1=buy, 2=sell, 3=close_all)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded or trained.")
        
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)


def train_hft_agent():
    """Main training script"""
    
    # Initialize agent
    agent = HFTRLAgent()
    
    # Prepare data
    data = agent.prepare_training_data()
    
    if not data:
        print("Error: No data fetched. Check symbols and internet connection.")
        return
    
    # Use primary symbol for training
    primary_symbol = list(data.keys())[0]
    df = data[primary_symbol]

    # Train learned pattern model first
    agent.train_pattern_model(df)
    
    # Create environments
    agent.create_environments(df)
    
    # Build model
    agent.build_model('PPO')
    
    # Train
    agent.train(
        total_timesteps=RL_CONFIG['steps_per_episode'] * RL_CONFIG['max_episodes'],
        eval_frequency=10000
    )
    
    # Evaluate
    results = agent.evaluate(num_episodes=5)
    
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    print(f"Average Return: {results['avg_return']:.2%}")
    print(f"Average Sharpe Ratio: {results['avg_sharpe']:.2f}")
    print(f"Average Win Rate: {results['avg_win_rate']:.1%}")
    print("="*50)


if __name__ == '__main__':
    train_hft_agent()
