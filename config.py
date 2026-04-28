"""Configuration for RL HFT Trading Agent"""

# Data Configuration
DATA_CONFIG = {
    'interval': '5m',  # 5-minute candles for quick testing
    'symbols': ['SPY', 'QQQ'],  # Quick test with 2 symbols
    'lookback_days': 30,  # 30 days of recent data
    'cache_dir': '.cache/hft_data',
}

# Pattern Recognition
PATTERN_CONFIG = {
    'lookback_window': 20,  # Candles to analyze for pattern
    'patterns': ['triangle', 'wedge', 'flag', 'channel', 'head_shoulders'],
    'min_pattern_confidence': 0.65,
    'learned_model_path': 'models/pattern/learned_pattern_detector.pkl',
    'forward_horizon': 6,  # Label quality check horizon in bars
    'forward_return_threshold': 0.0012,  # 12 bps directional threshold
}

# Market Regime Detection
REGIME_CONFIG = {
    'sma_short': 5,
    'sma_long': 20,
    'rsi_period': 14,
    'atr_period': 14,
}

# RL Agent
RL_CONFIG = {
    'agent_type': 'PPO',  # Policy Gradient method
    'model_architecture': [256, 256, 128],  # Neural network layers
    'learning_rate': 1e-4,
    'batch_size': 64,
    'gamma': 0.99,  # Discount factor
    'gae_lambda': 0.95,
    'clip_ratio': 0.2,
    'max_episodes': 60,  # Quick test: 60 episodes
    'steps_per_episode': 1000,
}

# Extended Training Config (for better convergence)
EXTENDED_RL_CONFIG = {
    'agent_type': 'PPO',
    'model_architecture': [512, 256, 128],  # Larger network
    'learning_rate': 5e-5,  # Lower LR for stability
    'batch_size': 64,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_ratio': 0.2,
    'max_episodes': 200,  # More episodes
    'steps_per_episode': 1000,
}

# Trading Configuration
TRADING_CONFIG = {
    'initial_capital': 100000,
    'position_size': 0.25,  # 25% per trade (balanced: larger bets than original)
    'max_positions': 1,  # Keep exposure tighter for HFT regime noise
    'stop_loss_pct': 0.02,  # 2% stop loss
    'take_profit_pct': 0.04,  # 4% take profit
    'transaction_cost': 0.001,  # 0.1% per trade
    'slippage': 0.0005,  # 0.05% slippage
    'entry_confidence_threshold': 0.15,  # Relaxed to encourage trading
    'entry_cooldown_steps': 7,  # Moderate frequency (35 min on 5m candles, vs 10 min before)
    'min_hold_steps': 3,
    'max_trades_per_episode': 150,  # Moderate cap (vs 260 original)
    'assist_enabled': True,
    'assist_entry_margin': 0.10,
    'assist_bull_threshold': 0.55,
    'assist_exit_bear_threshold': 0.55,
}

# Reward Configuration
REWARD_CONFIG = {
    'win_reward': 1.0,
    'loss_penalty': -1.0,
    'hold_reward': -0.006,  # Small penalty for inaction
    'max_hold_penalty': -0.5,  # Penalty for holding too long
    'max_hold_steps': 100,
    'invalid_action_penalty': -0.03,
    'overtrade_penalty': -0.01,
    'drawdown_penalty_scale': 0.10,
    'entry_quality_bonus_scale': 0.12,
}
