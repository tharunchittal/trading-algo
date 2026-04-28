"""RL Trading Environment and Agent"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
try:
    import gym
    from gym import spaces
except ImportError:
    import gymnasium as gym
    from gymnasium import spaces
from config import TRADING_CONFIG, REWARD_CONFIG, RL_CONFIG, PATTERN_CONFIG
from pattern_recognizer import PatternRecognizer
from market_regime import MarketRegimeDetector

class HFTTradingEnv(gym.Env):
    """High-Frequency Trading Environment for RL Agent"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df: pd.DataFrame, initial_capital: float = None, learned_model_path: str = None, auto_train_pattern_model: bool = True):
        """
        Initialize trading environment.
        
        Args:
            df: DataFrame with OHLCV data
            initial_capital: Starting capital
        """
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital or TRADING_CONFIG['initial_capital']
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        
        # Initialize components
        pattern_model_path = learned_model_path or PATTERN_CONFIG.get('learned_model_path')
        self.pattern_recognizer = PatternRecognizer(learned_model_path=pattern_model_path)
        if auto_train_pattern_model and not self.pattern_recognizer.learned_enabled:
            self.pattern_recognizer.train_learned_detector(self.df, max_samples=2000, save_after_train=True)
        self.regime_detector = MarketRegimeDetector()
        self.df = self.regime_detector.add_indicators(self.df)

        self.last_regime_features = np.zeros(14, dtype=np.float32)
        
        # Trading state
        self.cash = self.initial_capital
        self.capital = self.initial_capital
        self.holdings = {}  # {symbol: {qty, entry_price, entry_time}}
        self.trades_executed = []
        self.nav_history = [self.initial_capital]
        self.last_trade_step = -10_000
        self.closed_trade_count = 0
        self.total_trade_actions = 0
        
        # Action space: 0=hold, 1=buy, 2=sell, 3=close_all
        self.action_space = spaces.Discrete(4)
        
        # Observation space: [price_features (20), pattern_features (8), regime_features (14)]
        # = 42 dimensional state
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(42,), dtype=np.float32
        )
        
    def _get_observation(self) -> np.ndarray:
        """Build current observation vector"""
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        
        # Recent price action (last 20 candles)
        start_idx = max(0, self.current_step - 20)
        window = self.df.iloc[start_idx:self.current_step + 1]
        
        price_features = []
        
        # Normalize prices
        closes = window['close'].values
        if len(closes) > 0:
            close_norm = (closes - closes.min()) / (closes.max() - closes.min() + 1e-8)
            price_features.extend(close_norm[:20])
        
        # Pad if necessary
        while len(price_features) < 20:
            price_features.insert(0, 0.5)
        
        price_features = price_features[-20:]  # Take last 20
        
        # Pattern features
        pattern_feature = self.pattern_recognizer.get_pattern_feature_vector(window)
        
        # Regime features
        regime_feature = self.regime_detector.get_regime_features(window)[:14]
        self.last_regime_features = regime_feature.astype(np.float32)
        obs = np.concatenate([price_features, pattern_feature, regime_feature]).astype(np.float32)
        
        # Ensure correct size
        if len(obs) < 42:
            obs = np.pad(obs, (0, 42 - len(obs)), mode='constant', constant_values=0.5)
        else:
            obs = obs[:42]
        
        return obs
    
    def reset(self):
        """Reset environment"""
        start_step = max(50, PATTERN_CONFIG['lookback_window'] + 20)
        self.current_step = min(start_step, self.max_steps)
        self.cash = self.initial_capital
        self.capital = self.initial_capital
        self.holdings = {}
        self.trades_executed = []
        self.nav_history = [self.initial_capital]
        self.last_trade_step = -10_000
        self.closed_trade_count = 0
        self.total_trade_actions = 0
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action and return reward.
        
        Args:
            action: 0=hold, 1=buy, 2=sell, 3=close_all
            
        Returns:
            (observation, reward, done, info)
        """
        current_price = self.df.iloc[self.current_step]['close']
        window = self.df.iloc[max(0, self.current_step - PATTERN_CONFIG['lookback_window']) : self.current_step + 1]
        signal_strength = self._signal_strength(window)
        reward = 0.0
        
        action = self._assist_action(action, signal_strength)

        if action == 1:  # BUY
            reward += self._execute_buy(current_price, signal_strength)
        
        elif action == 2:  # SELL
            reward += self._execute_sell(current_price, signal_strength)        
        elif action == 3:  # CLOSE ALL
            reward += self._close_all_positions(current_price)
        
        else:  # HOLD (action == 0)
            reward += REWARD_CONFIG['hold_reward']
        
        # Penalize for holding too long
        reward += self._penalize_long_holds()

        # Small drawdown-aware penalty to discourage unstable policies.
        reward += self._drawdown_penalty()
        
        # Update position values
        self.capital = self._calculate_portfolio_value(current_price)
        self.nav_history.append(self.capital)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        obs = self._get_observation()
        info = {
            'capital': self.capital,
            'positions': len(self.holdings),
            'price': current_price
        }
        
        # Return old Gym format (obs, reward, done, info) for compatibility
        return obs, reward, done, info
    
    def _execute_buy(self, price: float, signal_strength: float) -> float:
        """Buy if capital allows"""
        if signal_strength < TRADING_CONFIG['entry_confidence_threshold']:
            return REWARD_CONFIG['invalid_action_penalty']

        if (self.current_step - self.last_trade_step) < TRADING_CONFIG['entry_cooldown_steps']:
            return REWARD_CONFIG['invalid_action_penalty']

        if self.total_trade_actions >= TRADING_CONFIG['max_trades_per_episode']:
            return REWARD_CONFIG['overtrade_penalty']

        position_size = TRADING_CONFIG['position_size'] * self.cash
        
        if position_size > price and len(self.holdings) < TRADING_CONFIG['max_positions']:
            qty = int(position_size / price)
            
            if qty > 0:
                cost = qty * price * (1 + TRADING_CONFIG['transaction_cost'])
                
                if cost <= self.cash:
                    self.cash -= cost
                    
                    position_id = f"pos_{self.current_step}_{len(self.holdings)}"
                    self.holdings[position_id] = {
                        'qty': qty,
                        'entry_price': price,
                        'entry_time': self.current_step
                    }
                    
                    self.trades_executed.append({
                        'type': 'buy',
                        'time': self.current_step,
                        'price': price,
                        'qty': qty
                    })
                    self.last_trade_step = self.current_step
                    self.total_trade_actions += 1
                    
                    quality_bonus = REWARD_CONFIG['entry_quality_bonus_scale'] * signal_strength
                    return REWARD_CONFIG['win_reward'] * 0.3 + quality_bonus
        
        return REWARD_CONFIG['invalid_action_penalty']
    
    def _execute_sell(self, price: float, signal_strength: float) -> float:
        """Sell oldest position"""
        if not self.holdings:
            return REWARD_CONFIG['invalid_action_penalty']
        
        position_id = list(self.holdings.keys())[0]
        position = self.holdings[position_id]

        hold_steps = self.current_step - position['entry_time']
        if hold_steps < TRADING_CONFIG['min_hold_steps']:
            return REWARD_CONFIG['invalid_action_penalty'] * 0.5
        
        proceeds = position['qty'] * price * (1 - TRADING_CONFIG['transaction_cost'] - TRADING_CONFIG['slippage'])
        self.cash += proceeds
        
        entry_price = position['entry_price']
        pnl = (price - entry_price) / entry_price
        
        del self.holdings[position_id]
        
        self.trades_executed.append({
            'type': 'sell',
            'time': self.current_step,
            'price': price,
            'qty': position['qty'],
            'pnl': pnl
        })
        self.last_trade_step = self.current_step
        self.total_trade_actions += 1
        self.closed_trade_count += 1
        
        # Reward based on profit/loss
        if pnl > 0:
            reward = REWARD_CONFIG['win_reward'] * min(abs(pnl) * 10, 1.0)
        else:
            reward = REWARD_CONFIG['loss_penalty'] * min(abs(pnl) * 10, 1.0)

        # Penalize noisy exits if not supported by bearish signal.
        if signal_strength < TRADING_CONFIG['entry_confidence_threshold']:
            reward += REWARD_CONFIG['overtrade_penalty']
        
        return reward
    
    def _close_all_positions(self, price: float) -> float:
        """Close all open positions"""
        total_reward = 0.0
        
        for position_id in list(self.holdings.keys()):
            position = self.holdings[position_id]
            proceeds = position['qty'] * price * (1 - TRADING_CONFIG['transaction_cost'])
            self.cash += proceeds
            
            entry_price = position['entry_price']
            pnl = (price - entry_price) / entry_price
            
            if pnl > 0:
                total_reward += REWARD_CONFIG['win_reward'] * min(abs(pnl) * 5, 1.0)
            else:
                total_reward += REWARD_CONFIG['loss_penalty'] * min(abs(pnl) * 5, 1.0)
            self.closed_trade_count += 1
            self.total_trade_actions += 1
        
        self.holdings.clear()
        self.last_trade_step = self.current_step
        return total_reward

    def _signal_strength(self, window: pd.DataFrame) -> float:
        """Estimate bullish conviction from learned pattern + regime context."""
        if window.empty:
            return 0.0

        p = self.pattern_recognizer.get_pattern_feature_vector(window)
        pattern_conf = float(np.clip(p[6], 0.0, 1.0))
        direction = float(np.clip(p[7], -1.0, 1.0))

        r = self.regime_detector.get_regime_features(window)
        bull_prob = float(np.clip(r[0], 0.0, 1.0))
        strength = float(np.clip(r[3], 0.0, 1.0))

        bullish_component = pattern_conf * max(direction, 0.0)
        regime_component = (0.6 * bull_prob) + (0.4 * strength)
        return float(np.clip(0.6 * bullish_component + 0.4 * regime_component, 0.0, 1.0))

    def _assist_action(self, action: int, signal_strength: float) -> int:
        """Nudge policy away from dead-zone behavior while keeping constraints strict."""
        if not TRADING_CONFIG.get('assist_enabled', False):
            return action

        bull_prob = float(self.last_regime_features[0]) if len(self.last_regime_features) > 0 else 0.0
        bear_prob = float(self.last_regime_features[1]) if len(self.last_regime_features) > 1 else 0.0

        if action == 0 and not self.holdings:
            enter_threshold = TRADING_CONFIG['entry_confidence_threshold'] + TRADING_CONFIG.get('assist_entry_margin', 0.10)
            if signal_strength >= enter_threshold and bull_prob >= TRADING_CONFIG.get('assist_bull_threshold', 0.55):
                return 1

        if action == 0 and self.holdings:
            if bear_prob >= TRADING_CONFIG.get('assist_exit_bear_threshold', 0.55):
                return 2

        return action
    def _drawdown_penalty(self) -> float:
        """Apply small penalty when NAV is below its peak."""
        if len(self.nav_history) < 2:
            return 0.0
        nav = np.array(self.nav_history, dtype=np.float64)
        peak = float(np.max(nav))
        if peak <= 0:
            return 0.0
        dd = (peak - float(nav[-1])) / peak
        return -REWARD_CONFIG['drawdown_penalty_scale'] * dd
    
    def _penalize_long_holds(self) -> float:
        """Penalize positions held too long"""
        penalty = 0.0
        
        for position in self.holdings.values():
            hold_duration = self.current_step - position['entry_time']
            
            if hold_duration > REWARD_CONFIG['max_hold_steps']:
                penalty += REWARD_CONFIG['max_hold_penalty']
        
        return penalty
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        position_value = sum(
            pos['qty'] * current_price
            for pos in self.holdings.values()
        )
        return self.cash + position_value
    
    def render(self, mode='human'):
        """Render environment state"""
        print(f"Step {self.current_step}: Capital=${self.capital:.2f}, Positions={len(self.holdings)}")
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        nav = np.array(self.nav_history)
        returns = np.diff(nav) / nav[:-1]
        
        return {
            'total_trades': len(self.trades_executed),
            'final_value': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 390),  # Annualized
            'max_drawdown': self._calculate_max_drawdown(),
            'win_rate': self._calculate_win_rate(),
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        nav = np.array(self.nav_history)
        cummax = np.maximum.accumulate(nav)
        drawdown = (nav - cummax) / cummax
        return float(np.min(drawdown))
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from closed trades"""
        winning = sum(1 for t in self.trades_executed if t.get('pnl', 0) > 0)
        total = sum(1 for t in self.trades_executed if 'pnl' in t)
        
        return winning / total if total > 0 else 0.0


if __name__ == '__main__':
    from data_provider import HFTDataProvider
    
    # Test environment
    provider = HFTDataProvider()
    df = provider.get_historical_data('SPY', days=30)
    
    env = HFTTradingEnv(df)
    obs = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Random test
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        if done:
            print(f"\nEpisode done!")
            print(env.get_stats())
            break
