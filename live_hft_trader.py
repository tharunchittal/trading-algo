"""Live Trading Executor using trained RL agent"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
import json
from pathlib import Path

from hft_trading_env import HFTTradingEnv
from data_provider import HFTDataProvider
from pattern_recognizer import PatternRecognizer
from market_regime import MarketRegimeDetector
from config import TRADING_CONFIG, PATTERN_CONFIG


class LiveHFTTrader:
    """Execute high-frequency trades using trained RL agent"""
    
    def __init__(self, model_path: str = 'models/hft_agent.zip', symbols: List[str] = None):
        """
        Initialize live trader.
        
        Args:
            model_path: Path to trained model
            symbols: Symbols to trade
        """
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
        except:
            print(f"Warning: Could not load model from {model_path}")
            self.model = None
        
        self.symbols = symbols or ['SPY']
        self.data_provider = HFTDataProvider()
        self.pattern_recognizer = PatternRecognizer(learned_model_path=PATTERN_CONFIG.get('learned_model_path'))
        self.regime_detector = MarketRegimeDetector()
        
        # Trading state
        self.portfolio = {
            'cash': TRADING_CONFIG['initial_capital'],
            'positions': {},  # {symbol: {qty, avg_price, entry_time}}
            'trades': [],
            'nav_history': []
        }
        
        self.trading_log = []
    
    def update_market_data(self, symbol: str, latest_bar: Dict = None) -> pd.DataFrame:
        """
        Update market data for symbol.
        
        Args:
            symbol: Stock ticker
            latest_bar: Optional latest OHLCV bar as dict
            
        Returns:
            Updated DataFrame
        """
        df = self.data_provider.get_historical_data(symbol, days=1)
        
        if latest_bar is not None:
            df.loc[len(df)] = latest_bar
        
        return df
    
    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Generate trading signal using patterns and regime.
        
        Args:
            symbol: Stock ticker
            df: OHLCV DataFrame
            
        Returns:
            Signal dict with action and confidence
        """
        if df.empty or len(df) < 50:
            return {'action': 0, 'confidence': 0.0, 'reason': 'Insufficient data'}
        
        # Detect patterns
        if not self.pattern_recognizer.learned_enabled:
            # Train once using recent data so live decisions use learned pattern inference.
            self.pattern_recognizer.train_learned_detector(df, max_samples=1500)

        latest_pattern = self.pattern_recognizer.detect_latest_pattern(df)
        patterns = [latest_pattern] if latest_pattern is not None else []
        
        # Detect regime
        df_with_indicators = self.regime_detector.add_indicators(df.copy())
        regime = self.regime_detector.detect_regime(df_with_indicators)
        regime_strength = self.regime_detector.detect_regime_strength(df_with_indicators)
        
        signal = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'patterns': [(p.name, p.confidence, p.direction) for p in patterns[-3:]],
            'regime': regime,
            'regime_strength': regime_strength,
        }
        
        # Score signals
        bullish_score = 0
        bearish_score = 0
        
        # Pattern signals
        for pattern in patterns[-3:]:
            if pattern.direction == 'bullish':
                bullish_score += pattern.confidence
            else:
                bearish_score += pattern.confidence
        
        # Regime signals
        if regime == 'bull':
            bullish_score += regime_strength
        elif regime == 'bear':
            bearish_score += regime_strength
        
        # Determine action
        min_conf = TRADING_CONFIG.get('entry_confidence_threshold', 0.58)

        if bullish_score > bearish_score + min_conf and bullish_score > min_conf:
            signal['action'] = 1  # Buy
            signal['confidence'] = min(bullish_score / 3, 1.0)
            signal['reason'] = f"Bullish patterns ({bullish_score:.2f}), regime: {regime}"
        
        elif bearish_score > bullish_score + min_conf and bearish_score > min_conf:
            signal['action'] = 2  # Sell
            signal['confidence'] = min(bearish_score / 3, 1.0)
            signal['reason'] = f"Bearish patterns ({bearish_score:.2f}), regime: {regime}"
        
        else:
            signal['action'] = 0  # Hold
            signal['confidence'] = 0.0
            signal['reason'] = 'Neutral signal'
        
        return signal
    
    def execute_trade(self, symbol: str, action: int, price: float) -> Dict:
        """
        Execute trade action.
        
        Args:
            symbol: Stock ticker
            action: 0=hold, 1=buy, 2=sell, 3=close_all
            price: Current price
            
        Returns:
            Trade execution record
        """
        execution = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'price': price,
            'status': 'pending',
            'details': {}
        }
        
        if action == 1:  # BUY
            execution = self._execute_buy(symbol, price, execution)
        
        elif action == 2:  # SELL
            execution = self._execute_sell(symbol, price, execution)
        
        elif action == 3:  # CLOSE ALL
            execution = self._close_all(symbol, price, execution)
        
        self.trading_log.append(execution)
        return execution
    
    def _execute_buy(self, symbol: str, price: float, execution: Dict) -> Dict:
        """Execute buy order"""
        position_size = TRADING_CONFIG['position_size'] * self.portfolio['cash']
        qty = int(position_size / price)
        
        if qty > 0:
            cost = qty * price * (1 + TRADING_CONFIG['transaction_cost'])
            
            if cost <= self.portfolio['cash']:
                self.portfolio['cash'] -= cost
                
                if symbol not in self.portfolio['positions']:
                    self.portfolio['positions'][symbol] = {
                        'qty': 0,
                        'avg_price': 0,
                        'entry_time': datetime.now().isoformat()
                    }
                
                pos = self.portfolio['positions'][symbol]
                pos['qty'] += qty
                pos['avg_price'] = price
                
                execution['status'] = 'executed'
                execution['details'] = {
                    'qty': qty,
                    'cost': cost,
                    'position_size': position_size
                }
            else:
                execution['status'] = 'insufficient_funds'
                execution['details'] = {'required': cost, 'available': self.portfolio['cash']}
        else:
            execution['status'] = 'insufficient_capital'
        
        return execution
    
    def _execute_sell(self, symbol: str, price: float, execution: Dict) -> Dict:
        """Execute sell order"""
        if symbol not in self.portfolio['positions']:
            execution['status'] = 'no_position'
            return execution
        
        pos = self.portfolio['positions'][symbol]
        qty = pos['qty']
        
        if qty > 0:
            proceeds = qty * price * (1 - TRADING_CONFIG['transaction_cost'] - TRADING_CONFIG['slippage'])
            self.portfolio['cash'] += proceeds
            
            pnl = (price - pos['avg_price']) * qty
            pnl_pct = (price - pos['avg_price']) / pos['avg_price']
            
            execution['status'] = 'executed'
            execution['details'] = {
                'qty': qty,
                'proceeds': proceeds,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }
            entry_time = datetime.fromisoformat(pos['entry_time'])
            hold_seconds = (datetime.now() - entry_time).total_seconds()
            
            # Record trade
            self.portfolio['trades'].append({
                'symbol': symbol,
                'entry_price': pos['avg_price'],
                'exit_price': price,
                'qty': qty,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'duration_seconds': hold_seconds
            })
            
            del self.portfolio['positions'][symbol]
        
        return execution
    
    def _close_all(self, symbol: str, price: float, execution: Dict) -> Dict:
        """Close all positions in symbol"""
        if symbol not in self.portfolio['positions']:
            execution['status'] = 'no_position'
            return execution
        
        execution = self._execute_sell(symbol, price, execution)
        execution['action'] = 3
        
        return execution
    
    def get_portfolio_value(self, prices: Dict[str, float] = None) -> float:
        """
        Get current portfolio value.
        
        Args:
            prices: Dict of {symbol: price}
            
        Returns:
            Total portfolio value
        """
        value = self.portfolio['cash']
        
        if prices:
            for symbol, price in prices.items():
                if symbol in self.portfolio['positions']:
                    qty = self.portfolio['positions'][symbol]['qty']
                    value += qty * price
        
        return value
    
    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics"""
        trades = self.portfolio['trades']
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_trade_pnl': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
            }
        
        total_pnl = sum(t['pnl'] for t in trades)
        winning = sum(1 for t in trades if t['pnl'] > 0)
        losing = sum(1 for t in trades if t['pnl'] < 0)
        
        return {
            'total_trades': len(trades),
            'winning_trades': winning,
            'losing_trades': losing,
            'win_rate': winning / len(trades) if trades else 0.0,
            'total_pnl': total_pnl,
            'avg_trade_pnl': total_pnl / len(trades) if trades else 0.0,
            'largest_win': max((t['pnl'] for t in trades), default=0),
            'largest_loss': min((t['pnl'] for t in trades), default=0),
        }
    
    def save_state(self, filepath: str = 'trader_state.json'):
        """Save trader state to file"""
        state = {
            'portfolio': self.portfolio,
            'stats': self.get_performance_stats(),
            'last_updated': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        print(f"Saved trader state to {filepath}")
    
    def print_summary(self):
        """Print trading summary"""
        stats = self.get_performance_stats()
        portfolio_value = self.get_portfolio_value()
        
        print("\n" + "="*60)
        print("LIVE TRADING SUMMARY")
        print("="*60)
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Cash: ${self.portfolio['cash']:,.2f}")
        print(f"Open Positions: {len(self.portfolio['positions'])}")
        print()
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Winning Trades: {stats['winning_trades']}")
        print(f"Losing Trades: {stats['losing_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1%}")
        print()
        print(f"Total P&L: ${stats['total_pnl']:,.2f}")
        print(f"Average Trade P&L: ${stats['avg_trade_pnl']:,.2f}")
        print(f"Largest Win: ${stats['largest_win']:,.2f}")
        print(f"Largest Loss: ${stats['largest_loss']:,.2f}")
        print("="*60 + "\n")


if __name__ == '__main__':
    # Example usage
    trader = LiveHFTTrader(symbols=['SPY', 'QQQ'])
    
    # Get data and generate signals
    for symbol in trader.symbols:
        df = trader.update_market_data(symbol)
        signal = trader.generate_signal(symbol, df)
        
        print(f"\n{symbol} Signal:")
        print(f"  Action: {['HOLD', 'BUY', 'SELL', 'CLOSE_ALL'][signal['action']]}")
        print(f"  Confidence: {signal['confidence']:.2f}")
        print(f"  Reason: {signal['reason']}")
    
    trader.print_summary()
