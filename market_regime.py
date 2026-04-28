"""Market regime detection (Bull/Bear identification)"""

import numpy as np
import pandas as pd
from config import REGIME_CONFIG

class MarketRegimeDetector:
    """Detect market regimes (bull, bear, neutral)"""
    
    def __init__(self):
        self.sma_short = REGIME_CONFIG['sma_short']
        self.sma_long = REGIME_CONFIG['sma_long']
        self.rsi_period = REGIME_CONFIG['rsi_period']
        self.atr_period = REGIME_CONFIG['atr_period']
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe"""
        df = df.copy()
        
        # Moving averages
        df['sma_short'] = df['close'].rolling(self.sma_short).mean()
        df['sma_long'] = df['close'].rolling(self.sma_long).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        # ATR (volatility)
        df['atr'] = self._calculate_atr(df, self.atr_period)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        
        return df
    
    def detect_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime.
        
        Returns:
            'bull', 'bear', or 'neutral'
        """
        if df.empty or len(df) < self.sma_long:
            return 'neutral'
        
        latest = df.iloc[-1]
        
        # SMA-based trend
        sma_bull = latest['close'] > latest['sma_short'] > latest['sma_long']
        sma_bear = latest['close'] < latest['sma_short'] < latest['sma_long']
        
        # RSI-based momentum
        rsi_bull = latest['rsi'] > 50
        rsi_bear = latest['rsi'] < 50
        
        # MACD-based trend
        macd_bull = latest['macd'] > latest['macd_signal']
        macd_bear = latest['macd'] < latest['macd_signal']
        
        # Score each regime
        bull_score = sum([sma_bull, rsi_bull, macd_bull])
        bear_score = sum([sma_bear, rsi_bear, macd_bear])
        
        if bull_score > bear_score:
            return 'bull'
        elif bear_score > bull_score:
            return 'bear'
        else:
            return 'neutral'
    
    def detect_regime_strength(self, df: pd.DataFrame) -> float:
        """
        Measure strength of current regime (0-1).
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Strength score 0-1
        """
        if df.empty or len(df) < self.sma_long:
            return 0.5
        
        latest = df.iloc[-1]
        
        # Distance from EMAs as % of price
        price = latest['close']
        sma_short = latest['sma_short']
        sma_long = latest['sma_long']
        
        distance_short = abs(price - sma_short) / sma_short
        distance_long = abs(price - sma_long) / sma_long
        
        # Strength: how far from moving averages
        strength = min((distance_short + distance_long) / 2, 1.0)
        
        return strength
    
    def detect_volatility_regime(self, df: pd.DataFrame) -> str:
        """
        Detect volatility regime.
        
        Returns:
            'high', 'normal', or 'low'
        """
        if df.empty or 'atr' not in df.columns:
            return 'normal'
        
        # Compare current ATR to average
        avg_atr = df['atr'].mean()
        current_atr = df['atr'].iloc[-1]
        
        ratio = current_atr / avg_atr
        
        if ratio > 1.3:
            return 'high'
        elif ratio < 0.7:
            return 'low'
        else:
            return 'normal'
    
    def get_regime_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract regime features for ML model.
        
        Returns:
            Feature vector (14,) with regime information
        """
        if df.empty or len(df) < self.sma_long:
            return np.zeros(14)
        
        latest = df.iloc[-1]
        
        features = []
        
        # Regime type (one-hot: bull, bear, neutral)
        regime = self.detect_regime(df)
        features.extend([
            1 if regime == 'bull' else 0,
            1 if regime == 'bear' else 0,
            1 if regime == 'neutral' else 0,
        ])
        
        # Regime strength
        strength = self.detect_regime_strength(df)
        features.append(strength)
        
        # Volatility regime
        vol_regime = self.detect_volatility_regime(df)
        features.extend([
            1 if vol_regime == 'high' else 0,
            1 if vol_regime == 'normal' else 0,
            1 if vol_regime == 'low' else 0,
        ])
        
        # Raw indicators
        features.append(latest['rsi'] / 100.0)  # Normalize RSI
        features.append(latest['macd'] / latest['close'] if latest['close'] != 0 else 0)  # MACD ratio
        
        # Bollinger Band position
        bb_range = latest['bb_upper'] - latest['bb_lower']
        if bb_range > 0:
            bb_position = (latest['close'] - latest['bb_lower']) / bb_range
        else:
            bb_position = 0.5
        features.append(np.clip(bb_position, 0, 1))
        
        # Price vs moving averages
        price_vs_sma_short = (latest['close'] - latest['sma_short']) / latest['sma_short']
        price_vs_sma_long = (latest['close'] - latest['sma_long']) / latest['sma_long']
        features.append(np.tanh(price_vs_sma_short))  # Normalized
        features.append(np.tanh(price_vs_sma_long))
        
        return np.array(features)
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr


if __name__ == '__main__':
    from data_provider import HFTDataProvider
    
    # Test regime detection
    provider = HFTDataProvider()
    df = provider.get_historical_data('SPY', days=60)
    
    detector = MarketRegimeDetector()
    df = detector.add_indicators(df)
    
    print(f"Current regime: {detector.detect_regime(df)}")
    print(f"Regime strength: {detector.detect_regime_strength(df):.2f}")
    print(f"Volatility: {detector.detect_volatility_regime(df)}")
    print(f"\nLast 5 rows:")
    print(df[['close', 'sma_short', 'sma_long', 'rsi', 'macd']].tail())
