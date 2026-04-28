"""Data provider for high-frequency market data"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from pathlib import Path
from config import DATA_CONFIG

class HFTDataProvider:
    """Fetch and cache high-frequency OHLCV data"""
    
    def __init__(self):
        self.cache_dir = Path(DATA_CONFIG['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache = {}
        
    def get_historical_data(self, symbol: str, days: int = None, interval: str = None) -> pd.DataFrame:
        """
        Fetch high-frequency OHLCV data from yfinance.
        
        Args:
            symbol: Stock ticker (e.g., 'SPY')
            days: Number of days back (default: config lookback_days)
            interval: Candle interval (default: config interval)
            
        Returns:
            DataFrame with OHLCV data
        """
        days = days or DATA_CONFIG['lookback_days']
        interval = interval or DATA_CONFIG['interval']
        
        # Check cache first
        cache_file = self.cache_dir / f"{symbol}_{interval}_{days}d.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
                print(f"Cache hit for {symbol} {interval} ({days}d): {len(df)} candles")
                return df
            except:
                pass
        
        # Fetch from yfinance
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"Fetching {symbol} {interval} data from {start_date.date()} to {end_date.date()}...")
        try:
            if interval == '1m' and days > 8:
                # Yahoo limits 1m downloads to ~8 days per request.
                # Pull in chunks and merge so longer lookbacks still work.
                parts = []
                chunk_start = start_date
                while chunk_start < end_date:
                    chunk_end = min(chunk_start + timedelta(days=7), end_date)
                    part = self._download_ohlcv(symbol, chunk_start, chunk_end, interval)
                    if not part.empty:
                        parts.append(part)
                    chunk_start = chunk_end

                if not parts:
                    print(f"Warning: No data for {symbol}")
                    return pd.DataFrame()

                df = pd.concat(parts).sort_index()
                df = df[~df.index.duplicated(keep='last')]
            else:
                df = self._download_ohlcv(symbol, start_date, end_date, interval)
            
            if df.empty:
                print(f"Warning: No data for {symbol}")
                return pd.DataFrame()
            
            df = df.dropna()
            
            # Cache it
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            
            print(f"Cached {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def _download_ohlcv(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
        """Download and normalize OHLCV from yfinance for a single date range."""
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            prepost=False
        )

        if df.empty:
            return df

        # Handle yfinance schema changes (Adj Close, MultiIndex, etc.)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]
        if 'adj_close' in df.columns and 'close' not in df.columns:
            df['close'] = df['adj_close']

        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"missing columns {missing}")

        return df[required]
    
    def get_multiple_symbols(self, symbols: list = None) -> dict:
        """Fetch data for multiple symbols"""
        symbols = symbols or DATA_CONFIG['symbols']
        data = {}
        
        for symbol in symbols:
            df = self.get_historical_data(symbol)
            if not df.empty:
                data[symbol] = df
        
        return data
    
    def normalize_ohlcv(self, df: pd.DataFrame) -> np.ndarray:
        """
        Normalize OHLCV data to [0, 1] range for neural network.
        
        Args:
            df: DataFrame with open, high, low, close, volume columns
            
        Returns:
            Normalized numpy array (n_samples, 5)
        """
        ohlcv = df[['open', 'high', 'low', 'close', 'volume']].values
        
        # Min-max normalization per column
        mins = ohlcv.min(axis=0)
        maxs = ohlcv.max(axis=0)
        normalized = (ohlcv - mins) / (maxs - mins + 1e-8)
        
        return normalized
    
    def create_sequences(self, df: pd.DataFrame, seq_len: int = 20) -> tuple:
        """
        Create sequences for time series learning.
        
        Args:
            df: DataFrame with OHLCV data
            seq_len: Sequence length
            
        Returns:
            (sequences, targets) - shape (n_samples, seq_len, 5) and (n_samples,)
        """
        normalized = self.normalize_ohlcv(df)
        
        sequences = []
        targets = []
        
        for i in range(len(normalized) - seq_len):
            seq = normalized[i:i+seq_len]
            # Target: 1 if next close is higher, 0 otherwise
            target = 1 if normalized[i+seq_len, 3] > normalized[i+seq_len-1, 3] else 0
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add returns column to dataframe"""
        df['returns'] = df['close'].pct_change()
        return df
    
    def get_latest_price(self, symbol: str) -> float:
        """Get latest closing price"""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info.get('currentPrice', ticker.history(period='1d')['Close'].iloc[-1])
        except:
            return None


if __name__ == '__main__':
    # Test data provider
    provider = HFTDataProvider()
    
    # Fetch data for primary symbols
    data = provider.get_multiple_symbols()
    
    for symbol, df in data.items():
        print(f"\n{symbol}: {len(df)} candles")
        print(df.head())
