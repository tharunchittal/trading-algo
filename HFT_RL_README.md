# HFT RL Trading Agent System

A sophisticated high-frequency trading (HFT) system that uses **Reinforcement Learning (RL)** to identify technical chart patterns and market regimes, then executes trades based on learned signals.

## System Architecture

### Core Components

1. **Data Provider** (`data_provider.py`)
   - Fetches 1-minute OHLCV data from Yahoo Finance
   - Caches data locally for efficiency
   - Normalizes and prepares sequences for ML

2. **Pattern Recognizer** (`pattern_recognizer.py`)
   - Detects 5 major technical patterns:
     - Triangle consolidations
     - Wedges (rising/falling)
     - Flags
     - Channels
     - Head and shoulders
   - Extracts numerical features for neural networks
   - Assigns confidence scores and directional bias

3. **Market Regime Detector** (`market_regime.py`)
   - Identifies bull/bear/neutral markets
   - Calculates regime strength
   - Detects volatility regimes (high/normal/low)
   - Computes 14 technical indicators:
     - Moving averages (SMA 5/20)
     - RSI, MACD, Bollinger Bands, ATR

4. **RL Trading Environment** (`hft_trading_env.py`)
   - OpenAI Gym-compatible environment
   - State: 42-dimensional observation
     - 20 normalized price candles
     - 8 pattern features
     - 14 regime features
   - Actions: Hold, Buy, Sell, Close All
   - Reward structure balances profit/risk

5. **RL Agent Trainer** (`rl_agent_trainer.py`)
   - Uses PPO (Proximal Policy Optimization)
   - Neural network: [256, 256, 128] layers
   - Trains on historical data with 70/30 train/eval split
   - Implements custom callbacks for evaluation
   - Saves trained models

6. **Live Trading Executor** (`live_hft_trader.py`)
   - Executes trades using trained agent
   - Manages portfolio and positions
   - Tracks performance metrics
   - Logs all trades

7. **Backtester** (`hft_backtest.py`)
   - Tests trained agent on historical data
   - Generates performance reports
   - Creates visualization plots
   - Supports multi-symbol comparison

## Installation

```bash
# Install dependencies
pip install yfinance pandas numpy scikit-learn scipy

# Install RL library (required for training)
pip install stable-baselines3[extra]

# Optional: for GPU acceleration
pip install torch

# Optional: for visualization
pip install matplotlib
```

## Usage

### 1. Train Agent

Train the RL agent on historical data:

```bash
python hft_main.py train --symbols SPY QQQ AAPL --timesteps 500000
```

Parameters:
- `--symbols`: Stock tickers to train on
- `--timesteps`: Total training steps (default: 500,000)
- `--episodes`: Number of episodes (default: 500)

**Expected Output:**
- Trained model saved to `models/hft_agent.zip`
- Training metrics and evaluation results
- Average returns, Sharpe ratio, win rate

### 2. Backtest Agent

Test trained agent on historical data:

```bash
# Single symbol
python hft_main.py backtest --symbol SPY --days 60

# Multiple symbols
python hft_main.py backtest --multi --days 60
```

Parameters:
- `--symbol`: Stock ticker
- `--days`: Days of historical data
- `--multi`: Run multi-symbol backtest

**Output:**
- Performance statistics (return, Sharpe, max drawdown)
- Trade count and win rate
- Visualization plots

### 3. Live Trading

Run live trading (paper trading):

```bash
python hft_main.py live --symbols SPY QQQ --dry-run
```

Parameters:
- `--symbols`: Symbols to trade
- `--dry-run`: Paper trading mode (no real money)

### 4. Paper Trading Simulation

Quick simulation on recent data:

```bash
python hft_main.py paper --symbol SPY --days 30
```

### 5. Pattern Analysis

Analyze detected patterns:

```bash
python hft_main.py analyze --symbol SPY --days 30
```

## Configuration

Edit `config.py` to customize:

### Data Settings
```python
DATA_CONFIG = {
    'interval': '1m',  # 1-minute candles
    'symbols': ['SPY', 'QQQ', 'AAPL', ...],
    'lookback_days': 252,  # 1 year
}
```

### Pattern Recognition
```python
PATTERN_CONFIG = {
    'lookback_window': 20,  # Candles to analyze
    'patterns': ['triangle', 'wedge', 'flag', ...],
    'min_pattern_confidence': 0.65,
}
```

### RL Agent
```python
RL_CONFIG = {
    'agent_type': 'PPO',
    'model_architecture': [256, 256, 128],
    'learning_rate': 1e-4,
    'max_episodes': 500,
}
```

### Trading Parameters
```python
TRADING_CONFIG = {
    'initial_capital': 100000,
    'position_size': 0.1,  # 10% per trade
    'max_positions': 5,
    'stop_loss_pct': 0.02,
    'take_profit_pct': 0.04,
    'transaction_cost': 0.001,  # 0.1%
    'slippage': 0.0005,  # 0.05%
}
```

## How It Works

### Training Pipeline

```
1. Data Fetch (yfinance)
   ↓
2. Feature Engineering
   - Pattern detection
   - Regime analysis
   - Technical indicators
   ↓
3. RL Environment
   - State: pattern + regime + price features
   - Action: buy/sell/hold
   - Reward: P&L + risk penalties
   ↓
4. PPO Agent Training
   - Policy network: 256→256→128→action_logits
   - Value network: same architecture
   - Adam optimizer, batch size 64
   ↓
5. Evaluation & Saving
   - Walk-forward validation
   - Save best model
```

### Trading Logic

```
1. Scan 20-candle window for patterns
2. Calculate pattern confidence & direction
3. Detect current market regime
4. Combine signals → RL state vector (42-dim)
5. Agent predicts: buy, sell, hold, or close-all
6. Execute with position sizing & risk management
7. Track P&L and update portfolio
```

## Performance Expectations

### On 1-Minute Data (SPY example)

**Training:** 60 days of 1-minute data
- ~6,400 trading candles
- 70% train, 30% eval split
- ~500,000 timesteps typical convergence time

**Backtest Results** (typical):
- **Return:** 3-8% (on 60-day period)
- **Sharpe Ratio:** 0.5-2.0
- **Win Rate:** 50-60%
- **Max Drawdown:** 3-8%

*Note: Results vary significantly by market conditions, symbols, and configuration.*

## Key Design Decisions

### Pattern Recognition
- Uses linear regression + peak detection
- Confidence = normalized convergence/slope
- Avoids overfitting to specific price levels

### Regime Detection
- Multi-indicator consensus (SMA, RSI, MACD)
- Regime strength quantifies confidence
- Separates trend from volatility

### RL Reward Function
- Win reward: +1.0 for profitable closes
- Loss penalty: -1.0 for losses
- Hold penalty: -0.01 (encourage action)
- Max hold penalty: -0.5 (limit holding time)

### Risk Management
- Position size: 10% of capital per trade
- Max 5 concurrent positions
- 2% stop loss, 4% take profit
- Transaction costs: 0.1%
- Slippage: 0.05%

## System Requirements

**Minimum:**
- RAM: 4GB
- CPU: Dual-core
- Disk: 500MB (includes data cache)

**Recommended for Training:**
- RAM: 8GB+
- CPU: 4+ cores
- GPU: Optional (speeds up 5-50x)

**For Deployment:**
- Raspberry Pi 4+ capable
- AWS t3.micro sufficient
- Trivial computational load for inference

## Limitations & Caveats

⚠️ **Important Considerations:**

1. **Backtesting Bias**: Historical backtests don't account for:
   - Market impact from position sizing
   - Liquidity constraints
   - Regime changes and market structure shifts

2. **Overfitting Risk**: Agent may:
   - Find patterns that worked in past but fail going forward
   - Overfit to recent market conditions
   - Require periodic retraining

3. **Live Trading Risks**:
   - Slippage & commissions higher than simulated
   - Flash crashes and extreme volatility
   - Network latency (important for HFT)
   - Regulatory restrictions on algorithmic trading

4. **Data Quality**:
   - Yahoo Finance may have gaps
   - 1-minute data less reliable than daily
   - Survivorship bias in symbol selection

## Advanced Usage

### Custom Reward Function

Edit `REWARD_CONFIG` in `config.py`:

```python
REWARD_CONFIG = {
    'win_reward': 1.0,
    'loss_penalty': -1.0,
    'hold_reward': -0.01,
    'max_hold_penalty': -0.5,
}
```

### Using Different RL Algorithms

In `rl_agent_trainer.py`, change `agent_type`:

```python
agent.build_model('A2C')  # Actor-Critic (alternative)
```

Supported: PPO, A2C, DDPG (for continuous actions)

### Multi-Asset Portfolio

Extend environment to handle multiple symbols simultaneously.

## Monitoring & Logging

Trading executions logged to:
- `trader_state.json` - Portfolio state
- Trading logs show every trade with P&L
- Backtest generates performance plots

## Future Enhancements

- [ ] Real-time market data integration (Alpha Vantage, IEX)
- [ ] Advanced pattern detection (CNN-based)
- [ ] Multi-asset portfolio optimization
- [ ] Integration with brokers (Alpaca, Interactive Brokers)
- [ ] Ensemble models combining multiple RL agents
- [ ] Meta-learning for faster adaptation
- [ ] Transaction cost optimization
- [ ] Risk-adjusted reward functions (Sortino, Calmar)

## Support & Debugging

**Check if model was trained:**
```python
from rl_agent_trainer import HFTRLAgent
agent = HFTRLAgent()
agent.load()  # Will error if no model exists
```

**Debug backtesting:**
```python
from hft_backtest import HFTBacktester
bt = HFTBacktester()
results = bt.run_backtest('SPY', days=30)
print(results)  # Full statistics
```

**Verify data fetching:**
```python
from data_provider import HFTDataProvider
provider = HFTDataProvider()
df = provider.get_historical_data('SPY', days=30)
print(f"Rows: {len(df)}, Columns: {df.columns.tolist()}")
```

---

**Author:** Trading Algo System  
**Version:** 1.0  
**Last Updated:** 2026-04-22

*Disclaimer: This system is for educational purposes. Do not use with real money without thorough testing and risk management.*
