# HFT RL Trading Agent - Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install -r hft_requirements.txt
```

**Troubleshooting:**
- If you get permission errors, add `--user` flag
- For ARM-based systems (Oracle A1), might need to build some packages from source

### 2. Verify Installation

```python
# Test imports
python -c "from rl_agent_trainer import HFTRLAgent; print('✓ RL modules OK')"
python -c "from data_provider import HFTDataProvider; print('✓ Data modules OK')"
python -c "import stable_baselines3; print('✓ RL library OK')"
```

### 3. Fetch Some Data

```bash
# Download 30 days of 1-minute SPY data
python data_provider.py
```

This will:
- Fetch data from Yahoo Finance
- Cache locally in `.cache/hft_data/`
- Show candle counts

## First Training (30 minutes on CPU)

### Basic Training

```bash
python hft_main.py train --symbols SPY --timesteps 100000
```

This will:
1. Fetch 1 year of 1-minute SPY data
2. Split 70% train, 30% eval
3. Train PPO agent for ~100,000 steps (~15 min on CPU)
4. Evaluate and save model to `models/hft_agent.zip`

### Track Progress

Training output shows:
```
Step 10000: Eval Reward = 45.32
Step 20000: Eval Reward = 62.15
Step 30000: Eval Reward = 58.47
...
Training completed!
Model saved to models/hft_agent.zip
```

### Multi-Symbol Training (Optional)

```bash
python hft_main.py train --symbols SPY QQQ AAPL --timesteps 200000
```

Uses first symbol's data but learns from multiple trading patterns.

## Testing the Trained Agent

### Quick Paper Trade

```bash
python hft_main.py paper --symbol SPY --days 7
```

Runs quick simulation, shows:
```
Simulation Results:
  Final Value: $103,450.23
  Return: 3.45%
  Sharpe: 1.23
  Win Rate: 58.5%
  Trades: 127
```

### Full Backtest

```bash
python hft_main.py backtest --symbol SPY --days 30
```

Generates detailed report + plot (`backtest_SPY.png`)

### Backtest All Symbols

```bash
python hft_main.py backtest --multi --days 30
```

Tests on SPY, QQQ, AAPL together.

## Generate Trading Signals

### Check Current Signals

```bash
python hft_main.py live --symbols SPY QQQ --dry-run
```

Output:
```
SPY Signal:
  Action: BUY
  Confidence: 0.78
  Reason: Bullish patterns (1.85), regime: bull

QQQ Signal:
  Action: HOLD
  Confidence: 0.00
  Reason: Neutral signal
```

### Analyze Patterns

```bash
python hft_main.py analyze --symbol SPY --days 30
```

Shows:
```
Detected 47 patterns in SPY:
  triangle: 18
  wedge_rising: 12
  flag: 11
  channel: 6

Current Regime: bull (strength: 0.73)
```

## Understanding Results

### Key Metrics

| Metric | Good | Acceptable | Risky |
|--------|------|-----------|-------|
| **Return** | >2% (60d) | 1-2% | <0% |
| **Sharpe Ratio** | >1.0 | 0.5-1.0 | <0.5 |
| **Win Rate** | >55% | 50-55% | <50% |
| **Max Drawdown** | <5% | 5-10% | >10% |

### Example Output Interpretation

```
BACKTEST RESULTS: SPY
Period: 2024-04-20 to 2024-06-19
Candles Tested: 6,850 (60 days)

Initial Capital: $100,000.00
Final Capital: $104,500.00
Total Return: 4.50%           ← Goal: >2%

Total Trades: 145
Win Rate: 57.2%               ← Goal: >55%

Sharpe Ratio: 1.42            ← Goal: >1.0
Max Drawdown: -3.21%          ← Goal: <5%
```

This is a **good result** – profitable with acceptable risk.

## Next Steps

### 1. Experiment with Configuration

Edit `config.py`:

```python
# Try longer position holds
REWARD_CONFIG['max_hold_steps'] = 200  # was 100

# Try smaller positions
TRADING_CONFIG['position_size'] = 0.05  # was 0.1

# Try stricter stops
TRADING_CONFIG['stop_loss_pct'] = 0.01  # was 0.02
```

Then retrain:
```bash
python hft_main.py train --timesteps 100000
```

### 2. Test Different Symbols

```bash
for symbol in AAPL MSFT NVDA TSLA; do
  echo "Testing $symbol..."
  python hft_main.py backtest --symbol $symbol --days 30
done
```

### 3. Longer Training

For better results, train longer:
```bash
# Standard: ~30 min on CPU
python hft_main.py train --timesteps 500000

# Extended: ~2 hours on CPU
python hft_main.py train --timesteps 2000000
```

### 4. Deploy Live (With Caution)

Once confident:
```bash
# Paper trading (simulated)
python hft_main.py live --symbols SPY QQQ --dry-run

# Review trader state
cat trader_state.json
```

## Troubleshooting

### "No data for SPY"
- Check internet connection
- Yahoo Finance might be rate-limiting
- Wait 30 seconds and retry

### Out of Memory
- Reduce `lookback_days` in config
- Use fewer symbols
- Run on machine with more RAM

### Model Training Slow
- Normal on CPU (30 min for 100k steps)
- Disable evaluation callbacks temporarily
- Consider cloud GPU (Google Colab, Kaggle, etc.)

### Agent Doesn't Learn
- Model needs more training time
- Try simpler symbol (SPY > NVDA)
- Check configuration in `config.py`
- Verify data is being fetched correctly

## File Structure

```
trading-algo/
├── config.py                  # Configuration file
├── data_provider.py           # Data fetching
├── pattern_recognizer.py      # Pattern detection
├── market_regime.py           # Regime detection
├── hft_trading_env.py         # RL environment
├── rl_agent_trainer.py        # Training code
├── live_hft_trader.py         # Live execution
├── hft_backtest.py            # Backtesting
├── hft_main.py                # CLI interface
├── HFT_RL_README.md           # Full documentation
├── hft_requirements.txt       # Dependencies
├── .cache/hft_data/           # Cached market data
└── models/                    # Trained models
    └── hft_agent.zip          # Trained agent
```

## Performance on Different Machines

### Local Development (MacBook M1, 16GB RAM)
- Data fetch: ~5 seconds
- Training (100k steps): ~15 minutes
- Backtest (60 days): ~8 seconds

### Cloud CPU (t3.medium, 4GB RAM)
- Data fetch: ~10 seconds
- Training (100k steps): ~25 minutes
- Backtest: ~15 seconds

### Oracle Ampere A1 (Always Free)
- Data fetch: ~20 seconds
- Training (100k steps): ~45 minutes
- Backtest: ~25 seconds

### With GPU (RTX 4090, 24GB VRAM)
- Training (100k steps): ~2 minutes
- Backtest: <1 second

## Important Disclaimers

⚠️ **READ CAREFULLY:**

1. **Backtesting Results Don't Predict Future Performance**
   - Past performance ≠ future results
   - Markets change, patterns break down
   - Overfitting is a real risk

2. **Slippage & Costs Are Minimized**
   - Real trading has higher costs
   - 1-minute candles are noisy
   - Liquidity may be an issue

3. **Start Small**
   - Don't trade real money without months of paper trading
   - Begin with $100-$500
   - Use position sizing: never risk >1% per trade

4. **Continuous Monitoring**
   - Agent needs retraining monthly
   - Market regimes shift
   - Adjust parameters based on live performance

5. **Broker Restrictions**
   - High-frequency strategies may violate broker TOS
   - Day trading rules apply (PDT rule in US)
   - Algorithms may be flagged as suspicious

## Getting Help

**Debug Mode:**
```python
# In Python shell
from data_provider import HFTDataProvider
from pattern_recognizer import PatternRecognizer

provider = HFTDataProvider()
df = provider.get_historical_data('SPY', days=10)
print(df.shape)  # Check data

recognizer = PatternRecognizer()
patterns = recognizer.detect_patterns(df)
print(f"Patterns: {len(patterns)}")  # Check pattern detection
```

**Common Issues:**
- Model not improving? Try `learning_rate=5e-4`
- Win rate too low? Increase `min_pattern_confidence=0.75`
- Too few trades? Increase `position_size=0.15`

---

**Next:** Read `HFT_RL_README.md` for full documentation

**Time Estimate:** 30 min setup + 30 min first training = 1 hour to first results ⏱️
