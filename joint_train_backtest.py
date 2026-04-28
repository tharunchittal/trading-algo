"""Run learned-pattern + RL joint training, then backtest on high-frequency data."""

import time
import os
from pathlib import Path

from data_provider import HFTDataProvider
from rl_agent_trainer import HFTRLAgent
from hft_backtest import HFTBacktester


def main() -> None:
    train_seconds = int(os.getenv("TRAIN_SECONDS", "1800"))
    chunk_steps = int(os.getenv("CHUNK_STEPS", "2048"))
    train_days = int(os.getenv("TRAIN_DAYS", "7"))
    backtest_days = int(os.getenv("BACKTEST_DAYS", "5"))
    symbol = os.getenv("SYMBOL", "SPY")
    model_dir = Path("models/joint_hf")
    model_file = model_dir / "hft_agent.zip"
    pattern_model = "models/pattern/learned_pattern_detector.pkl"

    print("=== JOINT TRAIN + BACKTEST START ===")

    provider = HFTDataProvider()
    # Prioritize high-frequency granularity over very long lookback.
    df = provider.get_historical_data(symbol, days=train_days, interval="1m")
    print(f"Loaded candles for training: {len(df)}")
    if df.empty or len(df) < 500:
        raise SystemExit("Insufficient high-frequency candles for training run")

    agent = HFTRLAgent(model_path=str(model_dir), learned_model_path=pattern_model)
    pattern_ok = agent.train_pattern_model(df, max_samples=3000)
    print(f"Pattern model ready: {pattern_ok} at {pattern_model}")

    agent.create_environments(df)
    agent.build_model("PPO")

    start = time.time()
    chunks = 0
    trained_steps = 0

    while time.time() - start < train_seconds:
        agent.model.learn(
            total_timesteps=chunk_steps,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        chunks += 1
        trained_steps += chunk_steps
        elapsed = time.time() - start
        print(f"Chunk {chunks}: trained_steps={trained_steps}, elapsed_sec={elapsed:.1f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    agent.model.save(str(model_file))
    print(f"RL model saved: {model_file.exists()} at {model_file}")

    eval_results = agent.evaluate(num_episodes=1)
    print("Eval avg_return:", eval_results["avg_return"])
    print("Eval avg_sharpe:", eval_results["avg_sharpe"])
    print("Eval avg_win_rate:", eval_results["avg_win_rate"])

    backtester = HFTBacktester(model_path=str(model_file))
    bt = backtester.run_backtest(symbol, days=backtest_days, interval="1m")
    if bt is None:
        raise SystemExit("Backtest failed to produce results")

    print("Backtest candles:", bt["candles_tested"])
    print("Backtest total_return:", bt["total_return"])
    print("Backtest sharpe_ratio:", bt["sharpe_ratio"])
    print("Backtest win_rate:", bt["win_rate"])
    print("Backtest total_trades:", bt["total_trades"])

    # Explicit cache reuse check (should print cache-hit lines).
    _ = provider.get_historical_data(symbol, days=backtest_days, interval="1m")
    _ = provider.get_historical_data(symbol, days=backtest_days, interval="1m")

    cache_dir = Path(".cache/hft_data")
    cache_files = sorted(str(p) for p in cache_dir.glob(f"{symbol}_1m_*d.pkl"))
    print(f"Cache files for {symbol} 1m:", cache_files)

    print("=== JOINT TRAIN + BACKTEST END ===")


if __name__ == "__main__":
    main()
