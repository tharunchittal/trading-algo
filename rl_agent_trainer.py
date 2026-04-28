#!/usr/bin/env python3
"""
rl_agent_trainer.py – DQN-based reinforcement learning trading agent.

Sub-commands
------------
diagnose  Check for issues that would prevent training/trading.
train     Train the DQN on historical daily data; saves model.pt
backtest  Load a saved model and run an equity-curve backtest.

Quick-start examples
--------------------
# 1) Diagnose first:
python rl_agent_trainer.py diagnose

# 2) Train for 3+ hours across a universe of tech stocks (10 yr history):
python rl_agent_trainer.py train \\
    --tickers AAPL MSFT NVDA GOOGL META AMZN TSLA AMD INTC ORCL \\
              CRM ADBE NFLX AVGO QCOM MU ANET PANW SNOW PLTR \\
    --start 2015-01-01 --end 2024-12-31 \\
    --train-hours 3.0 --model-path model.pt

# 3) Backtest on a held-out period:
python rl_agent_trainer.py backtest \\
    --tickers AAPL MSFT NVDA GOOGL META \\
    --start 2024-01-01 --end 2025-01-01 \\
    --model-path model.pt --initial-cash 100000
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional imports – graceful fallback messages
# ---------------------------------------------------------------------------
try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = Path("cache")
HOLD, BUY, SELL = 0, 1, 2
N_ACTIONS = 3

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "TSLA", "AMD", "INTC", "ORCL",
    "CRM", "ADBE", "NFLX", "AVGO", "QCOM",
    "MU", "ANET", "PANW", "SNOW", "PLTR",
]

# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

def _cache_path(symbol: str, interval: str, start: str, end: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    safe = f"{symbol}_{interval}_{start}_{end}.pkl".replace(":", "-")
    return CACHE_DIR / safe


def download_ohlcv(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance with disk caching."""
    cp = _cache_path(symbol, interval, start, end)
    if not force_refresh and cp.exists():
        with open(cp, "rb") as fh:
            return pickle.load(fh)

    if not _YF_OK:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError(
            f"No data returned for {symbol} ({start}→{end}, {interval}). "
            "Check the ticker, date range, and internet connection."
        )

    df = df.sort_index()

    # yfinance ≥0.2 uses MultiIndex columns (field, ticker) – flatten to plain names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    with open(cp, "wb") as fh:
        pickle.dump(df, fh)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _bollinger(series: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    pct_b = (series - lower) / (upper - lower + 1e-9)
    width = (upper - lower) / (sma + 1e-9)
    return pct_b, width, sma


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo, pc = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat(
        [hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ~25 normalised features per bar.
    Returns a DataFrame aligned with df; rows with NaN are dropped.
    """
    close = df["Close"]
    volume = df["Volume"]
    feat = pd.DataFrame(index=df.index)

    # Price returns (log)
    for lag in [1, 2, 3, 5, 10, 20, 60]:
        feat[f"ret_{lag}"] = np.log(close / close.shift(lag))

    # Volume ratio
    vol_ma = volume.rolling(20).mean()
    feat["vol_ratio"] = volume / (vol_ma + 1e-9)

    # RSI
    feat["rsi"] = _rsi(close, 14) / 100.0  # normalise 0-1

    # MACD (normalise by price)
    macd, sig = _macd(close)
    feat["macd"] = macd / (close + 1e-9)
    feat["macd_signal"] = sig / (close + 1e-9)
    feat["macd_hist"] = (macd - sig) / (close + 1e-9)

    # Bollinger
    pct_b, bb_width, sma20 = _bollinger(close, 20)
    feat["pct_b"] = pct_b
    feat["bb_width"] = bb_width

    # ATR ratio
    atr = _atr(df, 14)
    feat["atr_ratio"] = atr / (close + 1e-9)

    # 52-week position
    high52 = close.rolling(252).max()
    low52 = close.rolling(252).min()
    feat["week52_pos"] = (close - low52) / (high52 - low52 + 1e-9)

    # Trend slope (linear regression slope over last 20 bars, normalised)
    def _slope(s: pd.Series, w: int) -> pd.Series:
        x = np.arange(w, dtype=float)
        x -= x.mean()
        result = s.rolling(w).apply(
            lambda y: float(np.polyfit(x, y, 1)[0]) / (abs(y.mean()) + 1e-9),
            raw=True,
        )
        return result

    feat["slope20"] = _slope(close, 20)

    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.dropna(inplace=True)
    # Clip extreme values
    feat = feat.clip(-10, 10)
    return feat


N_FEATURES = 17  # 7 returns + vol_ratio + rsi + macd×3 + bollinger×2 + atr + week52_pos + slope


# ---------------------------------------------------------------------------
# DQN Neural Network
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    def __init__(self, n_features: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self._buf: deque = deque(maxlen=capacity)

    def push(self, t: Transition):
        self._buf.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Trading environment (single-ticker episode)
# ---------------------------------------------------------------------------

class TradingEnv:
    """
    Thin Gym-like environment for a single-ticker daily trading episode.

    Actions: HOLD=0, BUY=1, SELL=2
    State:   feature vector of shape (N_FEATURES + 3,)
               – features + [position_ratio, unrealised_pnl_pct, portfolio_return]
    """

    def __init__(
        self,
        features: pd.DataFrame,
        closes: pd.Series,
        initial_cash: float = 100_000.0,
        fee_rate: float = 0.001,
        slippage_bps: float = 5.0,
        buy_fraction: float = 0.20,
        sell_fraction: float = 0.50,
    ):
        self.features = features.values.astype(np.float32)
        self.closes = closes.values.astype(np.float32)
        assert len(self.features) == len(self.closes)

        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.slip = slippage_bps / 10_000.0
        self.buy_fraction = buy_fraction
        self.sell_fraction = sell_fraction

        self.n_steps = len(self.features)
        self.reset()

    @property
    def state_dim(self) -> int:
        return self.features.shape[1] + 3  # features + portfolio stats

    def reset(self) -> np.ndarray:
        self.t = 0
        self.cash = float(self.initial_cash)
        self.shares = 0.0
        self.entry_price = 0.0
        return self._obs()

    def _obs(self) -> np.ndarray:
        feat = self.features[self.t]
        price = self.closes[self.t]
        equity = self.cash + self.shares * price
        pos_ratio = (self.shares * price) / (equity + 1e-9)
        unreal_pnl = (price - self.entry_price) / (self.entry_price + 1e-9) if self.shares > 0 else 0.0
        port_return = equity / self.initial_cash - 1.0
        extra = np.array([pos_ratio, unreal_pnl, port_return], dtype=np.float32)
        return np.concatenate([feat, extra])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        price = float(self.closes[self.t])
        prev_equity = self.cash + self.shares * price

        if action == BUY and self.cash > 1.0:
            invest = self.cash * self.buy_fraction
            fill = price * (1.0 + self.slip)
            fee = invest * self.fee_rate
            net_invest = invest - fee
            bought = net_invest / fill if fill > 0 else 0.0
            if bought > 0:
                self.cash -= invest
                if self.shares == 0.0:
                    self.entry_price = fill
                else:
                    total = self.shares + bought
                    self.entry_price = (self.entry_price * self.shares + fill * bought) / total
                self.shares += bought

        elif action == SELL and self.shares > 0:
            sell_shares = self.shares * self.sell_fraction
            fill = price * (1.0 - self.slip)
            proceeds = sell_shares * fill
            fee = proceeds * self.fee_rate
            self.cash += proceeds - fee
            self.shares -= sell_shares
            if self.shares < 1e-9:
                self.shares = 0.0
                self.entry_price = 0.0

        self.t += 1
        done = self.t >= self.n_steps

        if done:
            # Liquidate
            if self.shares > 0:
                fill = float(self.closes[-1]) * (1.0 - self.slip)
                proceeds = self.shares * fill * (1.0 - self.fee_rate)
                self.cash += proceeds
                self.shares = 0.0

        new_price = float(self.closes[min(self.t, self.n_steps - 1)])
        new_equity = self.cash + self.shares * new_price
        reward = (new_equity - prev_equity) / (prev_equity + 1e-9)
        # Scale reward for stable gradients
        reward = float(np.clip(reward * 100.0, -5.0, 5.0))

        obs = self._obs() if not done else np.zeros(self.state_dim, dtype=np.float32)
        return obs, reward, done

    def final_equity(self) -> float:
        price = float(self.closes[-1])
        return self.cash + self.shares * price


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int = N_ACTIONS,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 256,
        target_update_freq: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.9999,
        hidden: int = 256,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps = 0

        self.policy_net = DQN(state_dim, n_actions, hidden).to(self.device)
        self.target_net = DQN(state_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(capacity=500_000)
        self.total_loss = 0.0
        self.loss_count = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(t)
            return int(q.argmax(dim=1).item())

    def store(self, t: Transition):
        self.buffer.push(t)

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        states = torch.tensor(
            np.array([t.state for t in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            [t.action for t in batch], dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            [t.done for t in batch], dtype=torch.float32, device=self.device
        )

        # Current Q
        q_current = self.policy_net(states).gather(1, actions).squeeze(1)

        # Target Q (Double DQN)
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Epsilon decay
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        self.steps += 1

        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        l = loss.item()
        self.total_loss += l
        self.loss_count += 1
        return l

    def avg_loss(self) -> float:
        if self.loss_count == 0:
            return 0.0
        v = self.total_loss / self.loss_count
        self.total_loss = 0.0
        self.loss_count = 0
        return v

    def save(self, path: str):
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "eps": self.eps,
                "steps": self.steps,
                "state_dim": self.policy_net.net[0].in_features,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DQNAgent":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        state_dim = ckpt["state_dim"]
        agent = cls(state_dim=state_dim, device=device)
        agent.policy_net.load_state_dict(ckpt["policy_state_dict"])
        agent.target_net.load_state_dict(ckpt["target_state_dict"])
        agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        agent.eps = ckpt.get("eps", agent.eps_end)
        agent.steps = ckpt.get("steps", 0)
        return agent


# ---------------------------------------------------------------------------
# Data preparation helper
# ---------------------------------------------------------------------------

def prepare_ticker_data(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Download, compute features, return (features_df, closes_series) or None on error."""
    try:
        df = download_ohlcv(symbol, start, end, interval)
    except Exception as exc:
        print(f"  [WARN] {symbol}: {exc}")
        return None

    if len(df) < 100:
        print(f"  [WARN] {symbol}: only {len(df)} bars – skipping")
        return None

    try:
        features = build_features(df)
    except Exception as exc:
        print(f"  [WARN] {symbol}: feature build failed – {exc}")
        return None

    # Align closes with features index
    closes = df["Close"].loc[features.index]

    return features, closes


# ---------------------------------------------------------------------------
# TRAIN sub-command
# ---------------------------------------------------------------------------

def cmd_train(args):
    if not _TORCH_OK:
        print("ERROR: PyTorch is required for training. Install with: pip install torch")
        sys.exit(1)
    if not _YF_OK:
        print("ERROR: yfinance is required for data. Install with: pip install yfinance")
        sys.exit(1)

    tickers: List[str] = [t.upper() for t in args.tickers]
    device = "cuda" if (args.device == "auto" and _TORCH_OK and torch.cuda.is_available()) else "cpu"
    if args.device not in ("auto", "cpu"):
        device = args.device

    print(f"\n{'='*60}")
    print(f"DQN Trading Agent – Training Run")
    print(f"{'='*60}")
    print(f"Tickers  : {', '.join(tickers)}")
    print(f"History  : {args.start} → {args.end}")
    print(f"Train hrs: {args.train_hours:.1f}h  (≈{args.train_hours*3600:.0f}s)")
    print(f"Device   : {device}")
    print(f"Model out: {args.model_path}")
    print()

    # ------------------------------------------------------------------
    # Download & prepare all ticker data
    # ------------------------------------------------------------------
    print("Downloading & preparing data (caching enabled)…")
    ticker_data: List[Tuple[pd.DataFrame, pd.Series]] = []
    for sym in tickers:
        result = prepare_ticker_data(sym, args.start, args.end)
        if result is not None:
            ticker_data.append(result)
            print(f"  ✓ {sym}: {len(result[0])} bars")

    if not ticker_data:
        print("ERROR: No usable data. Check your tickers and internet connection.")
        sys.exit(1)

    print(f"\n{len(ticker_data)}/{len(tickers)} tickers ready.\n")

    # Determine state dim from first ticker
    sample_feat, _ = ticker_data[0]
    state_dim = sample_feat.shape[1] + 3  # features + portfolio stats

    # ------------------------------------------------------------------
    # Initialise agent (or resume from existing model)
    # ------------------------------------------------------------------
    if os.path.exists(args.model_path) and not args.reset:
        print(f"Resuming from checkpoint: {args.model_path}")
        agent = DQNAgent.load(args.model_path, device=device)
        # Verify state_dim matches
        if agent.policy_net.net[0].in_features != state_dim:
            print("WARNING: Checkpoint state_dim mismatch – starting fresh.")
            agent = DQNAgent(state_dim=state_dim, device=device)
    else:
        agent = DQNAgent(state_dim=state_dim, device=device)

    # ------------------------------------------------------------------
    # Training loop – episode = full pass through one ticker's data
    # ------------------------------------------------------------------
    deadline = time.time() + args.train_hours * 3600.0
    episode = 0
    best_return = -np.inf
    print_every = 50  # episodes between progress prints

    print(f"Training for {args.train_hours:.1f} hours… (Ctrl-C to stop early)\n")

    try:
        while time.time() < deadline:
            # Pick a random ticker dataset
            features, closes = random.choice(ticker_data)

            # Random start window (min 200, max full length)
            max_start = max(0, len(features) - 200)
            start_idx = random.randint(0, max_start)
            end_idx = min(len(features), start_idx + random.randint(200, len(features) - start_idx + 1))

            feat_slice = features.iloc[start_idx:end_idx]
            close_slice = closes.iloc[start_idx:end_idx]

            env = TradingEnv(feat_slice, close_slice, fee_rate=args.fee_rate)
            state = env.reset()
            ep_reward = 0.0

            while True:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.store(Transition(state, action, reward, next_state, done))
                agent.learn()
                ep_reward += reward
                state = next_state
                if done:
                    break

            ep_return = env.final_equity() / env.initial_cash - 1.0
            episode += 1

            if ep_return > best_return:
                best_return = ep_return
                agent.save(args.model_path)  # save best so far

            # Periodic checkpoint (every 500 episodes regardless of best)
            if episode % 500 == 0:
                agent.save(args.model_path)

            if episode % print_every == 0:
                elapsed = time.time() - (deadline - args.train_hours * 3600.0)
                remaining = max(0, deadline - time.time())
                avg_loss = agent.avg_loss()
                print(
                    f"  ep {episode:>6d} | "
                    f"ep_ret {ep_return:+.2%} | "
                    f"best {best_return:+.2%} | "
                    f"eps {agent.eps:.4f} | "
                    f"loss {avg_loss:.5f} | "
                    f"buf {len(agent.buffer):>7d} | "
                    f"elapsed {elapsed/60:.1f}m | "
                    f"left {remaining/60:.1f}m"
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Final save
    agent.save(args.model_path)
    elapsed_total = time.time() - (deadline - args.train_hours * 3600.0)
    print(f"\n{'='*60}")
    print(f"Training complete: {episode} episodes in {elapsed_total/60:.1f} minutes")
    print(f"Best episode return: {best_return:+.2%}")
    print(f"Model saved to: {args.model_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# BACKTEST sub-command
# ---------------------------------------------------------------------------

@dataclass
class BacktestStats:
    symbol: str
    start: str
    end: str
    initial_cash: float
    final_equity: float
    total_return_pct: float
    annualised_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    num_trades: int
    win_rate_pct: float
    buy_hold_return_pct: float


def _annualised(total_ret: float, n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    years = n_days / 252.0
    return (1 + total_ret) ** (1.0 / max(years, 1e-6)) - 1.0


def _sharpe(daily_rets: pd.Series, risk_free: float = 0.0) -> float:
    excess = daily_rets - risk_free / 252.0
    std = excess.std()
    if std < 1e-9:
        return 0.0
    return float(excess.mean() / std * math.sqrt(252))


def backtest_one(
    agent: DQNAgent,
    features: pd.DataFrame,
    closes: pd.Series,
    initial_cash: float,
    fee_rate: float,
    slippage_bps: float,
    buy_fraction: float,
    sell_fraction: float,
) -> Tuple[pd.DataFrame, BacktestStats]:
    """Run deterministic backtest (no exploration) and return ledger + stats."""
    env = TradingEnv(
        features,
        closes,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        buy_fraction=buy_fraction,
        sell_fraction=sell_fraction,
    )
    state = env.reset()
    rows = []
    prev_equity = initial_cash
    trades = 0
    winning_trades = 0

    agent.policy_net.eval()
    with torch.no_grad():
        while True:
            price = float(env.closes[env.t])
            action = agent.select_action(state)  # eps=0 during eval
            next_state, reward, done = env.step(action)

            equity = env.cash + env.shares * float(env.closes[min(env.t, env.n_steps - 1)])
            rows.append(
                {
                    "date": features.index[min(env.t, len(features) - 1)],
                    "close": price,
                    "action": ["HOLD", "BUY", "SELL"][action],
                    "cash": env.cash,
                    "shares": env.shares,
                    "equity": equity,
                    "daily_return": equity / (prev_equity + 1e-9) - 1.0,
                }
            )
            if action != HOLD:
                trades += 1
                if equity > prev_equity:
                    winning_trades += 1
            prev_equity = equity
            state = next_state
            if done:
                break

    agent.policy_net.train()
    agent.eps = agent.eps_end  # keep low after backtest

    ledger = pd.DataFrame(rows).set_index("date")

    if ledger.empty:
        raise RuntimeError("Backtest ledger is empty.")

    final_eq = float(ledger["equity"].iloc[-1])
    total_ret = final_eq / initial_cash - 1.0
    n_days = len(ledger)
    ann_ret = _annualised(total_ret, n_days)
    sharpe_v = _sharpe(ledger["daily_return"])
    peak = ledger["equity"].cummax()
    max_dd = float(((ledger["equity"] / peak) - 1.0).min())
    win_rate = winning_trades / max(1, trades)

    bh_ret = float(ledger["close"].iloc[-1]) / float(ledger["close"].iloc[0]) - 1.0

    symbol = "N/A"
    stats = BacktestStats(
        symbol=symbol,
        start=str(ledger.index.min()),
        end=str(ledger.index.max()),
        initial_cash=initial_cash,
        final_equity=final_eq,
        total_return_pct=total_ret * 100,
        annualised_return_pct=ann_ret * 100,
        sharpe=sharpe_v,
        max_drawdown_pct=max_dd * 100,
        num_trades=trades,
        win_rate_pct=win_rate * 100,
        buy_hold_return_pct=bh_ret * 100,
    )
    return ledger, stats


def cmd_backtest(args):
    if not _TORCH_OK:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)
    if not _YF_OK:
        print("ERROR: yfinance is required. Install with: pip install yfinance")
        sys.exit(1)
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found: {args.model_path}")
        print("Run 'python rl_agent_trainer.py train ...' first.")
        sys.exit(1)

    tickers: List[str] = [t.upper() for t in args.tickers]
    device = "cpu"

    print(f"\n{'='*60}")
    print(f"DQN Trading Agent – Backtest")
    print(f"{'='*60}")
    print(f"Tickers      : {', '.join(tickers)}")
    print(f"Period       : {args.start} → {args.end}")
    print(f"Initial cash : ${args.initial_cash:,.0f}")
    print(f"Model        : {args.model_path}")
    print()

    agent = DQNAgent.load(args.model_path, device=device)
    agent.eps = 0.0  # fully greedy during backtest

    all_stats: List[BacktestStats] = []
    all_ledgers: Dict[str, pd.DataFrame] = {}

    for sym in tickers:
        result = prepare_ticker_data(sym, args.start, args.end)
        if result is None:
            continue
        features, closes = result
        if len(features) < 20:
            print(f"  [SKIP] {sym}: too few bars for backtest")
            continue

        # Verify state_dim compatibility
        need_dim = features.shape[1] + 3
        if agent.policy_net.net[0].in_features != need_dim:
            print(
                f"  [SKIP] {sym}: state_dim mismatch "
                f"(model={agent.policy_net.net[0].in_features}, data={need_dim})"
            )
            continue

        try:
            ledger, stats = backtest_one(
                agent,
                features,
                closes,
                initial_cash=args.initial_cash,
                fee_rate=args.fee_rate,
                slippage_bps=args.slippage_bps,
                buy_fraction=args.buy_fraction,
                sell_fraction=args.sell_fraction,
            )
        except Exception as exc:
            print(f"  [ERROR] {sym}: {exc}")
            continue

        stats.symbol = sym
        all_stats.append(stats)
        all_ledgers[sym] = ledger

    if not all_stats:
        print("No backtest results – check data availability and model compatibility.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print(f"\n{'─'*80}")
    print(
        f"{'Ticker':<8} {'Return':>9} {'Ann.Ret':>9} {'Sharpe':>7} "
        f"{'MaxDD':>8} {'Trades':>7} {'WinRate':>8} {'B&H':>9}"
    )
    print(f"{'─'*80}")
    for s in sorted(all_stats, key=lambda x: -x.total_return_pct):
        print(
            f"{s.symbol:<8} {s.total_return_pct:>+8.1f}% {s.annualised_return_pct:>+8.1f}% "
            f"{s.sharpe:>7.2f} {s.max_drawdown_pct:>+7.1f}% "
            f"{s.num_trades:>7d} {s.win_rate_pct:>7.1f}% {s.buy_hold_return_pct:>+8.1f}%"
        )
    print(f"{'─'*80}")

    # Portfolio-level aggregate (equal-weight)
    avg_ret = np.mean([s.total_return_pct for s in all_stats])
    avg_ann = np.mean([s.annualised_return_pct for s in all_stats])
    avg_sharpe = np.mean([s.sharpe for s in all_stats])
    avg_dd = np.mean([s.max_drawdown_pct for s in all_stats])
    avg_bh = np.mean([s.buy_hold_return_pct for s in all_stats])
    print(
        f"{'AVERAGE':<8} {avg_ret:>+8.1f}% {avg_ann:>+8.1f}% "
        f"{avg_sharpe:>7.2f} {avg_dd:>+7.1f}% {'':>7} {'':>8} {avg_bh:>+8.1f}%"
    )
    print(f"{'─'*80}\n")

    # Optional CSV export
    if args.out_csv:
        rows = []
        for s in all_stats:
            rows.append(vars(s))
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"Stats written to: {args.out_csv}")

    if args.out_ledger_dir:
        Path(args.out_ledger_dir).mkdir(parents=True, exist_ok=True)
        for sym, ledger in all_ledgers.items():
            p = Path(args.out_ledger_dir) / f"{sym}_ledger.csv"
            ledger.to_csv(p)
        print(f"Ledgers written to: {args.out_ledger_dir}/")


# ---------------------------------------------------------------------------
# DIAGNOSE sub-command
# ---------------------------------------------------------------------------

def cmd_diagnose(args):
    print(f"\n{'='*60}")
    print(f"DQN Trading Agent – Diagnostic Check")
    print(f"{'='*60}\n")

    issues: List[str] = []
    warnings_list: List[str] = []

    # 1) Python version
    v = sys.version_info
    if v < (3, 9):
        issues.append(f"Python {v.major}.{v.minor} is below the minimum 3.9")
    else:
        print(f"  [OK] Python {v.major}.{v.minor}.{v.micro}")

    # 2) Package checks
    pkg_checks = [
        ("numpy", "numpy", "2.0"),
        ("pandas", "pandas", "2.0"),
        ("yfinance", "yfinance", "0.2"),
        ("torch", "torch", "2.0"),
    ]
    for name, mod, min_ver in pkg_checks:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  [OK] {name} {ver}")
        except ImportError:
            issues.append(f"Package '{name}' not installed. Run: pip install {name}")

    # 3) PyTorch CUDA
    if _TORCH_OK:
        cuda = torch.cuda.is_available()
        print(f"  [{'OK' if cuda else 'INFO'}] CUDA available: {cuda} (CPU training will be slower but works)")

    # 4) yfinance connectivity test
    print("\n  Testing data connectivity…")
    if _YF_OK:
        try:
            test_df = yf.download(
                "AAPL",
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
            if test_df is not None and not test_df.empty:
                print(f"  [OK] Yahoo Finance reachable (got {len(test_df)} rows for AAPL)")
            else:
                issues.append(
                    "Yahoo Finance returned empty data. Check internet connection."
                )
        except Exception as exc:
            issues.append(f"Yahoo Finance connectivity failed: {exc}")
    else:
        issues.append("yfinance not installed – cannot test connectivity.")

    # 5) Model file check
    model_path = getattr(args, "model_path", "model.pt")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1e6
        print(f"  [OK] Model file found: {model_path} ({size_mb:.2f} MB)")
        # Try loading
        if _TORCH_OK:
            try:
                ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
                sdim = ckpt.get("state_dim", "?")
                steps = ckpt.get("steps", "?")
                print(f"       state_dim={sdim}, trained_steps={steps}")
            except Exception as exc:
                issues.append(f"Model file is corrupt or incompatible: {exc}")
    else:
        warnings_list.append(
            f"Model file '{model_path}' not found – run 'train' to create it."
        )

    # 6) Cache directory
    if CACHE_DIR.exists():
        cached = list(CACHE_DIR.glob("*.pkl"))
        print(f"  [OK] Cache directory exists with {len(cached)} cached files")
    else:
        print(f"  [INFO] Cache directory '{CACHE_DIR}' does not exist yet (will be created on first download)")

    # 7) Existing backtest file
    bt_file = "papertrade_and_backtest_hourly_points.py"
    if os.path.exists(bt_file):
        print(f"  [OK] {bt_file} present")
    else:
        warnings_list.append(f"{bt_file} not found in current directory.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    if warnings_list:
        print("WARNINGS:")
        for w in warnings_list:
            print(f"  ⚠  {w}")
        print()

    if issues:
        print("BLOCKING ISSUES:")
        for iss in issues:
            print(f"  ✗  {iss}")
        print("\nFix the above issues before training.\n")
        sys.exit(1)
    else:
        print("All checks passed – ready to train and trade.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_data_args(p):
    p.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        metavar="TICKER",
        help="Space-separated list of ticker symbols",
    )
    p.add_argument("--start", default="2015-01-01", help="Data start date YYYY-MM-DD")
    p.add_argument("--end", default="2024-12-31", help="Data end date YYYY-MM-DD")
    p.add_argument("--fee-rate", type=float, default=0.001, help="e.g. 0.001 = 0.10%%")
    p.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage in basis points")


def main():
    parser = argparse.ArgumentParser(
        description="DQN-based RL trading agent – train and backtest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- diagnose ----
    p_diag = sub.add_parser("diagnose", help="Check for issues before training")
    p_diag.add_argument("--model-path", default="model.pt")
    p_diag.set_defaults(func=cmd_diagnose)

    # ---- train ----
    p_train = sub.add_parser("train", help="Train the DQN agent")
    _add_common_data_args(p_train)
    p_train.add_argument(
        "--train-hours",
        type=float,
        default=3.0,
        help="How many wall-clock hours to train (default: 3.0)",
    )
    p_train.add_argument("--model-path", default="model.pt", help="Where to save the model")
    p_train.add_argument(
        "--reset",
        action="store_true",
        help="Ignore existing checkpoint and train from scratch",
    )
    p_train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p_train.set_defaults(func=cmd_train)

    # ---- backtest ----
    p_bt = sub.add_parser("backtest", help="Backtest a saved model")
    _add_common_data_args(p_bt)
    p_bt.add_argument("--model-path", default="model.pt")
    p_bt.add_argument("--initial-cash", type=float, default=100_000.0)
    p_bt.add_argument("--buy-fraction", type=float, default=0.20, help="Fraction of cash to invest per BUY signal")
    p_bt.add_argument("--sell-fraction", type=float, default=0.50, help="Fraction of position to liquidate per SELL signal")
    p_bt.add_argument("--out-csv", default="", help="Optional path for per-ticker stats CSV")
    p_bt.add_argument("--out-ledger-dir", default="", help="Optional directory for per-ticker ledger CSVs")
    p_bt.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
