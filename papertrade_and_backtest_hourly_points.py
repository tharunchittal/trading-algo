import math
import argparse
from dataclasses import dataclass
import pandas as pd
import yfinance as yf


@dataclass
class Config:
    symbol: str
    start: str
    end: str
    initial_cash: float
    fee_rate: float          # e.g. 0.001 = 0.10% per trade value
    slippage_bps: float      # e.g. 5 = 5 basis points
    allow_short: bool
    execute_next_bar: bool   # if True: fill at next bar close to reduce lookahead bias
    out_csv: str | None


def order_usd_from_delta_points(delta_points: float) -> float:
    """
    Rule:
      down => buy $1 per 1 point down
      up   => sell $1 per 1 point up
    Use floor so only full points trigger.
    Returns: +N buy $N, -N sell $N
    """
    if delta_points < 0:
        return float(math.floor(abs(delta_points)))   # BUY $N
    if delta_points > 0:
        return float(-math.floor(delta_points))       # SELL $N
    return 0.0


def download_hourly(symbol: str, start: str, end: str) -> pd.Series:
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1h",
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError("No data returned. Try a shorter date range (Yahoo 1h history is limited).")
    df = df.sort_index()
    closes = df["Close"].astype(float)
    if len(closes) < 4:
        raise RuntimeError("Not enough hourly bars returned for backtest.")
    return closes


def run_backtest(cfg: Config):
    closes = download_hourly(cfg.symbol, cfg.start, cfg.end)

    cash = float(cfg.initial_cash)
    shares = 0.0

    rows = []

    last_i = len(closes) - 1

    # We create signals from (i-1 -> i). If next-bar execution, we fill at i+1.
    for i in range(1, len(closes)):
        prev_close = float(closes.iloc[i - 1])
        signal_close = float(closes.iloc[i])
        signal_time = closes.index[i]

        delta_points = signal_close - prev_close
        order_usd = order_usd_from_delta_points(delta_points)  # +buy, -sell

        # Fill timing/price
        if cfg.execute_next_bar:
            if i + 1 > last_i:
                break
            fill_close_raw = float(closes.iloc[i + 1])
            fill_time = closes.index[i + 1]
        else:
            fill_close_raw = signal_close
            fill_time = signal_time

        # Slippage model
        slip = cfg.slippage_bps / 10_000.0
        if order_usd > 0:
            fill_price = fill_close_raw * (1.0 + slip)
        elif order_usd < 0:
            fill_price = fill_close_raw * (1.0 - slip)
        else:
            fill_price = fill_close_raw

        trade_shares = 0.0
        trade_value = 0.0
        fee = 0.0
        skipped = False

        if order_usd > 0:
            # BUY $order_usd notional
            trade_value = float(order_usd)
            fee = trade_value * cfg.fee_rate
            total_cost = trade_value + fee

            if total_cost <= cash and fill_price > 0:
                trade_shares = trade_value / fill_price
                cash -= total_cost
                shares += trade_shares
            else:
                skipped = True
                trade_value = 0.0
                fee = 0.0
                trade_shares = 0.0

        elif order_usd < 0:
            # SELL $abs(order_usd) notional
            desired_value = float(abs(order_usd))
            desired_shares = (desired_value / fill_price) if fill_price > 0 else 0.0

            if not cfg.allow_short:
                desired_shares = min(desired_shares, shares)

            if desired_shares > 0:
                trade_value = desired_shares * fill_price
                fee = trade_value * cfg.fee_rate
                cash += (trade_value - fee)
                shares -= desired_shares
                trade_shares = -desired_shares
            else:
                skipped = True
                trade_value = 0.0
                fee = 0.0
                trade_shares = 0.0

        # Mark-to-market equity at the signal close (consistent equity curve)
        equity = cash + shares * signal_close

        rows.append({
            "signal_time": signal_time,
            "fill_time": fill_time,
            "prev_close": prev_close,
            "signal_close": signal_close,
            "delta_points": float(delta_points),
            "order_usd": float(order_usd),
            "fill_price": float(fill_price),
            "trade_value_usd": float(trade_value),
            "fee_usd": float(fee),
            "trade_shares": float(trade_shares),
            "skipped": bool(skipped),
            "cash_usd": float(cash),
            "shares": float(shares),
            "equity_usd": float(equity),
        })

    ledger = pd.DataFrame(rows).set_index("signal_time")
    if ledger.empty:
        raise RuntimeError("Ledger is empty—date range may be too short or data unavailable.")

    # Metrics
    ending_equity = float(ledger["equity_usd"].iloc[-1])
    pnl = ending_equity - cfg.initial_cash
    ret = ending_equity / cfg.initial_cash - 1.0
    peak = ledger["equity_usd"].cummax()
    drawdown = (ledger["equity_usd"] / peak) - 1.0
    max_dd = float(drawdown.min())
    num_trades = int((ledger["trade_shares"].abs() > 0).sum())
    num_skipped = int(ledger["skipped"].sum())

    stats = {
        "symbol": cfg.symbol,
        "start": str(ledger.index.min()),
        "end": str(ledger.index.max()),
        "initial_cash_usd": float(cfg.initial_cash),
        "ending_equity_usd": ending_equity,
        "pnl_usd": float(pnl),
        "return_pct": float(ret * 100.0),
        "max_drawdown_pct": float(max_dd * 100.0),
        "num_trades": num_trades,
        "num_skipped_orders": num_skipped,
        "fee_rate": float(cfg.fee_rate),
        "slippage_bps": float(cfg.slippage_bps),
        "allow_short": bool(cfg.allow_short),
        "execute_next_bar": bool(cfg.execute_next_bar),
    }

    return ledger, stats


def main():
    ap = argparse.ArgumentParser(
        description="Paper-trade + backtest: $1 per point hourly move using Yahoo Finance 1h data."
    )
    ap.add_argument("--symbol", required=True, help="US ticker, e.g. AAPL, MSFT, TSLA")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--initial-cash", type=float, default=10_000.0)
    ap.add_argument("--fee-rate", type=float, default=0.0, help="e.g. 0.001 = 0.10%")
    ap.add_argument("--slippage-bps", type=float, default=0.0, help="e.g. 5 = 5 bps")
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--execute-next-bar", action="store_true", help="Fill on next bar close (less lookahead bias)")
    ap.add_argument("--out-csv", default="", help="Optional path to write ledger CSV (e.g. ledger.csv)")
    args = ap.parse_args()

    cfg = Config(
        symbol=args.symbol.upper().strip(),
        start=args.start,
        end=args.end,
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        allow_short=args.allow_short,
        execute_next_bar=args.execute_next_bar,
        out_csv=(args.out_csv.strip() or None),
    )

    ledger, stats = run_backtest(cfg)

    print("\nRESULTS")
    for k, v in stats.items():
        print(f"{k}: {v}")

    if cfg.out_csv:
        ledger.to_csv(cfg.out_csv)
        print(f"\nWrote ledger CSV to: {cfg.out_csv}")


if __name__ == "__main__":
    main()