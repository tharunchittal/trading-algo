import math
import argparse
from dataclasses import dataclass
import pandas as pd
import yfinance as yf


DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL", "NFLX", "AVGO",
    "SMCI", "PLTR", "COIN", "SNOW", "ARM", "RIVN", "NIO", "MARA", "RIOT", "SOFI",
]


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


def points_down_last_24h(prev_close: float, signal_close: float) -> float:
    """
    Buy rule: buy $1 for every full point the stock fell in the last 24h.
    """
    delta = signal_close - prev_close
    if delta < 0:
        return float(math.floor(abs(delta)))
    return 0.0


def download_daily(symbol: str, start: str, end: str) -> pd.Series:
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError("No data returned. Try a longer date range.")
    df = df.sort_index()
    # Handle multi-column DataFrame from yfinance by selecting Close column
    if isinstance(df["Close"], pd.DataFrame):
        closes = df["Close"][symbol].astype(float)
    else:
        closes = df["Close"].astype(float)
    if len(closes) < 2:
        raise RuntimeError("Not enough daily bars returned for backtest.")
    return closes


def get_universe_symbols(universe: str) -> list[str]:
    if universe.strip().lower() == "default":
        return DEFAULT_UNIVERSE
    return [s.strip().upper() for s in universe.split(",") if s.strip()]


def annualized_volatility(symbol: str, year: int) -> float | None:
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    closes = download_daily(symbol, start, end)
    rets = closes.pct_change().dropna()
    if rets.empty:
        return None
    return float(rets.std() * math.sqrt(252.0))


def select_most_volatile_symbols(year: int, top_n: int, universe_symbols: list[str]) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for symbol in universe_symbols:
        try:
            vol = annualized_volatility(symbol, year)
            if vol is not None:
                ranked.append((symbol, vol))
        except Exception:
            continue

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def run_backtest(cfg: Config):
    closes = download_daily(cfg.symbol, cfg.start, cfg.end)

    cash = float(cfg.initial_cash)
    lots: list[dict] = []
    total_shares = 0.0

    rows = []
    last_i = len(closes) - 1

    for i in range(1, len(closes)):
        prev_close = float(closes.iloc[i - 1])
        signal_close = float(closes.iloc[i])
        signal_time = closes.index[i]

        buy_notional_target = points_down_last_24h(prev_close, signal_close)

        if cfg.execute_next_bar:
            if i + 1 > last_i:
                break
            fill_close_raw = float(closes.iloc[i + 1])
            fill_time = closes.index[i + 1]
        else:
            fill_close_raw = signal_close
            fill_time = signal_time

        slip = cfg.slippage_bps / 10_000.0

        buy_shares = 0.0
        buy_value = 0.0
        buy_fee = 0.0
        sell_shares = 0.0
        sell_value = 0.0
        sell_fee = 0.0
        skipped = False

        # SELL rule: for each lot, sell $1 per full point above (entry + 1)
        sell_price = fill_close_raw * (1.0 - slip)
        for lot in lots:
            gain_over_threshold = signal_close - (lot["entry_price"] + 1.0)
            if gain_over_threshold <= 0:
                continue

            sell_notional_target = min(
                float(math.floor(gain_over_threshold)),
                lot["remaining_notional"],
            )
            if sell_notional_target <= 0 or sell_price <= 0:
                continue

            shares_to_sell = min(lot["shares"], sell_notional_target / sell_price)
            if shares_to_sell <= 0:
                continue

            realized_value = shares_to_sell * sell_price
            lot["shares"] -= shares_to_sell
            lot["remaining_notional"] -= realized_value
            sell_shares += shares_to_sell
            sell_value += realized_value

        lots = [lot for lot in lots if lot["shares"] > 1e-12 and lot["remaining_notional"] > 1e-9]

        if sell_shares > 0:
            sell_fee = sell_value * cfg.fee_rate
            cash += sell_value - sell_fee
            total_shares -= sell_shares

        # BUY rule: buy $1 per full point down in last 24h
        if buy_notional_target > 0:
            buy_price = fill_close_raw * (1.0 + slip)
            buy_value = float(buy_notional_target)
            buy_fee = buy_value * cfg.fee_rate
            total_cost = buy_value + buy_fee

            if total_cost <= cash and buy_price > 0:
                buy_shares = buy_value / buy_price
                cash -= total_cost
                total_shares += buy_shares
                lots.append(
                    {
                        "entry_price": float(buy_price),
                        "shares": float(buy_shares),
                        "remaining_notional": float(buy_value),
                    }
                )
            else:
                skipped = True
                buy_shares = 0.0
                buy_value = 0.0
                buy_fee = 0.0

        equity = cash + total_shares * signal_close

        rows.append({
            "signal_time": signal_time,
            "fill_time": fill_time,
            "prev_close": float(prev_close),
            "signal_close": signal_close,
            "points_down_24h": float(buy_notional_target),
            "fill_close_raw": float(fill_close_raw),
            "buy_shares": float(buy_shares),
            "buy_value_usd": float(buy_value),
            "buy_fee_usd": float(buy_fee),
            "sell_shares": float(sell_shares),
            "sell_value_usd": float(sell_value),
            "sell_fee_usd": float(sell_fee),
            "skipped": bool(skipped),
            "cash_usd": float(cash),
            "total_shares": float(total_shares),
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
    num_buys = int((ledger["buy_shares"] > 0).sum())
    num_sells = int((ledger["sell_shares"] > 0).sum())
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
        "num_buy_trades": num_buys,
        "num_sell_trades": num_sells,
        "num_skipped_orders": num_skipped,
        "fee_rate": float(cfg.fee_rate),
        "slippage_bps": float(cfg.slippage_bps),
        "allow_short": bool(cfg.allow_short),
        "execute_next_bar": bool(cfg.execute_next_bar),
    }

    return ledger, stats


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Backtest strategy: buy $1 per 1-point drop over the past 24h, and sell $1 per 1-point gain "
            "above (entry + 1). Can auto-select the most volatile stocks for a given year."
        )
    )
    ap.add_argument("--symbol", default="", help="Single ticker, e.g. AAPL")
    ap.add_argument("--start", default="", help="YYYY-MM-DD")
    ap.add_argument("--end", default="", help="YYYY-MM-DD")
    ap.add_argument("--year", type=int, default=0, help="If set, select most volatile symbols in this year and trade that year")
    ap.add_argument("--top-n", type=int, default=5, help="How many most-volatile symbols to trade when --year is set")
    ap.add_argument("--universe", default="default", help="'default' or comma-separated tickers used for volatility ranking")
    ap.add_argument("--initial-cash", type=float, default=10_000.0)
    ap.add_argument("--fee-rate", type=float, default=0.0, help="e.g. 0.001 = 0.10%")
    ap.add_argument("--slippage-bps", type=float, default=0.0, help="e.g. 5 = 5 bps")
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--execute-next-bar", action="store_true", help="Fill on next bar close (less lookahead bias)")
    ap.add_argument("--out-csv", default="", help="Optional path to write ledger CSV; in multi-symbol mode files are suffixed by symbol")
    args = ap.parse_args()

    if args.year:
        start = f"{args.year}-01-01"
        end = f"{args.year + 1}-01-01"
    else:
        start = args.start.strip()
        end = args.end.strip()

    if not start or not end:
        raise RuntimeError("Provide --start and --end, or provide --year.")

    symbols_to_trade: list[str]
    if args.year:
        universe_symbols = get_universe_symbols(args.universe)
        ranked = select_most_volatile_symbols(args.year, args.top_n, universe_symbols)
        if not ranked:
            raise RuntimeError("Could not rank volatility for the provided universe/year.")
        symbols_to_trade = [symbol for symbol, _ in ranked]
        print("\nMOST VOLATILE SYMBOLS")
        for symbol, vol in ranked:
            print(f"{symbol}: annualized_vol={vol:.4f}")
    else:
        symbol = args.symbol.upper().strip()
        if not symbol:
            raise RuntimeError("Provide --symbol for single-symbol mode, or --year for volatility-selection mode.")
        symbols_to_trade = [symbol]

    all_stats = []
    for symbol in symbols_to_trade:
        cfg = Config(
            symbol=symbol,
            start=start,
            end=end,
            initial_cash=args.initial_cash,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
            allow_short=args.allow_short,
            execute_next_bar=args.execute_next_bar,
            out_csv=(args.out_csv.strip() or None),
        )

        ledger, stats = run_backtest(cfg)
        all_stats.append(stats)

        print("\nRESULTS")
        for k, v in stats.items():
            print(f"{k}: {v}")

        if cfg.out_csv:
            if len(symbols_to_trade) == 1:
                out_path = cfg.out_csv
            else:
                if cfg.out_csv.endswith(".csv"):
                    out_path = cfg.out_csv[:-4] + f"_{symbol}.csv"
                else:
                    out_path = cfg.out_csv + f"_{symbol}.csv"
            ledger.to_csv(out_path)
            print(f"\nWrote ledger CSV to: {out_path}")

    if len(all_stats) > 1:
        summary = pd.DataFrame(all_stats)
        summary = summary.sort_values("return_pct", ascending=False)
        print("\nSUMMARY (sorted by return_pct)")
        for _, r in summary.iterrows():
            print(
                f"{r['symbol']}: return_pct={r['return_pct']:.4f}, pnl_usd={r['pnl_usd']:.2f}, "
                f"max_drawdown_pct={r['max_drawdown_pct']:.4f}"
            )


if __name__ == "__main__":
    main()