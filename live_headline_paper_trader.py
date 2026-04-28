import argparse
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

import pandas as pd
import yfinance as yf

from google_trends_trading_bot import (
    cache_json_path,
    ensure_cache_dir,
    http_get_json,
    http_get_text,
    parse_pub_dt,
    parse_rss_items,
    read_cached_json,
    sentiment_score,
    write_cached_json,
)


@dataclass
class LiveConfig:
    symbol: str
    initial_cash: float
    order_usd: float
    entry_sentiment: float
    exit_sentiment: float
    event_window: int
    min_headlines: int
    poll_seconds: int
    cooldown_minutes: int
    fee_rate: float
    slippage_bps: float
    max_position_usd: float
    finnhub_token: str
    use_finnhub: bool
    rss_feeds: list[str]
    cache_dir: str
    refresh_cache: bool
    state_file: str
    out_csv: str | None
    max_cycles: int


def parse_ts(raw: str | None) -> pd.Timestamp | None:
    if not raw:
        return None
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.tz_localize(None)


def stable_uid(parts: list[str]) -> str:
    text = "|".join(parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_state(cfg: LiveConfig) -> dict:
    path = Path(cfg.state_file)
    cached = read_cached_json(path)
    if isinstance(cached, dict):
        return cached
    return {
        "cash": float(cfg.initial_cash),
        "shares": 0.0,
        "last_seen_ts": None,
        "last_trade_ts": None,
        "seen_ids": [],
        "sentiment_buffer": [],
    }


def save_state(cfg: LiveConfig, state: dict) -> None:
    path = Path(cfg.state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_cached_json(path, state)


def append_row_csv(path: str | None, row: dict) -> None:
    if not path:
        return
    df = pd.DataFrame([row])
    out_path = Path(path)
    write_header = not out_path.exists()
    df.to_csv(out_path, mode="a", header=write_header, index=False)


def fetch_finnhub_live(cfg: LiveConfig, now_ts: pd.Timestamp) -> list[dict]:
    if not cfg.use_finnhub or not cfg.finnhub_token:
        return []

    start = (now_ts - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    end = (now_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    key = f"{cfg.symbol}|{start}|{end}|{cfg.finnhub_token[:6]}"
    cpath = cache_json_path(cfg.cache_dir, "live_finnhub", key)

    ttl_seconds = max(30, min(cfg.poll_seconds, 300))
    if not cfg.refresh_cache:
        cached = read_cached_json(cpath)
        if isinstance(cached, dict):
            fetched_at = parse_ts(cached.get("fetched_at"))
            items = cached.get("items")
            if fetched_at is not None and isinstance(items, list):
                age = (now_ts - fetched_at).total_seconds()
                if age <= ttl_seconds:
                    return items

    params = urlencode({
        "symbol": cfg.symbol,
        "from": start,
        "to": end,
        "token": cfg.finnhub_token,
    })
    url = f"https://finnhub.io/api/v1/company-news?{params}"

    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            payload = http_get_json(url)
            if isinstance(payload, list):
                write_cached_json(cpath, {"fetched_at": now_ts.isoformat(), "items": payload})
                return payload
            return []
        except HTTPError as exc:
            if exc.code == 429 and attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
                continue
            return []
        except (URLError, TimeoutError, ValueError):
            return []

    return []


def fetch_rss_live(cfg: LiveConfig, now_ts: pd.Timestamp) -> list[dict]:
    default_feed = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={cfg.symbol}&region=US&lang=en-US"
    feeds = cfg.rss_feeds or [default_feed]
    combined: list[dict] = []

    ttl_seconds = max(30, min(cfg.poll_seconds, 300))
    for feed_url in feeds:
        key = f"{cfg.symbol}|{feed_url}"
        cpath = cache_json_path(cfg.cache_dir, "live_rss", key)

        payload: list[dict] = []
        if not cfg.refresh_cache:
            cached = read_cached_json(cpath)
            if isinstance(cached, dict):
                fetched_at = parse_ts(cached.get("fetched_at"))
                items = cached.get("items")
                if fetched_at is not None and isinstance(items, list):
                    age = (now_ts - fetched_at).total_seconds()
                    if age <= ttl_seconds:
                        payload = items

        if not payload:
            try:
                xml_text = http_get_text(feed_url)
                payload = parse_rss_items(xml_text)
                write_cached_json(cpath, {"fetched_at": now_ts.isoformat(), "items": payload})
            except (HTTPError, URLError, TimeoutError):
                payload = []

        combined.extend(payload)

    return combined


def normalize_new_items(
    cfg: LiveConfig,
    state: dict,
    now_ts: pd.Timestamp,
    finnhub_items: list[dict],
    rss_items: list[dict],
) -> list[dict]:
    seen_ids = set(str(x) for x in state.get("seen_ids", []))
    last_seen_ts = parse_ts(state.get("last_seen_ts"))
    min_ts = now_ts - pd.Timedelta(hours=48)

    normalized: list[dict] = []

    for item in finnhub_items:
        dt = pd.to_datetime(item.get("datetime", 0), unit="s", utc=True, errors="coerce")
        if pd.isna(dt):
            continue
        dt = dt.tz_localize(None)
        if dt < min_ts:
            continue
        if last_seen_ts is not None and dt <= last_seen_ts:
            continue

        title = str(item.get("headline") or "").strip()
        summary = str(item.get("summary") or "").strip()
        url = str(item.get("url") or "")
        uid = "fh|" + stable_uid([dt.isoformat(), title, url])
        if uid in seen_ids:
            continue

        normalized.append(
            {
                "id": uid,
                "published": dt,
                "source": "finnhub",
                "title": title,
                "summary": summary,
                "sentiment": sentiment_score(f"{title} {summary}"),
            }
        )

    for item in rss_items:
        dt = parse_pub_dt(item.get("published"))
        if dt is None:
            continue
        if dt < min_ts:
            continue
        if last_seen_ts is not None and dt <= last_seen_ts:
            continue

        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or "").strip()
        uid = "rss|" + stable_uid([dt.isoformat(), title])
        if uid in seen_ids:
            continue

        normalized.append(
            {
                "id": uid,
                "published": dt,
                "source": "rss",
                "title": title,
                "summary": summary,
                "sentiment": sentiment_score(f"{title} {summary}"),
            }
        )

    normalized.sort(key=lambda x: x["published"])
    return normalized


def fetch_live_price(symbol: str) -> float:
    df = yf.download(
        symbol,
        period="1d",
        interval="1m",
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        df = yf.download(
            symbol,
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    if df is None or df.empty:
        raise RuntimeError("Unable to fetch live price for symbol")

    close_col = df["Close"]
    if isinstance(close_col, pd.DataFrame):
        last_close = float(close_col[symbol].dropna().iloc[-1])
    else:
        last_close = float(close_col.dropna().iloc[-1])
    return last_close


def run_live_paper(cfg: LiveConfig) -> None:
    ensure_cache_dir(cfg.cache_dir)
    state = load_state(cfg)

    cash = float(state.get("cash", cfg.initial_cash))
    shares = float(state.get("shares", 0.0))
    seen_ids = [str(x) for x in state.get("seen_ids", [])]
    sentiment_buffer = [float(x) for x in state.get("sentiment_buffer", [])]
    last_trade_ts = parse_ts(state.get("last_trade_ts"))
    last_seen_ts = parse_ts(state.get("last_seen_ts"))

    cycle = 0
    while True:
        now_ts = pd.Timestamp.now(tz="UTC").tz_localize(None)

        finnhub_items = fetch_finnhub_live(cfg, now_ts)
        rss_items = fetch_rss_live(cfg, now_ts)
        new_items = normalize_new_items(cfg, state, now_ts, finnhub_items, rss_items)

        if new_items:
            for item in new_items:
                sentiment_buffer.append(float(item["sentiment"]))
                seen_ids.append(item["id"])
            last_seen_ts = max(item["published"] for item in new_items)

        if len(sentiment_buffer) > 500:
            sentiment_buffer = sentiment_buffer[-500:]
        if len(seen_ids) > 5000:
            seen_ids = seen_ids[-5000:]

        lookback_scores = sentiment_buffer[-cfg.event_window:]
        agg_sentiment = float(sum(lookback_scores) / len(lookback_scores)) if lookback_scores else 0.0
        headline_count = len(new_items)

        price = fetch_live_price(cfg.symbol)
        slip = cfg.slippage_bps / 10_000.0
        action = "HOLD"
        trade_shares = 0.0
        trade_value = 0.0
        fee_usd = 0.0

        cooldown_ok = (
            last_trade_ts is None
            or (now_ts - last_trade_ts).total_seconds() >= cfg.cooldown_minutes * 60
        )

        entry_signal = agg_sentiment >= cfg.entry_sentiment and headline_count >= cfg.min_headlines
        exit_signal = agg_sentiment <= cfg.exit_sentiment and headline_count >= cfg.min_headlines

        position_usd = shares * price

        if shares <= 0 and entry_signal and cooldown_ok:
            notional_cap = max(0.0, cfg.max_position_usd - position_usd)
            notional = min(cfg.order_usd, cash, notional_cap)
            if notional > 0 and price > 0:
                buy_price = price * (1.0 + slip)
                trade_shares = notional / buy_price
                fee_usd = notional * cfg.fee_rate
                total_cost = notional + fee_usd
                if total_cost <= cash:
                    cash -= total_cost
                    shares += trade_shares
                    trade_value = notional
                    action = "BUY"
                    last_trade_ts = now_ts

        elif shares > 0 and exit_signal and cooldown_ok:
            sell_price = price * (1.0 - slip)
            gross = shares * sell_price
            fee_usd = gross * cfg.fee_rate
            cash += gross - fee_usd
            trade_shares = -shares
            trade_value = gross
            shares = 0.0
            action = "SELL"
            last_trade_ts = now_ts

        equity = cash + shares * price
        row = {
            "timestamp": now_ts.isoformat(),
            "symbol": cfg.symbol,
            "price": float(price),
            "new_headlines": int(headline_count),
            "agg_sentiment": float(agg_sentiment),
            "entry_signal": bool(entry_signal),
            "exit_signal": bool(exit_signal),
            "action": action,
            "trade_shares": float(trade_shares),
            "trade_value_usd": float(trade_value),
            "fee_usd": float(fee_usd),
            "cash_usd": float(cash),
            "shares": float(shares),
            "equity_usd": float(equity),
        }

        print(
            f"{row['timestamp']} action={action} price={price:.2f} "
            f"sent={agg_sentiment:.3f} headlines={headline_count} equity={equity:.2f}"
        )

        append_row_csv(cfg.out_csv, row)

        state = {
            "cash": float(cash),
            "shares": float(shares),
            "last_seen_ts": last_seen_ts.isoformat() if last_seen_ts is not None else None,
            "last_trade_ts": last_trade_ts.isoformat() if last_trade_ts is not None else None,
            "seen_ids": seen_ids,
            "sentiment_buffer": sentiment_buffer,
        }
        save_state(cfg, state)

        cycle += 1
        if cfg.max_cycles > 0 and cycle >= cfg.max_cycles:
            break

        time.sleep(cfg.poll_seconds)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Stage-1 live paper trader: event-driven headline sentiment with Finnhub + RSS, "
            "15-minute default polling, dedupe, cooldowns, fees, slippage, and persistent state/cache."
        )
    )
    ap.add_argument("--symbol", required=True, help="Stock ticker, e.g. AAPL")
    ap.add_argument("--initial-cash", type=float, default=10000.0)
    ap.add_argument("--order-usd", type=float, default=1000.0)
    ap.add_argument("--entry-sentiment", type=float, default=0.20)
    ap.add_argument("--exit-sentiment", type=float, default=-0.20)
    ap.add_argument("--event-window", type=int, default=20, help="Rolling number of headline events")
    ap.add_argument("--min-headlines", type=int, default=1, help="Minimum new headlines this cycle to trade")
    ap.add_argument("--poll-seconds", type=int, default=900, help="Polling cadence in seconds (900 = 15 min)")
    ap.add_argument("--cooldown-minutes", type=int, default=30, help="Minimum minutes between trades")
    ap.add_argument("--fee-rate", type=float, default=0.0005, help="e.g. 0.0005 = 5 bps")
    ap.add_argument("--slippage-bps", type=float, default=5.0, help="e.g. 5 = 5 bps")
    ap.add_argument("--max-position-usd", type=float, default=3000.0)
    ap.add_argument("--finnhub-token", default=os.environ.get("FINNHUB_API_KEY", ""))
    ap.add_argument("--disable-finnhub", action="store_true")
    ap.add_argument("--rss-feed", action="append", default=[], help="Optional RSS feed URL. Can repeat")
    ap.add_argument("--cache-dir", default=".cache/headline_bot")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--state-file", default="", help="Optional JSON file for live state persistence")
    ap.add_argument("--out-csv", default="", help="Optional CSV ledger path")
    ap.add_argument("--max-cycles", type=int, default=0, help="0 = run forever; set >0 for testing")
    args = ap.parse_args()

    cache_dir = args.cache_dir.strip() or ".cache/headline_bot"
    ensure_cache_dir(cache_dir)

    state_file = args.state_file.strip()
    if not state_file:
        state_file = str(Path(cache_dir) / f"live_state_{args.symbol.upper().strip()}.json")

    use_finnhub = (not args.disable_finnhub) and bool(args.finnhub_token.strip())

    cfg = LiveConfig(
        symbol=args.symbol.upper().strip(),
        initial_cash=float(args.initial_cash),
        order_usd=float(args.order_usd),
        entry_sentiment=float(args.entry_sentiment),
        exit_sentiment=float(args.exit_sentiment),
        event_window=int(args.event_window),
        min_headlines=int(args.min_headlines),
        poll_seconds=max(30, int(args.poll_seconds)),
        cooldown_minutes=max(0, int(args.cooldown_minutes)),
        fee_rate=max(0.0, float(args.fee_rate)),
        slippage_bps=max(0.0, float(args.slippage_bps)),
        max_position_usd=max(0.0, float(args.max_position_usd)),
        finnhub_token=args.finnhub_token.strip(),
        use_finnhub=use_finnhub,
        rss_feeds=[str(x).strip() for x in args.rss_feed if str(x).strip()],
        cache_dir=cache_dir,
        refresh_cache=bool(args.refresh_cache),
        state_file=state_file,
        out_csv=(args.out_csv.strip() or None),
        max_cycles=max(0, int(args.max_cycles)),
    )

    run_live_paper(cfg)


if __name__ == "__main__":
    main()
