import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd
import yfinance as yf


@dataclass
class Config:
    symbol: str
    start: str
    end: str
    initial_cash: float
    order_usd: float
    entry_sentiment: float
    exit_sentiment: float
    sentiment_window: int
    min_headlines: int
    finnhub_token: str
    use_finnhub: bool
    rss_feeds: list[str]
    cache_dir: str
    refresh_cache: bool
    out_csv: str | None


def ensure_cache_dir(cache_dir: str) -> Path:
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_path(cache_dir: str, prefix: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return ensure_cache_dir(cache_dir) / f"{prefix}_{digest}.csv"


def cache_json_path(cache_dir: str, prefix: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return ensure_cache_dir(cache_dir) / f"{prefix}_{digest}.json"


def read_cached_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_cached_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def read_cached_series(path: Path) -> pd.Series | None:
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "date" not in df.columns or "value" not in df.columns:
        return None

    date_raw = df["date"].astype(str)
    if date_raw.str.contains("/").any():
        # Support cached weekly period labels such as "2020-01-06/2020-01-12".
        period_idx = pd.PeriodIndex(date_raw, freq="W")
        idx = period_idx.end_time.tz_localize(None)
    else:
        idx = pd.to_datetime(date_raw, utc=True).dt.tz_localize(None)

    s = pd.Series(df["value"].astype(float).values, index=idx, dtype=float)
    return s.sort_index()


def write_cached_series(path: Path, series: pd.Series) -> None:
    out = pd.DataFrame({"date": pd.to_datetime(series.index), "value": series.values})
    out.to_csv(path, index=False)


def download_daily_closes(symbol: str, start: str, end: str, cache_dir: str, refresh_cache: bool) -> pd.Series:
    key = f"{symbol}|{start}|{end}|1d"
    cpath = cache_path(cache_dir, "prices", key)
    if not refresh_cache:
        cached = read_cached_series(cpath)
        if cached is not None and not cached.empty:
            return cached

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError("No daily price data returned.")

    close_col = df["Close"]
    if isinstance(close_col, pd.DataFrame):
        closes = close_col[symbol].astype(float)
    else:
        closes = close_col.astype(float)

    if closes.empty:
        raise RuntimeError("No close prices found in downloaded data.")

    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    closes = closes.sort_index()
    write_cached_series(cpath, closes)
    return closes


POSITIVE_WORDS = {
    "beat", "beats", "bull", "bullish", "buy", "upgrade", "upgrades", "surge",
    "growth", "record", "strong", "outperform", "raise", "raised", "gain",
    "gains", "rally", "wins", "win", "profit", "profits", "expands", "expansion",
}

NEGATIVE_WORDS = {
    "miss", "misses", "bear", "bearish", "sell", "downgrade", "downgrades", "drop",
    "falls", "fall", "decline", "weak", "lawsuit", "probe", "cuts", "cut",
    "warning", "warns", "loss", "losses", "plunge", "plunges", "fraud", "default",
}


def sentiment_score(text: str) -> float:
    words = [w.strip(".,!?;:\"'()[]{}<>").lower() for w in text.split()]
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return float((pos - neg) / total)


def parse_pub_dt(raw: str | None) -> pd.Timestamp | None:
    if not raw:
        return None
    parsed = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.tz_localize(None)


def http_get_json(url: str) -> Any:
    req = Request(url, headers={"User-Agent": "trading-algo/1.0"})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "trading-algo/1.0"})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def download_finnhub_news(
    symbol: str,
    start: str,
    end: str,
    token: str,
    cache_dir: str,
    refresh_cache: bool,
) -> list[dict]:
    key = f"finnhub|{symbol}|{start}|{end}|{token[:6]}"
    cpath = cache_json_path(cache_dir, "finnhub_news", key)
    if not refresh_cache:
        cached = read_cached_json(cpath)
        if isinstance(cached, list):
            return cached

    if not token:
        return []

    params = urlencode({
        "symbol": symbol,
        "from": start,
        "to": end,
        "token": token,
    })
    url = f"https://finnhub.io/api/v1/company-news?{params}"

    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            payload = http_get_json(url)
            if isinstance(payload, list):
                write_cached_json(cpath, payload)
                return payload
            return []
        except HTTPError as exc:
            if exc.code == 429 and attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
                continue
            return []
        except (URLError, TimeoutError, json.JSONDecodeError):
            return []

    return []


def parse_rss_items(xml_text: str) -> list[dict]:
    items: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    # RSS 2.0
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        desc = (item.findtext("description") or "").strip()
        pub = item.findtext("pubDate")
        items.append({"title": title, "summary": desc, "published": pub})

    # Atom
    atom_entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
    for entry in atom_entries:
        title = (entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
        summary = (entry.findtext("{http://www.w3.org/2005/Atom}summary") or "").strip()
        pub = entry.findtext("{http://www.w3.org/2005/Atom}updated")
        items.append({"title": title, "summary": summary, "published": pub})

    return items


def download_rss_news(
    symbol: str,
    start: str,
    end: str,
    rss_feeds: list[str],
    cache_dir: str,
    refresh_cache: bool,
) -> list[dict]:
    default_feed = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    feeds = rss_feeds or [default_feed]

    all_items: list[dict] = []
    for feed_url in feeds:
        key = f"rss|{feed_url}|{symbol}"
        cpath = cache_json_path(cache_dir, "rss_news", key)

        payload: list[dict] = []
        if not refresh_cache:
            cached = read_cached_json(cpath)
            if isinstance(cached, list):
                payload = cached

        if not payload:
            try:
                xml_text = http_get_text(feed_url)
                payload = parse_rss_items(xml_text)
                write_cached_json(cpath, payload)
            except (HTTPError, URLError, TimeoutError):
                payload = []

        all_items.extend(payload)

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    filtered: list[dict] = []
    for item in all_items:
        dt = parse_pub_dt(item.get("published"))
        if dt is None:
            continue
        if start_ts <= dt.normalize() < end_ts:
            filtered.append(item)
    return filtered


def build_daily_headline_frame(
    symbol: str,
    start: str,
    end: str,
    finnhub_token: str,
    use_finnhub: bool,
    rss_feeds: list[str],
    cache_dir: str,
    refresh_cache: bool,
) -> pd.DataFrame:
    records: list[dict] = []

    if use_finnhub:
        finnhub_items = download_finnhub_news(symbol, start, end, finnhub_token, cache_dir, refresh_cache)
        for item in finnhub_items:
            dt = pd.to_datetime(item.get("datetime", 0), unit="s", utc=True, errors="coerce")
            if pd.isna(dt):
                continue
            dt = dt.tz_localize(None)
            title = str(item.get("headline") or "")
            summary = str(item.get("summary") or "")
            score = sentiment_score(f"{title} {summary}")
            records.append({"date": dt.normalize(), "sentiment": score})

    rss_items = download_rss_news(symbol, start, end, rss_feeds, cache_dir, refresh_cache)
    for item in rss_items:
        dt = parse_pub_dt(item.get("published"))
        if dt is None:
            continue
        title = str(item.get("title") or "")
        summary = str(item.get("summary") or "")
        score = sentiment_score(f"{title} {summary}")
        records.append({"date": dt.normalize(), "sentiment": score})

    if not records:
        return pd.DataFrame(columns=["sentiment", "headline_count"])

    df = pd.DataFrame(records)
    out = df.groupby("date")["sentiment"].agg(["mean", "count"]).rename(
        columns={"mean": "sentiment", "count": "headline_count"}
    )
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def build_signal_frame(
    closes: pd.Series,
    daily_headlines: pd.DataFrame,
    sentiment_window: int,
    entry_sentiment: float,
    exit_sentiment: float,
    min_headlines: int,
) -> pd.DataFrame:
    price_day = closes.to_frame("close")
    price_day.index = pd.to_datetime(price_day.index).tz_localize(None)

    daily = daily_headlines.copy()
    if daily.empty:
        daily = pd.DataFrame(index=price_day.index, data={"sentiment": 0.0, "headline_count": 0})

    df = price_day.join(daily, how="left")
    df["sentiment"] = df["sentiment"].fillna(0.0)
    df["headline_count"] = df["headline_count"].fillna(0).astype(int)

    if len(df) < sentiment_window + 3:
        raise RuntimeError("Not enough aligned daily price + headline data for backtest.")

    df["sentiment_sma"] = df["sentiment"].rolling(sentiment_window).mean()
    df["entry_signal"] = (df["sentiment_sma"] >= entry_sentiment) & (df["headline_count"] >= min_headlines)
    df["exit_signal"] = df["sentiment_sma"] <= exit_sentiment
    return df


def run_backtest(cfg: Config) -> tuple[pd.DataFrame, dict]:
    closes = download_daily_closes(cfg.symbol, cfg.start, cfg.end, cfg.cache_dir, cfg.refresh_cache)
    headlines = build_daily_headline_frame(
        symbol=cfg.symbol,
        start=cfg.start,
        end=cfg.end,
        finnhub_token=cfg.finnhub_token,
        use_finnhub=cfg.use_finnhub,
        rss_feeds=cfg.rss_feeds,
        cache_dir=cfg.cache_dir,
        refresh_cache=cfg.refresh_cache,
    )
    df = build_signal_frame(
        closes=closes,
        daily_headlines=headlines,
        sentiment_window=cfg.sentiment_window,
        entry_sentiment=cfg.entry_sentiment,
        exit_sentiment=cfg.exit_sentiment,
        min_headlines=cfg.min_headlines,
    )

    cash = float(cfg.initial_cash)
    shares = 0.0
    in_position = False

    rows: list[dict] = []
    trade_count = 0

    # Execute signals at the next day's close to avoid lookahead bias.
    for i in range(1, len(df) - 1):
        day = df.index[i]
        this_close = float(df["close"].iloc[i])
        next_close = float(df["close"].iloc[i + 1])

        action = "HOLD"
        trade_shares = 0.0
        trade_value = 0.0

        if (not in_position) and bool(df["entry_signal"].iloc[i]):
            notional = min(cfg.order_usd, cash)
            if notional > 0 and next_close > 0:
                trade_shares = notional / next_close
                cash -= notional
                shares += trade_shares
                in_position = True
                action = "BUY"
                trade_value = notional
                trade_count += 1

        elif in_position and bool(df["exit_signal"].iloc[i]):
            if shares > 0 and next_close > 0:
                trade_value = shares * next_close
                cash += trade_value
                trade_shares = -shares
                shares = 0.0
                in_position = False
                action = "SELL"
                trade_count += 1

        equity = cash + shares * this_close
        rows.append(
            {
                "date": str(day.date()),
                "close": this_close,
                "sentiment": float(df["sentiment"].iloc[i]),
                "headline_count": int(df["headline_count"].iloc[i]),
                "sentiment_sma": float(df["sentiment_sma"].iloc[i]) if pd.notna(df["sentiment_sma"].iloc[i]) else math.nan,
                "entry_signal": bool(df["entry_signal"].iloc[i]),
                "exit_signal": bool(df["exit_signal"].iloc[i]),
                "action": action,
                "trade_shares": float(trade_shares),
                "trade_value_usd": float(trade_value),
                "cash_usd": float(cash),
                "shares": float(shares),
                "equity_usd": float(equity),
            }
        )

    # Liquidate at final close for clean PnL accounting.
    final_close = float(df["close"].iloc[-1])
    if shares > 0:
        liquidation_value = shares * final_close
        cash += liquidation_value
        shares = 0.0
        trade_count += 1
    ending_equity = cash

    ledger = pd.DataFrame(rows)
    if ledger.empty:
        raise RuntimeError("No rows produced in ledger. Try a wider date range.")

    pnl = ending_equity - cfg.initial_cash
    ret = (ending_equity / cfg.initial_cash - 1.0) * 100.0
    peak = ledger["equity_usd"].cummax()
    drawdown = (ledger["equity_usd"] / peak) - 1.0

    stats = {
        "symbol": cfg.symbol,
        "period_start": cfg.start,
        "period_end": cfg.end,
        "initial_cash_usd": float(cfg.initial_cash),
        "order_usd": float(cfg.order_usd),
        "entry_sentiment": float(cfg.entry_sentiment),
        "exit_sentiment": float(cfg.exit_sentiment),
        "sentiment_window_days": int(cfg.sentiment_window),
        "min_headlines": int(cfg.min_headlines),
        "finnhub_enabled": bool(cfg.use_finnhub),
        "ending_equity_usd": float(ending_equity),
        "pnl_usd": float(pnl),
        "return_pct": float(ret),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "trade_count": int(trade_count),
    }

    return ledger, stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Headline sentiment trading bot: blend Finnhub and RSS headlines, buy on positive "
            "sentiment, and exit on negative sentiment."
        )
    )
    ap.add_argument("--symbol", required=True, help="Stock ticker, e.g. AAPL")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--initial-cash", type=float, default=10000.0)
    ap.add_argument("--order-usd", type=float, default=1000.0, help="USD notional per buy")
    ap.add_argument("--entry-sentiment", type=float, default=0.15, help="Buy when sentiment SMA is at or above this value")
    ap.add_argument("--exit-sentiment", type=float, default=-0.15, help="Sell when sentiment SMA is at or below this value")
    ap.add_argument("--sentiment-window", type=int, default=3, help="Rolling days for sentiment smoothing")
    ap.add_argument("--min-headlines", type=int, default=1, help="Minimum headlines needed on signal day")
    ap.add_argument("--finnhub-token", default=os.environ.get("FINNHUB_API_KEY", ""), help="Finnhub API key (or FINNHUB_API_KEY env var)")
    ap.add_argument("--disable-finnhub", action="store_true", help="Disable Finnhub and use only RSS")
    ap.add_argument("--rss-feed", action="append", default=[], help="Optional RSS feed URL. Can be provided multiple times")
    ap.add_argument("--cache-dir", default=".cache/headline_bot", help="Local cache directory for price and headlines")
    ap.add_argument("--refresh-cache", action="store_true", help="Ignore cache and fetch fresh data")
    ap.add_argument("--out-csv", default="", help="Optional output CSV for ledger")
    args = ap.parse_args()

    use_finnhub = (not args.disable_finnhub) and bool(args.finnhub_token.strip())

    cfg = Config(
        symbol=args.symbol.upper().strip(),
        start=args.start,
        end=args.end,
        initial_cash=args.initial_cash,
        order_usd=args.order_usd,
        entry_sentiment=args.entry_sentiment,
        exit_sentiment=args.exit_sentiment,
        sentiment_window=args.sentiment_window,
        min_headlines=args.min_headlines,
        finnhub_token=args.finnhub_token.strip(),
        use_finnhub=use_finnhub,
        rss_feeds=[str(x).strip() for x in args.rss_feed if str(x).strip()],
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        out_csv=(args.out_csv.strip() or None),
    )

    ledger, stats = run_backtest(cfg)

    print("\nRESULTS")
    for k, v in stats.items():
        print(f"{k}: {v}")

    if cfg.out_csv:
        ledger.to_csv(cfg.out_csv, index=False)
        print(f"\nWrote ledger CSV to: {cfg.out_csv}")


if __name__ == "__main__":
    main()
