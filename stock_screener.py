import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


@dataclass
class ScreenerConfig:
    symbol_list: str | None
    min_price: float
    max_price: float
    max_market_cap_usd: float
    min_market_cap_usd: float
    min_avg_volume: float
    min_revenue_growth: float
    max_pe_ratio: float
    min_beta: float
    max_beta: float
    score_weight_growth: float
    score_weight_volatility: float
    score_weight_valuation: float
    score_weight_size: float
    output_csv: str | None
    cache_dir: str


TECH_UNIVERSE = [
    # Large-cap reference
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "NFLX",
    # Mid-cap
    "ARM", "SNOW", "PLTR", "COIN", "MARA", "RIOT",
    # Small-cap / Micro-cap tech
    "SMCI", "AVGO", "SOFI", "NIO", "RIVN",
    # Additional small-cap tech picks
    "UPST", "RBLX", "CRWD", "PSTG", "ZM", "DOCU", "OKTA", "SVGN",
    "HUBS", "TWLO", "TTD", "NET", "DDOG", "FSLY", "ESTC", "SPLK",
    "SMAR", "PAYX", "ILMN", "ADBE", "CRM", "INTU", "NOW", "WDAY",
    "ANSS", "LRCX", "ASML", "QCOM", "AMAT", "CDNS", "SNPS",
    "GOOG", "GEVO", "MELI", "SE", "AFRM", "ROKU", "SQ", "SHOP",
]


def fetch_stock_metrics(symbol: str) -> dict | None:
    """Fetch key metrics for a stock."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        price = info.get("currentPrice", 0.0)
        if not price or price <= 0:
            return None

        market_cap = info.get("marketCap", 0)
        avg_volume = info.get("averageVolume", 0)
        pe_ratio = info.get("trailingPE", float("inf"))
        beta = info.get("beta", 1.0)
        revenue_per_share = info.get("revenuePerShare", 0.0)
        trailing_twelve_months_revenue = info.get("totalRevenue", 0)

        # Fetch historical data for metrics
        hist = ticker.history(period="2y")
        if hist.empty:
            return None

        # Calculate 1-year return
        one_year_ago = len(hist) - 252
        if one_year_ago > 0:
            price_1y_ago = hist["Close"].iloc[one_year_ago]
            one_year_return = (price - price_1y_ago) / price_1y_ago if price_1y_ago > 0 else 0.0
        else:
            one_year_return = 0.0

        # Calculate volatility
        returns = hist["Close"].pct_change().dropna()
        volatility = returns.std() * math.sqrt(252) if len(returns) > 0 else 0.0

        # Revenue growth (YoY if available)
        quarterly_data = ticker.quarterly_financials
        if quarterly_data is not None and not quarterly_data.empty:
            try:
                revenues = quarterly_data.loc["Total Revenue"] if "Total Revenue" in quarterly_data.index else None
                if revenues is not None and len(revenues) >= 2:
                    latest_revenue = revenues.iloc[0]
                    prev_revenue = revenues.iloc[1]
                    revenue_growth = (latest_revenue - prev_revenue) / prev_revenue if prev_revenue > 0 else 0.0
                else:
                    revenue_growth = 0.0
            except Exception:
                revenue_growth = 0.0
        else:
            revenue_growth = 0.0

        # PEG ratio estimate (if we have growth)
        if pe_ratio > 0 and revenue_growth > 0:
            peg_ratio = pe_ratio / (revenue_growth * 100 + 1)
        else:
            peg_ratio = float("inf")

        return {
            "symbol": symbol,
            "price": float(price),
            "market_cap": float(market_cap) if market_cap else 0.0,
            "avg_volume": float(avg_volume) if avg_volume else 0.0,
            "pe_ratio": float(pe_ratio) if pe_ratio and pe_ratio != float("inf") else 0.0,
            "beta": float(beta) if beta else 1.0,
            "volatility": float(volatility),
            "one_year_return": float(one_year_return),
            "revenue_growth": float(revenue_growth),
            "peg_ratio": float(peg_ratio) if peg_ratio != float("inf") else 0.0,
        }
    except Exception as e:
        return None


def score_stock(metrics: dict, cfg: ScreenerConfig) -> float:
    """Compute composite UPsIDE POTENTIAL score."""
    score = 0.0

    # Growth component (higher is better)
    growth_score = 0.0
    if metrics["revenue_growth"] > cfg.min_revenue_growth:
        growth_score = min(100.0, metrics["revenue_growth"] * 200)
    score += growth_score * cfg.score_weight_growth

    # Volatility component (moderate volatility = upside potential)
    vol_score = 0.0
    if 0.3 <= metrics["volatility"] <= 1.2:
        vol_score = 100.0
    elif metrics["volatility"] < 0.3:
        vol_score = 50.0
    elif metrics["volatility"] > 1.2:
        vol_score = 75.0
    score += vol_score * cfg.score_weight_volatility

    # Valuation component (lower PE = more upside from rerating)
    val_score = 0.0
    if metrics["pe_ratio"] > 0 and metrics["pe_ratio"] < cfg.max_pe_ratio * 2:
        val_score = max(0.0, 100.0 - (metrics["pe_ratio"] / 50.0) * 100.0)
    score += val_score * cfg.score_weight_valuation

    # Size component (smaller = more upside potential)
    size_score = 0.0
    if metrics["market_cap"] > 0 and metrics["market_cap"] < cfg.max_market_cap_usd:
        ratio = metrics["market_cap"] / cfg.max_market_cap_usd
        size_score = (1.0 - ratio) * 100.0
    score += size_score * cfg.score_weight_size

    return score


def screen_stocks(cfg: ScreenerConfig) -> pd.DataFrame:
    """Screen universe of tech stocks for upside potential."""
    if cfg.symbol_list:
        symbols = [s.strip().upper() for s in cfg.symbol_list.split(",") if s.strip()]
    else:
        symbols = TECH_UNIVERSE

    results: list[dict] = []

    print(f"\nScreening {len(symbols)} tech stocks...")
    for i, symbol in enumerate(symbols):
        print(f"  [{i+1}/{len(symbols)}] {symbol}...", end=" ")
        metrics = fetch_stock_metrics(symbol)
        if metrics is None:
            print("SKIP (no data)")
            continue

        # Apply filters
        if not (cfg.min_price <= metrics["price"] <= cfg.max_price):
            print(f"SKIP (price)")
            continue
        if not (cfg.min_market_cap_usd <= metrics["market_cap"] <= cfg.max_market_cap_usd):
            print(f"SKIP (market_cap)")
            continue
        if metrics["avg_volume"] < cfg.min_avg_volume:
            print(f"SKIP (volume)")
            continue
        if metrics["pe_ratio"] > 0 and metrics["pe_ratio"] > cfg.max_pe_ratio * 1.5:
            print(f"SKIP (pe_ratio)")
            continue
        if not (cfg.min_beta <= metrics["beta"] <= cfg.max_beta):
            print(f"SKIP (beta)")
            continue

        score = score_stock(metrics, cfg)
        metrics["upside_score"] = float(score)
        results.append(metrics)
        print(f"OK (score={score:.1f})")

    if not results:
        print("No stocks passed filters.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("upside_score", ascending=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Tech stock screener: find small-cap tech companies with huge upside potential "
            "based on growth, volatility, valuation, and market cap."
        )
    )
    ap.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbol list (or use default tech universe)",
    )
    ap.add_argument("--min-price", type=float, default=2.0)
    ap.add_argument("--max-price", type=float, default=1000.0)
    ap.add_argument("--min-market-cap", type=float, default=100e6, help="Min market cap in USD")
    ap.add_argument("--max-market-cap", type=float, default=5e9, help="Max market cap in USD")
    ap.add_argument("--min-avg-volume", type=float, default=500_000)
    ap.add_argument("--min-revenue-growth", type=float, default=0.0, help="Min annual revenue growth rate")
    ap.add_argument("--max-pe-ratio", type=float, default=100.0)
    ap.add_argument("--min-beta", type=float, default=0.5, help="Min beta for upside volatility")
    ap.add_argument("--max-beta", type=float, default=3.0, help="Max beta risk tolerance")
    ap.add_argument("--weight-growth", type=float, default=0.4)
    ap.add_argument("--weight-volatility", type=float, default=0.25)
    ap.add_argument("--weight-valuation", type=float, default=0.2)
    ap.add_argument("--weight-size", type=float, default=0.15)
    ap.add_argument("--out-csv", default="", help="Output CSV path")
    ap.add_argument("--cache-dir", default=".cache/stock_screener")
    args = ap.parse_args()

    cfg = ScreenerConfig(
        symbol_list=args.symbols.strip() or None,
        min_price=float(args.min_price),
        max_price=float(args.max_price),
        min_market_cap_usd=float(args.min_market_cap),
        max_market_cap_usd=float(args.max_market_cap),
        min_avg_volume=float(args.min_avg_volume),
        min_revenue_growth=float(args.min_revenue_growth),
        max_pe_ratio=float(args.max_pe_ratio),
        min_beta=float(args.min_beta),
        max_beta=float(args.max_beta),
        score_weight_growth=float(args.weight_growth),
        score_weight_volatility=float(args.weight_volatility),
        score_weight_valuation=float(args.weight_valuation),
        score_weight_size=float(args.weight_size),
        output_csv=(args.out_csv.strip() or None),
        cache_dir=args.cache_dir.strip() or ".cache/stock_screener",
    )

    results = screen_stocks(cfg)

    if not results.empty:
        print("\nTOP UPSIDE CANDIDATES:")
        print(results[["symbol", "price", "market_cap", "pe_ratio", "volatility", "revenue_growth", "upside_score"]].to_string(index=False))

        if cfg.output_csv:
            results.to_csv(cfg.output_csv, index=False)
            print(f"\nWrote results to: {cfg.output_csv}")


if __name__ == "__main__":
    main()
