"""
Strategy v1 — Phase 2: Composite Score Top-N Rotation

Strategy rules:
  - Every 10 trading days, score all ~500 S&P 500 stocks via composite signal
  - Buy the top-N by score (default N=15), equal weight
  - Hold cash for any remaining allocation when fewer than N stocks score above threshold
  - No risk management yet (added in Phase 4)

Routing:
  - Sector classifier for all stocks (all 11 sectors)
  - Sector regressor where val_ic >= 0.07, global v2 regressor otherwise
  - Financial Services: clf only (regressor inverted)

Usage:
    python strategy_v1.py                        # default N=15
    python strategy_v1.py --n 10                 # top-10 positions
    python strategy_v1.py --n 20 --min-score 0.5 # with score threshold
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd

from backtest import BacktestEngine, PerformanceAnalyzer
from scorer import SignalGenerator

warnings.filterwarnings("ignore")


def make_strategy(generator: SignalGenerator, n_positions: int = 15,
                  min_score: float = 0.0):
    """
    Returns a strategy function compatible with BacktestEngine.

    Parameters
    ----------
    generator    : pre-loaded SignalGenerator
    n_positions  : max number of simultaneous long positions
    min_score    : minimum composite score to enter a position
                   (0.0 = take top-N regardless of absolute score)
    """
    def strategy(date, available_df, portfolio):
        scores = generator.score(date)
        if scores is None or len(scores) == 0:
            return {}

        # Filter to tickers active in the market on this date
        available = set(available_df["ticker"].tolist())
        scores = scores[scores["ticker"].isin(available)]

        # Apply minimum score threshold
        if min_score > 0:
            scores = scores[scores["score"] >= min_score]

        # Take top-N
        top = scores.head(n_positions)
        if len(top) == 0:
            return {}

        w = 1.0 / len(top)
        return dict(zip(top["ticker"], [w] * len(top)))

    strategy.__name__ = f"top_{n_positions}_score"
    return strategy


def run(n_positions: int = 15, min_score: float = 0.0,
        start: str = "2022-02-01", end: str = "2026-04-22",
        rebalance_days: int = 10, save_csv: bool = True):

    # ── Load signal generator ─────────────────────────────────────────────────
    gen = SignalGenerator()

    # ── Run backtest ──────────────────────────────────────────────────────────
    engine = BacktestEngine(rebalance_days=rebalance_days)
    strategy_fn = make_strategy(gen, n_positions=n_positions, min_score=min_score)

    print(f"\nStrategy: Top-{n_positions} composite score  "
          f"| min_score={min_score}  | rebalance every {rebalance_days}d")
    results = engine.run(strategy_fn, start=start, end=end)

    # ── Save NAV ──────────────────────────────────────────────────────────────
    if save_csv:
        results.analyzer.save_nav_csv("backtest_nav_v1.csv")

    # ── Trade breakdown ───────────────────────────────────────────────────────
    tlog = results.portfolio.trade_log
    buys  = [t for t in tlog if t.side == "BUY"]
    sells = [t for t in tlog if "SELL" in t.side]
    total_tc = sum(t.cost for t in tlog)

    print(f"\nTrade Summary:")
    print(f"  Total trades   : {len(tlog):,}  ({len(buys):,} buys, {len(sells):,} sells)")
    print(f"  Avg positions  : {len(buys) / max(1, len(tlog)//2 + 1):.1f} per rebalance")
    print(f"  Total tx costs : ${total_tc:,.0f}  ({total_tc/results.initial_cash:.2%} of capital)")

    # ── Top holdings analysis ─────────────────────────────────────────────────
    if buys:
        buy_df = pd.DataFrame([{
            "ticker": t.ticker, "date": t.date, "value": t.shares * t.price
        } for t in buys])
        top_held = buy_df.groupby("ticker")["value"].sum().sort_values(ascending=False)
        print(f"\nTop 10 most-traded tickers by total dollar value:")
        for tkr, val in top_held.head(10).items():
            print(f"  {tkr:<6}  ${val:>12,.0f}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy v1: Top-N composite score rotation")
    parser.add_argument("--n",          type=int,   default=15,   help="Number of positions")
    parser.add_argument("--min-score",  type=float, default=0.0,  help="Min score threshold")
    parser.add_argument("--rebalance",  type=int,   default=10,   help="Rebalance frequency (trading days)")
    parser.add_argument("--start",      type=str,   default="2022-02-01")
    parser.add_argument("--end",        type=str,   default="2026-04-22")
    args = parser.parse_args()

    run(
        n_positions    = args.n,
        min_score      = args.min_score,
        rebalance_days = args.rebalance,
        start          = args.start,
        end            = args.end,
    )
