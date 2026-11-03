"""
Strategy v2 — Phase 3: Smart Rotation with Hysteresis + Sector Caps

Improvements over v1:
  - Daily evaluation (checks every trading day, not fixed N-day cycle)
  - Hysteresis: only replace a position when a candidate beats it by
    switch_threshold (default 0.05) — eliminates most unnecessary churn
  - Minimum hold period: 5 trading days before a position can be replaced
  - Sector concentration cap: max 30% in any single sector
  - Cash as a position: holds cash if fewer than N stocks pass threshold

Usage:
    python strategy_v2.py                         # default params
    python strategy_v2.py --n 15 --threshold 0.05
    python strategy_v2.py --n 10 --threshold 0.08 --min-hold 10
"""

import sys
import math
import warnings
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Optional

from backtest import BacktestEngine, PerformanceAnalyzer, Portfolio
from scorer import SignalGenerator

warnings.filterwarnings("ignore")

CACHE_PATH = "models/scores_cache.parquet"


# ── Strategy class ────────────────────────────────────────────────────────────

class StrategyV2:
    """
    Stateful rotation strategy with hysteresis, min-hold, and sector caps.

    Each trading day the strategy:
      1. Looks up today's pre-computed scores
      2. For each held position eligible for replacement (min_hold elapsed):
           - If a non-held stock scores > held_score + switch_threshold: swap
      3. Fills any empty slots with best candidates respecting sector cap
      4. Returns equal-weight target (cash holds anything not allocated)
    """

    def __init__(
        self,
        scores_df:       pd.DataFrame,
        n_positions:     int   = 15,
        switch_threshold: float = 0.05,
        min_hold_days:   int   = 5,
        max_sector_pct:  float = 0.30,
    ):
        # Build fast lookup: date -> DataFrame sorted by score descending
        scores_df["date"] = pd.to_datetime(scores_df["date"])
        self._scores: Dict[pd.Timestamp, pd.DataFrame] = {
            d: grp.sort_values("score", ascending=False).reset_index(drop=True)
            for d, grp in scores_df.groupby("date")
        }

        self.n          = n_positions
        self.threshold  = switch_threshold
        self.min_hold   = min_hold_days
        # max positions allowed per sector
        self.max_per_sector = max(1, math.ceil(max_sector_pct * n_positions))

        # State tracking
        self._trade_day:  int  = 0                         # increments each call
        self._entry_day:  Dict[str, int]   = {}           # ticker -> trade_day when entered
        self._entry_score: Dict[str, float] = {}          # ticker -> score at entry
        self._held:       set = set()                     # current target set

    # ── Core logic ────────────────────────────────────────────────────────────
    def __call__(
        self,
        date:         pd.Timestamp,
        available_df: pd.DataFrame,
        portfolio:    Portfolio,
    ) -> Dict[str, float]:

        self._trade_day += 1
        scores = self._scores.get(date)

        # If no scores for this date: hold current positions unchanged
        if scores is None or len(scores) == 0:
            return self._equal_weight(self._held)

        # Filter scores to tickers active today
        available = set(available_df["ticker"].tolist()) if len(available_df) > 0 else set()
        if available:
            scores = scores[scores["ticker"].isin(available)].reset_index(drop=True)

        if len(scores) == 0:
            return self._equal_weight(self._held)

        score_map  = dict(zip(scores["ticker"], scores["score"]))
        sector_map = dict(zip(scores["ticker"], scores["sector"]))

        # Sync held set with actual portfolio (handles forced sells, delistings)
        actual_held = {t for t in portfolio.positions if portfolio.positions[t] > 0}
        # Keep only what's in both our target and actual portfolio
        self._held = self._held & actual_held
        # Any new positions that appeared (shouldn't happen, but be safe)
        for t in actual_held - self._held:
            self._held.add(t)
            if t not in self._entry_day:
                self._entry_day[t] = self._trade_day

        target = set(self._held)

        # ── Step 1: try to replace each eligible held position ────────────────
        # Process worst-scoring held positions first (most likely replaceable)
        eligible = [
            t for t in target
            if (self._trade_day - self._entry_day.get(t, 0)) >= self.min_hold
        ]
        eligible_sorted = sorted(eligible, key=lambda t: score_map.get(t, 0))

        for held_ticker in eligible_sorted:
            today_score  = score_map.get(held_ticker, 0.0)
            # Use HIGH-WATER MARK: compare against entry score to avoid
            # swapping on temporary dips (flip-flopping). Only replace when
            # a candidate clearly beats where the stock WAS when we bought it.
            entry_score  = self._entry_score.get(held_ticker, today_score)
            held_score   = max(today_score, entry_score)

            # Find best non-held candidate that respects sector cap
            candidate = self._best_candidate(scores, score_map, sector_map,
                                             target, exclude={held_ticker})
            if candidate is None:
                continue

            cand_ticker = candidate["ticker"]
            cand_score  = candidate["score"]

            # Switch only if clearly better than the high-water mark score
            if cand_score > held_score + self.threshold:
                target.discard(held_ticker)
                target.add(cand_ticker)
                self._entry_day[cand_ticker]   = self._trade_day
                self._entry_score[cand_ticker] = cand_score

        # ── Step 2: fill empty slots (up to N positions) ──────────────────────
        while len(target) < self.n:
            candidate = self._best_candidate(scores, score_map, sector_map, target)
            if candidate is None:
                break
            t = candidate["ticker"]
            target.add(t)
            self._entry_day[t]   = self._trade_day
            self._entry_score[t] = candidate["score"]

        # ── Cleanup: remove entry tracking for positions no longer held ───────
        for t in list(self._entry_day):
            if t not in target:
                del self._entry_day[t]
                self._entry_score.pop(t, None)

        changed = (target != self._held)
        self._held = target

        if not changed:
            # No position changes — return current weights so the engine
            # computes zero-delta trades (avoids daily drift rebalancing)
            close_prices = dict(zip(available_df["ticker"], available_df["Close"]))
            current_w = portfolio.weights(close_prices)
            return current_w if current_w else self._equal_weight(target)

        return self._equal_weight(target)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _best_candidate(
        self,
        scores:     pd.DataFrame,
        score_map:  Dict[str, float],
        sector_map: Dict[str, str],
        target:     set,
        exclude:    set = frozenset(),
    ) -> Optional[dict]:
        """Return best non-held ticker that fits within sector caps."""
        # Sector count in target (minus exclude)
        working = target - exclude
        sector_count: Dict[str, int] = {}
        for t in working:
            s = sector_map.get(t, "Unknown")
            sector_count[s] = sector_count.get(s, 0) + 1

        for _, row in scores.iterrows():
            t = row["ticker"]
            if t in target and t not in exclude:
                continue  # already held (and not being replaced)
            if t in exclude:
                continue  # the position being considered for replacement
            sector = row["sector"]
            if sector_count.get(sector, 0) < self.max_per_sector:
                return row.to_dict()
        return None

    @staticmethod
    def _equal_weight(tickers: set) -> Dict[str, float]:
        if not tickers:
            return {}
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}


# ── Runner ────────────────────────────────────────────────────────────────────

def run(
    n_positions:      int   = 15,
    switch_threshold: float = 0.05,
    min_hold_days:    int   = 5,
    max_sector_pct:   float = 0.30,
    start:            str   = "2022-02-01",
    end:              str   = "2026-04-22",
    save_csv:         bool  = True,
):
    # ── Load signal generator and pre-compute scores ──────────────────────────
    gen = SignalGenerator()
    scores_df = gen.precompute(start="2022-01-01", end=end, cache_path=CACHE_PATH)

    # ── Build strategy ────────────────────────────────────────────────────────
    strategy = StrategyV2(
        scores_df,
        n_positions      = n_positions,
        switch_threshold = switch_threshold,
        min_hold_days    = min_hold_days,
        max_sector_pct   = max_sector_pct,
    )

    # ── Run backtest (daily evaluation) ───────────────────────────────────────
    engine = BacktestEngine(rebalance_days=1)
    print(f"\nStrategy v2: N={n_positions}  threshold={switch_threshold}"
          f"  min_hold={min_hold_days}d  max_sector={max_sector_pct:.0%}")
    results = engine.run(strategy, start=start, end=end)

    if save_csv:
        results.analyzer.save_nav_csv("backtest_nav_v2.csv")

    # ── Trade summary ─────────────────────────────────────────────────────────
    tlog     = results.portfolio.trade_log
    buys     = [t for t in tlog if t.side == "BUY"]
    sells    = [t for t in tlog if "SELL" in t.side]
    total_tc = sum(t.cost for t in tlog)

    # Estimate number of rebalances that actually changed something
    buy_dates = pd.Series([t.date for t in buys]).nunique()

    print(f"\nTrade Summary:")
    print(f"  Total trades   : {len(tlog):,}  ({len(buys):,} buys, {len(sells):,} sells)")
    print(f"  Days with trades: {buy_dates}")
    print(f"  Total tx costs : ${total_tc:,.0f}  ({total_tc/results.initial_cash:.2%} of capital)")

    # Top holdings
    if buys:
        buy_df   = pd.DataFrame([{"ticker": t.ticker, "value": t.shares * t.price}
                                  for t in buys])
        top_held = buy_df.groupby("ticker")["value"].sum().sort_values(ascending=False)
        print(f"\nTop 10 most-traded tickers by total $ value:")
        for tkr, val in top_held.head(10).items():
            print(f"  {tkr:<6}  ${val:>12,.0f}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy v2: hysteresis rotation")
    parser.add_argument("--n",          type=int,   default=15)
    parser.add_argument("--threshold",  type=float, default=0.05)
    parser.add_argument("--min-hold",   type=int,   default=5)
    parser.add_argument("--max-sector", type=float, default=0.30)
    parser.add_argument("--start",      type=str,   default="2022-02-01")
    parser.add_argument("--end",        type=str,   default="2026-04-22")
    args = parser.parse_args()

    run(
        n_positions      = args.n,
        switch_threshold = args.threshold,
        min_hold_days    = args.min_hold,
        max_sector_pct   = args.max_sector,
        start            = args.start,
        end              = args.end,
    )
