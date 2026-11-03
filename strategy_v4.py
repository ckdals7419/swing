"""
Strategy v4 — Phase 5: Walk-Forward Model Improvements

Identical strategy logic to strategy_v3 (N=25, no risk overlays —
the Phase 4 best config) but powered by walk-forward vintage models
from scorer_v2. This isolates the model improvement signal from any
strategy-layer changes.

The core hypothesis: training on ≤6-month-old data will materially
improve OOS IC, which should translate to better stock selection.

Usage:
    python strategy_v4.py                           # default N=25
    python strategy_v4.py --n 10 --threshold 0.10   # concentrated
    python strategy_v4.py --compare                 # show v3 side-by-side

Compare command runs both v3 and v4 scorers and prints tearsheets.
"""

import sys
import math
import warnings
import argparse
import numpy as np
import pandas as pd
from typing import Dict

from backtest import BacktestEngine, PerformanceAnalyzer, Portfolio
from scorer_v2 import SignalGeneratorV2

warnings.filterwarnings("ignore")

CACHE_PATH_V4 = "models/scores_cache_v4.parquet"


# ── Strategy (reuses StrategyV3 logic; only scorer changes) ───────────────────

class StrategyV4:
    """
    Long-only rotation using walk-forward XGBoost signals.

    Identical to StrategyV3 in all strategy-layer decisions:
      - Equal-weight N positions
      - Hysteresis rotation (switch_threshold + min_hold)
      - Sector concentration cap (max_sector_pct)
      - Optional min score filter
    No regime or circuit-breaker overlays (those added no alpha in Phase 4).
    """

    def __init__(
        self,
        scores_df:        pd.DataFrame,
        n_positions:      int   = 25,
        switch_threshold: float = 0.10,
        min_hold_days:    int   = 10,
        max_sector_pct:   float = 0.30,
        min_score:        float = 0.0,
    ):
        scores_df["date"] = pd.to_datetime(scores_df["date"])
        self._scores: Dict[pd.Timestamp, pd.DataFrame] = {
            d: grp.sort_values("score", ascending=False).reset_index(drop=True)
            for d, grp in scores_df.groupby("date")
        }
        self.n          = n_positions
        self.threshold  = switch_threshold
        self.min_hold   = min_hold_days
        self.max_sector = max(1, math.ceil(max_sector_pct * n_positions))
        self.min_score  = min_score

        self._trade_day:  int  = 0
        self._entry_day:  Dict[str, int]   = {}
        self._entry_score: Dict[str, float] = {}
        self._held:       set = set()

    def __call__(
        self,
        date: pd.Timestamp,
        available_df: pd.DataFrame,
        portfolio: Portfolio,
    ) -> Dict[str, float]:

        self._trade_day += 1
        scores = self._scores.get(date)
        if scores is None or len(scores) == 0:
            return self._equal_weight(self._held)

        available = set(available_df["ticker"].tolist()) if len(available_df) > 0 else set()
        if available:
            scores = scores[scores["ticker"].isin(available)].reset_index(drop=True)

        scores = scores[scores["score"] >= self.min_score].reset_index(drop=True)

        if len(scores) == 0:
            self._held = set()
            self._entry_day.clear()
            self._entry_score.clear()
            return {}

        score_map  = dict(zip(scores["ticker"], scores["score"]))
        sector_map = dict(zip(scores["ticker"], scores["sector"]))

        # Sync held set
        actual_held = {t for t in portfolio.positions if portfolio.positions[t] > 0}
        self._held = self._held & actual_held
        for t in actual_held - self._held:
            self._held.add(t)
            self._entry_day.setdefault(t, self._trade_day)

        target = set(self._held)

        # Hysteresis rotation
        eligible = sorted(
            [t for t in target
             if (self._trade_day - self._entry_day.get(t, 0)) >= self.min_hold],
            key=lambda t: score_map.get(t, 0)
        )
        for held_t in eligible:
            today_score  = score_map.get(held_t, 0.0)
            entry_score  = self._entry_score.get(held_t, today_score)
            held_score   = max(today_score, entry_score)
            candidate    = self._best_candidate(scores, score_map, sector_map,
                                                target, exclude={held_t})
            if candidate is None:
                continue
            if candidate["score"] > held_score + self.threshold:
                target.discard(held_t)
                ct = candidate["ticker"]
                target.add(ct)
                self._entry_day[ct]   = self._trade_day
                self._entry_score[ct] = candidate["score"]

        # Fill empty slots
        while len(target) < self.n:
            candidate = self._best_candidate(scores, score_map, sector_map, target)
            if candidate is None:
                break
            t = candidate["ticker"]
            target.add(t)
            self._entry_day[t]   = self._trade_day
            self._entry_score[t] = candidate["score"]

        # Cleanup stale tracking
        for t in list(self._entry_day):
            if t not in target:
                del self._entry_day[t]
                self._entry_score.pop(t, None)

        self._held = target
        return self._equal_weight(target)

    def _best_candidate(self, scores, score_map, sector_map, target, exclude=frozenset()):
        working = target - exclude
        sector_count: Dict[str, int] = {}
        for t in working:
            s = sector_map.get(t, "Unknown")
            sector_count[s] = sector_count.get(s, 0) + 1

        for _, row in scores.iterrows():
            t = row["ticker"]
            if t in target and t not in exclude:
                continue
            if t in exclude:
                continue
            sector = row["sector"]
            if sector_count.get(sector, 0) < self.max_sector:
                return row.to_dict()
        return None

    def _equal_weight(self, tickers):
        if not tickers:
            return {}
        w = 1.0 / max(len(tickers), 1)
        return {t: w for t in tickers}


# ── Runner ────────────────────────────────────────────────────────────────────

def run(
    n_positions:      int   = 25,
    switch_threshold: float = 0.10,
    min_hold_days:    int   = 10,
    max_sector_pct:   float = 0.30,
    min_score:        float = 0.0,
    start:            str   = "2022-02-01",
    end:              str   = "2026-04-22",
    save_csv:         bool  = True,
):
    gen = SignalGeneratorV2()
    scores_df = gen.precompute(start="2022-01-01", end=end, cache_path=CACHE_PATH_V4)

    strategy = StrategyV4(
        scores_df,
        n_positions=n_positions,
        switch_threshold=switch_threshold,
        min_hold_days=min_hold_days,
        max_sector_pct=max_sector_pct,
        min_score=min_score,
    )

    engine = BacktestEngine(rebalance_days=1)
    print(f"\nStrategy v4 (walk-forward): N={n_positions}  "
          f"threshold={switch_threshold}  min_hold={min_hold_days}d  "
          f"max_sector={max_sector_pct:.0%}")

    results = engine.run(strategy, start=start, end=end)

    if save_csv:
        results.analyzer.save_nav_csv("backtest_nav_v4.csv")

    tlog  = results.portfolio.trade_log
    buys  = [t for t in tlog if t.side == "BUY"]
    sells = [t for t in tlog if "SELL" in t.side]
    total_tc = sum(t.cost for t in tlog)
    buy_dates = pd.Series([t.date for t in buys]).nunique() if buys else 0

    print(f"\nTrade Summary:")
    print(f"  Total trades    : {len(tlog):,}  ({len(buys):,} buys, {len(sells):,} sells)")
    print(f"  Days with trades: {buy_dates}")
    print(f"  Total tx costs  : ${total_tc:,.0f}  ({total_tc/results.initial_cash:.2%} of capital)")

    return results


def run_compare(start="2022-02-01", end="2026-04-22"):
    """Run v3 (static) and v4 (walk-forward) side-by-side for comparison."""
    from scorer import SignalGenerator
    from strategy_v3 import StrategyV3, RegimeDetector

    # ── v3 (static) ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Running v3 (static models, trained ≤ 2019-12-31)")
    print("="*60)
    gen_v3 = SignalGenerator()
    scores_v3 = gen_v3.precompute(
        start="2022-01-01", end=end, cache_path="models/scores_cache.parquet"
    )
    regime = RegimeDetector.__new__(RegimeDetector)
    regime.get_regime = lambda date: {"exposure": 1.0, "regime": "always_on"}
    strat_v3 = StrategyV3(scores_v3, regime, n_positions=25)
    engine_v3 = BacktestEngine(rebalance_days=1)
    res_v3 = engine_v3.run(strat_v3, start=start, end=end)

    # ── v4 (walk-forward) ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Running v4 (walk-forward models)")
    print("="*60)
    gen_v4 = SignalGeneratorV2()
    scores_v4 = gen_v4.precompute(
        start="2022-01-01", end=end, cache_path=CACHE_PATH_V4
    )
    strat_v4 = StrategyV4(scores_v4, n_positions=25)
    engine_v4 = BacktestEngine(rebalance_days=1)
    res_v4 = engine_v4.run(strat_v4, start=start, end=end)

    # ── Side-by-side tearsheet ─────────────────────────────────────────────────
    a3 = res_v3.analyzer
    a4 = res_v4.analyzer

    spy = res_v3.spy
    ab = PerformanceAnalyzer(
        spy / spy.iloc[0] * res_v3.initial_cash,
        risk_free=0.045
    )

    print("\n")
    print("=" * 65)
    print(f"  Comparison Tearsheet  ({start} → {end})")
    print("=" * 65)
    print(f"  {'Metric':<26} {'v3 (static)':>12}  {'v4 (wf)':>10}  {'SPY':>8}")
    print("-" * 65)

    rows = [
        ("Total Return",    a3.total_return(),    a4.total_return(),    ab.total_return()),
        ("Ann. Return",     a3.annualized_return(), a4.annualized_return(), ab.annualized_return()),
        ("Max Drawdown",    a3.max_drawdown(),    a4.max_drawdown(),    ab.max_drawdown()),
        ("Sharpe Ratio",    a3.sharpe(),          a4.sharpe(),          ab.sharpe()),
        ("Sortino Ratio",   a3.sortino(),         a4.sortino(),         ab.sortino()),
        ("Calmar Ratio",    a3.calmar(),          a4.calmar(),          ab.calmar()),
        ("Win Rate (Mo.)",  a3.win_rate_monthly(), a4.win_rate_monthly(), ab.win_rate_monthly()),
    ]
    for name, v3_val, v4_val, spy_val in rows:
        if "Rate" in name or "Return" in name or "Drawdown" in name:
            print(f"  {name:<26} {v3_val:>11.2%}  {v4_val:>9.2%}  {spy_val:>7.2%}")
        else:
            print(f"  {name:<26} {v3_val:>11.3f}  {v4_val:>9.3f}  {spy_val:>7.3f}")
    print("=" * 65)

    res_v3.analyzer.save_nav_csv("backtest_nav_v3.csv")
    res_v4.analyzer.save_nav_csv("backtest_nav_v4.csv")

    return res_v3, res_v4


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy v4: walk-forward models")
    parser.add_argument("--n",          type=int,   default=25)
    parser.add_argument("--threshold",  type=float, default=0.10)
    parser.add_argument("--min-hold",   type=int,   default=10)
    parser.add_argument("--max-sector", type=float, default=0.30)
    parser.add_argument("--min-score",  type=float, default=0.0)
    parser.add_argument("--start",      type=str,   default="2022-02-01")
    parser.add_argument("--end",        type=str,   default="2026-04-22")
    parser.add_argument("--compare",    action="store_true",
                        help="Run v3 vs v4 side-by-side comparison")
    args = parser.parse_args()

    if args.compare:
        run_compare(start=args.start, end=args.end)
    else:
        run(
            n_positions=args.n,
            switch_threshold=args.threshold,
            min_hold_days=args.min_hold,
            max_sector_pct=args.max_sector,
            min_score=args.min_score,
            start=args.start,
            end=args.end,
        )
