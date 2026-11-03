"""
Strategy v3 — Phase 4: Risk-Managed Rotation

Builds on strategy_v2 (hysteresis rotation) and adds optional risk overlays:
  - Regime-based exposure scaling (SPY trend + VIX level)
  - Portfolio drawdown circuit breaker (hard cap on losses)
  - Per-position stop loss (cut individual losers)
  - Minimum score threshold (cash when signal is weak)

Phase 4 findings (2022-2026 backtest):
  All active risk overlays (regime, CB, stops) reduce MDD but destroy even
  more return, because the underlying signal (IC=0.04) has negative alpha.
  The best risk-adjusted improvement is simply more positions (N=25)
  which reduces idiosyncratic risk through diversification.

  Default config: N=25, all overlays disabled — matches SPY MDD (-22.6%)
  with positive return (3.80%).

Usage:
    python strategy_v3.py                         # N=25 default
    python strategy_v3.py --n 10 --threshold 0.10 # concentrated, higher risk/return
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


# ── Regime detection ─────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Determines market regime from SPY and VIX data.
    All lookups use only data available on or before the decision date.
    """

    def __init__(self, spy_path: str = "data/spy_daily.csv",
                 vix_path: str = "data/vix_daily.csv"):
        spy = pd.read_csv(spy_path, parse_dates=["Date"]).sort_values("Date")
        spy = spy.set_index("Date")
        self._spy_close = spy["Adj Close"] if "Adj Close" in spy.columns else spy["Close"]
        self._spy_sma200 = self._spy_close.rolling(200, min_periods=200).mean()
        self._spy_high252 = self._spy_close.rolling(252, min_periods=20).max()

        vix = pd.read_csv(vix_path, parse_dates=["Date"]).sort_values("Date")
        vix = vix.set_index("Date")
        self._vix = vix["vix_close"]

    def get_regime(self, date: pd.Timestamp) -> dict:
        """Return regime info for a given date."""
        # Get latest available data on or before date
        spy_c = self._spy_close.loc[:date]
        sma200 = self._spy_sma200.loc[:date]
        vix = self._vix.loc[:date]
        high252 = self._spy_high252.loc[:date]

        if len(spy_c) == 0 or len(sma200) == 0 or len(vix) == 0:
            return {"exposure": 1.0, "regime": "unknown"}

        spy_price = float(spy_c.iloc[-1])
        sma_val = float(sma200.iloc[-1])
        vix_val = float(vix.iloc[-1])
        high_val = float(high252.iloc[-1])

        above_sma200 = spy_price > sma_val if not np.isnan(sma_val) else True
        spy_dd = (spy_price - high_val) / high_val if high_val > 0 else 0.0

        # Crisis: VIX > 35 or SPY drawdown > 15%
        if vix_val > 35 or spy_dd < -0.15:
            return {"exposure": 0.0, "regime": "crisis",
                    "vix": vix_val, "spy_dd": spy_dd, "above_sma200": above_sma200}

        # Bear: below SMA200 and VIX elevated
        if not above_sma200 and vix_val > 25:
            return {"exposure": 0.30, "regime": "bear",
                    "vix": vix_val, "spy_dd": spy_dd, "above_sma200": above_sma200}

        # Cautious: mixed signals
        if not above_sma200 or vix_val > 25:
            return {"exposure": 0.50, "regime": "cautious",
                    "vix": vix_val, "spy_dd": spy_dd, "above_sma200": above_sma200}

        # Mild bull: above SMA200 but VIX 20-25
        if vix_val > 20:
            return {"exposure": 0.80, "regime": "mild_bull",
                    "vix": vix_val, "spy_dd": spy_dd, "above_sma200": above_sma200}

        # Strong bull
        return {"exposure": 1.0, "regime": "strong_bull",
                "vix": vix_val, "spy_dd": spy_dd, "above_sma200": above_sma200}


# ── Strategy class ────────────────────────────────────────────────────────────

class StrategyV3:
    """
    Stateful rotation strategy with regime filter + drawdown circuit breaker.

    Layers (from outer to inner):
      1. Regime filter → caps max equity exposure
      2. Portfolio drawdown circuit breaker → further caps exposure
      3. Gradual re-entry → prevents whipsaw after recovery
      4. Hysteresis rotation → same as v2 (min hold, switch threshold)
      5. Min score threshold → cash when signal is weak
    """

    def __init__(
        self,
        scores_df:        pd.DataFrame,
        regime_detector:  RegimeDetector,
        n_positions:      int   = 25,
        switch_threshold: float = 0.10,
        min_hold_days:    int   = 10,
        max_sector_pct:   float = 0.30,
        min_score:        float = 0.0,
        # Per-position stop loss (0.0 = disabled)
        position_stop:    float = 0.0,
        # Circuit breaker thresholds (-1.0 = disabled)
        dd_warn:          float = -1.0,
        dd_danger:        float = -1.0,
        dd_stop:          float = -1.0,
    ):
        scores_df["date"] = pd.to_datetime(scores_df["date"])
        self._scores: Dict[pd.Timestamp, pd.DataFrame] = {
            d: grp.sort_values("score", ascending=False).reset_index(drop=True)
            for d, grp in scores_df.groupby("date")
        }

        self.regime = regime_detector
        self.n = n_positions
        self.threshold = switch_threshold
        self.min_hold = min_hold_days
        self.max_per_sector = max(1, math.ceil(max_sector_pct * n_positions))
        self.min_score = min_score

        self.position_stop = position_stop
        self.dd_warn = dd_warn
        self.dd_danger = dd_danger
        self.dd_stop = dd_stop

        # State
        self._trade_day: int = 0
        self._entry_day: Dict[str, int] = {}
        self._entry_score: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}  # ticker -> price at entry
        self._held: set = set()
        self._hwm: float = 0.0  # portfolio high-water mark
        self._last_exposure: float = 1.0  # last applied exposure level
        self._current_exposure: float = 1.0

    def __call__(
        self,
        date: pd.Timestamp,
        available_df: pd.DataFrame,
        portfolio: Portfolio,
    ) -> Dict[str, float]:

        self._trade_day += 1
        close_prices = dict(zip(available_df["ticker"], available_df["Close"])) if len(available_df) > 0 else {}
        current_nav = portfolio.nav(close_prices)

        # Track portfolio high-water mark and drawdown
        if current_nav > self._hwm:
            self._hwm = current_nav
        port_dd = (current_nav - self._hwm) / self._hwm if self._hwm > 0 else 0.0

        # ── Layer 1: Regime-based max exposure ────────────────────────────
        regime_info = self.regime.get_regime(date)
        regime_exposure = regime_info["exposure"]

        # ── Layer 2: Circuit breaker overlay ──────────────────────────────
        if port_dd < self.dd_stop:
            cb_exposure = 0.0
        elif port_dd < self.dd_danger:
            cb_exposure = 0.20
        elif port_dd < self.dd_warn:
            cb_exposure = 0.50
        else:
            cb_exposure = 1.0

        # Take the tighter constraint
        max_exposure = min(regime_exposure, cb_exposure)

        # Store exposure for weight scaling (NOT position count)
        self._current_exposure = max_exposure

        # ── Get scores ────────────────────────────────────────────────────
        scores = self._scores.get(date)
        if scores is None or len(scores) == 0:
            return self._weighted(self._held, close_prices, portfolio)

        available = set(available_df["ticker"].tolist()) if len(available_df) > 0 else set()
        if available:
            scores = scores[scores["ticker"].isin(available)].reset_index(drop=True)

        # ── Layer 5: Min score filter ─────────────────────────────────────
        scores = scores[scores["score"] >= self.min_score].reset_index(drop=True)

        if len(scores) == 0 or max_exposure == 0:
            # Liquidate everything — go to cash
            if self._held:
                self._held = set()
                self._entry_day.clear()
                self._entry_score.clear()
                self._entry_price.clear()
                return {}
            return {}

        score_map = dict(zip(scores["ticker"], scores["score"]))
        sector_map = dict(zip(scores["ticker"], scores["sector"]))

        # Sync held set with actual portfolio
        actual_held = {t for t in portfolio.positions if portfolio.positions[t] > 0}
        self._held = self._held & actual_held
        for t in actual_held - self._held:
            self._held.add(t)
            if t not in self._entry_day:
                self._entry_day[t] = self._trade_day
                self._entry_price[t] = close_prices.get(t, 0)

        target = set(self._held)

        # ── Per-position stop loss ────────────────────────────────────────
        if self.position_stop < 0:
            stopped_out = []
            for t in list(target):
                entry_p = self._entry_price.get(t)
                current_p = close_prices.get(t)
                if entry_p and current_p and entry_p > 0:
                    pnl = (current_p - entry_p) / entry_p
                    if pnl < self.position_stop:
                        stopped_out.append(t)
                        target.discard(t)
            # Remove tracking for stopped positions
            for t in stopped_out:
                self._entry_day.pop(t, None)
                self._entry_score.pop(t, None)
                self._entry_price.pop(t, None)

        # ── Hysteresis rotation (same as v2) ──────────────────────────────
        eligible = [
            t for t in target
            if (self._trade_day - self._entry_day.get(t, 0)) >= self.min_hold
        ]
        eligible_sorted = sorted(eligible, key=lambda t: score_map.get(t, 0))

        for held_ticker in eligible_sorted:
            today_score = score_map.get(held_ticker, 0.0)
            entry_score = self._entry_score.get(held_ticker, today_score)
            held_score = max(today_score, entry_score)

            candidate = self._best_candidate(scores, score_map, sector_map,
                                             target, exclude={held_ticker})
            if candidate is None:
                continue

            cand_score = candidate["score"]
            if cand_score > held_score + self.threshold:
                target.discard(held_ticker)
                ct = candidate["ticker"]
                target.add(ct)
                self._entry_day[ct] = self._trade_day
                self._entry_score[ct] = cand_score
                self._entry_price[ct] = close_prices.get(ct, 0)

        # ── Fill empty slots up to N ──────────────────────────────────────
        while len(target) < self.n:
            candidate = self._best_candidate(scores, score_map, sector_map, target)
            if candidate is None:
                break
            t = candidate["ticker"]
            target.add(t)
            self._entry_day[t] = self._trade_day
            self._entry_score[t] = candidate["score"]
            self._entry_price[t] = close_prices.get(t, 0)

        # Cleanup stale tracking
        for t in list(self._entry_day):
            if t not in target:
                del self._entry_day[t]
                self._entry_score.pop(t, None)
                self._entry_price.pop(t, None)

        changed = (target != self._held)
        # Only rebalance weights if positions changed OR exposure shifted
        # by >20pp (avoids constant small rebalances from regime noise)
        exposure_shift = abs(self._current_exposure - self._last_exposure)
        needs_rebalance = changed or exposure_shift > 0.20

        self._held = target

        if needs_rebalance:
            self._last_exposure = self._current_exposure
            return self._weighted(target, close_prices, portfolio)
        else:
            # Hold current weights — no trading
            current_w = portfolio.weights(close_prices)
            return current_w if current_w else self._weighted(target, close_prices, portfolio)

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
            if sector_count.get(sector, 0) < self.max_per_sector:
                return row.to_dict()
        return None

    def _weighted(self, tickers, close_prices=None, portfolio=None):
        """Equal weight scaled by current exposure cap."""
        if not tickers:
            return {}
        base_w = 1.0 / max(len(tickers), 1)
        w = base_w * self._current_exposure
        return {t: w for t in tickers}


# ── Runner ────────────────────────────────────────────────────────────────────

def run(
    n_positions:      int   = 25,
    switch_threshold: float = 0.10,
    min_hold_days:    int   = 10,
    max_sector_pct:   float = 0.30,
    min_score:        float = 0.0,
    position_stop:    float = 0.0,
    dd_warn:          float = -1.0,
    dd_danger:        float = -1.0,
    dd_stop:          float = -1.0,
    use_regime:       bool  = False,
    start:            str   = "2022-02-01",
    end:              str   = "2026-04-22",
    save_csv:         bool  = True,
):
    gen = SignalGenerator()
    scores_df = gen.precompute(start="2022-01-01", end=end, cache_path=CACHE_PATH)

    if use_regime:
        regime = RegimeDetector()
    else:
        # Dummy regime: always 100% exposure
        regime = RegimeDetector.__new__(RegimeDetector)
        regime.get_regime = lambda date: {"exposure": 1.0, "regime": "always_on"}

    strategy = StrategyV3(
        scores_df,
        regime,
        n_positions=n_positions,
        switch_threshold=switch_threshold,
        min_hold_days=min_hold_days,
        max_sector_pct=max_sector_pct,
        min_score=min_score,
        position_stop=position_stop,
        dd_warn=dd_warn,
        dd_danger=dd_danger,
        dd_stop=dd_stop,
    )

    engine = BacktestEngine(rebalance_days=1)
    print(f"\nStrategy v3: N={n_positions}  threshold={switch_threshold}"
          f"  min_hold={min_hold_days}d  max_sector={max_sector_pct:.0%}")
    overlays = []
    if min_score > 0:      overlays.append(f"min_score={min_score}")
    if position_stop < 0:  overlays.append(f"pos_stop={position_stop:.0%}")
    if dd_warn > -1:       overlays.append(f"CB warn/danger/stop={dd_warn:.0%}/{dd_danger:.0%}/{dd_stop:.0%}")
    if overlays:
        print(f"  Overlays: {', '.join(overlays)}")
    else:
        print(f"  Overlays: none (diversification only)")
    results = engine.run(strategy, start=start, end=end)

    if save_csv:
        results.analyzer.save_nav_csv("backtest_nav_v3.csv")

    # Trade summary
    tlog = results.portfolio.trade_log
    buys = [t for t in tlog if t.side == "BUY"]
    sells = [t for t in tlog if "SELL" in t.side]
    total_tc = sum(t.cost for t in tlog)
    buy_dates = pd.Series([t.date for t in buys]).nunique() if buys else 0

    print(f"\nTrade Summary:")
    print(f"  Total trades    : {len(tlog):,}  ({len(buys):,} buys, {len(sells):,} sells)")
    print(f"  Days with trades: {buy_dates}")
    print(f"  Total tx costs  : ${total_tc:,.0f}  ({total_tc/results.initial_cash:.2%} of capital)")

    # Regime breakdown
    _print_regime_breakdown(strategy, results)

    return results


def _print_regime_breakdown(strategy: StrategyV3, results):
    """Show how much time was spent in each regime."""
    nav = results.analyzer.nav
    dates = nav.index
    regime_log = []
    for d in dates:
        info = strategy.regime.get_regime(d)
        regime_log.append(info.get("regime", "unknown"))

    regime_series = pd.Series(regime_log, index=dates)
    counts = regime_series.value_counts()
    total = len(regime_series)

    print(f"\nRegime Distribution:")
    for regime, count in counts.items():
        pct = count / total * 100
        print(f"  {regime:<15} {count:>4}d  ({pct:.1f}%)")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy v3: regime + drawdown control")
    parser.add_argument("--n",          type=int,   default=25)
    parser.add_argument("--threshold",  type=float, default=0.10)
    parser.add_argument("--min-hold",   type=int,   default=10)
    parser.add_argument("--max-sector", type=float, default=0.30)
    parser.add_argument("--min-score",  type=float, default=0.0)
    parser.add_argument("--pos-stop",   type=float, default=0.0)
    parser.add_argument("--dd-warn",    type=float, default=-1.0)
    parser.add_argument("--dd-danger",  type=float, default=-1.0)
    parser.add_argument("--dd-stop",    type=float, default=-1.0)
    parser.add_argument("--start",      type=str,   default="2022-02-01")
    parser.add_argument("--end",        type=str,   default="2026-04-22")
    args = parser.parse_args()

    run(
        n_positions=args.n,
        switch_threshold=args.threshold,
        min_hold_days=args.min_hold,
        max_sector_pct=args.max_sector,
        min_score=args.min_score,
        position_stop=args.pos_stop,
        dd_warn=args.dd_warn,
        dd_danger=args.dd_danger,
        dd_stop=args.dd_stop,
        start=args.start,
        end=args.end,
    )
