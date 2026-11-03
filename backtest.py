"""
Backtesting Engine — Phase 1

Simulates a long-only rotation strategy over the test period.

Design:
  - Signal generated at day T close (strategy callback)
  - Trades filled at day T+1 open  (no look-ahead)
  - NAV marked at day T close
  - Transaction cost: 10 bps per side (20 bps round trip)

Classes:
  Portfolio           — position/cash/NAV state
  BacktestEngine      — event loop over trading days
  PerformanceAnalyzer — metrics + text tearsheet

Strategy interface:
  fn(date, available_df, portfolio) -> Dict[str, float]
    available_df  : DataFrame with columns [ticker, Open, High, Low, Close, Volume]
                    for every ticker active on `date` (uses Close prices)
    portfolio     : current Portfolio state (read-only in strategy)
    returns       : {ticker: weight}  weights sum <= 1.0, rest stays in cash

Usage:
  from backtest import BacktestEngine, PerformanceAnalyzer

  def my_strategy(date, available_df, portfolio):
      tickers = available_df['ticker'].tolist()
      w = 1.0 / len(tickers)
      return {t: w for t in tickers}

  engine = BacktestEngine()
  results = engine.run(my_strategy, start="2022-02-01", end="2026-04-22")
  PerformanceAnalyzer(results).print_tearsheet()
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

PRICES_PATH    = "data/sp500_clean_ohlcv.parquet"
SPY_PATH       = "data/spy_daily.csv"
TRANSACTION_BP = 10   # bps per side


# ── Portfolio ─────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    date:   pd.Timestamp
    ticker: str
    shares: float
    price:  float
    side:   str   # "BUY" | "SELL"
    cost:   float # transaction cost paid ($)


class Portfolio:
    """Tracks cash, positions, and NAV over time."""

    def __init__(self, initial_cash: float = 1_000_000.0):
        self.initial_cash = initial_cash
        self.cash         = initial_cash
        self.positions: Dict[str, float] = {}   # ticker -> shares
        self.nav_series:  List[Tuple]    = []   # (date, nav)
        self.trade_log:   List[Trade]    = []

    # ── NAV ────────────────────────────────────────────────────────────────────
    def nav(self, close_prices: Dict[str, float]) -> float:
        equity = sum(
            shares * close_prices.get(ticker, 0.0)
            for ticker, shares in self.positions.items()
            if shares > 0
        )
        return self.cash + equity

    def record_nav(self, date: pd.Timestamp, close_prices: Dict[str, float]):
        self.nav_series.append((date, self.nav(close_prices)))

    # ── Weights ────────────────────────────────────────────────────────────────
    def weights(self, close_prices: Dict[str, float]) -> Dict[str, float]:
        n = self.nav(close_prices)
        if n <= 0:
            return {}
        return {
            t: (s * close_prices.get(t, 0.0)) / n
            for t, s in self.positions.items()
            if s > 0 and close_prices.get(t, 0.0) > 0
        }

    # ── Rebalance ──────────────────────────────────────────────────────────────
    def rebalance(
        self,
        date:            pd.Timestamp,
        target_weights:  Dict[str, float],
        open_prices:     Dict[str, float],
        cost_bps:        float = TRANSACTION_BP,
    ):
        """Execute trades to reach target_weights. Fills at open_prices."""
        cost_rate = cost_bps / 10_000.0
        current_nav = self.nav(open_prices)
        if current_nav <= 0:
            return

        # Target dollar values
        target_dollars = {t: w * current_nav for t, w in target_weights.items()
                          if open_prices.get(t, 0) > 0}

        # Current dollar values at open
        current_dollars = {t: s * open_prices.get(t, 0.0)
                           for t, s in self.positions.items()
                           if open_prices.get(t, 0.0) > 0}

        # Compute share deltas
        all_tickers = set(target_dollars) | set(current_dollars)
        deltas: Dict[str, float] = {}
        for ticker in all_tickers:
            price = open_prices.get(ticker, 0.0)
            if price <= 0:
                continue
            diff = target_dollars.get(ticker, 0.0) - current_dollars.get(ticker, 0.0)
            if abs(diff) >= price * 0.01:  # ignore sub-cent-share trades
                deltas[ticker] = diff / price

        # Execute sells first (free up cash), then buys
        for ticker, delta in sorted(deltas.items(), key=lambda x: x[1]):
            price = open_prices[ticker]
            tc    = abs(delta) * price * cost_rate

            if delta < 0:  # SELL
                max_sell = self.positions.get(ticker, 0.0)
                sell_shares = min(-delta, max_sell)
                if sell_shares > 0:
                    self.positions[ticker] = self.positions.get(ticker, 0.0) - sell_shares
                    self.cash += sell_shares * price - tc
                    self.trade_log.append(Trade(date, ticker, sell_shares, price, "SELL", tc))
            else:           # BUY
                max_affordable = max(0.0, (self.cash - tc) / price)
                buy_shares = min(delta, max_affordable)
                if buy_shares > 1e-6:
                    self.positions[ticker] = self.positions.get(ticker, 0.0) + buy_shares
                    self.cash -= buy_shares * price + tc
                    self.trade_log.append(Trade(date, ticker, buy_shares, price, "BUY", tc))

        # Liquidate any tickers dropped from target (not covered above if they
        # had no open price today — sell at last known price if available)
        for ticker in list(self.positions):
            if self.positions[ticker] < 1e-6:
                del self.positions[ticker]

    def liquidate_delisted(
        self,
        date:    pd.Timestamp,
        delisted: List[str],
        prices:  Dict[str, float],
        cost_bps: float = TRANSACTION_BP,
    ):
        """Force-sell tickers that are no longer in the universe."""
        cost_rate = cost_bps / 10_000.0
        for ticker in delisted:
            shares = self.positions.get(ticker, 0.0)
            price  = prices.get(ticker, 0.0)
            if shares > 0 and price > 0:
                tc = shares * price * cost_rate
                self.cash += shares * price - tc
                self.trade_log.append(Trade(date, ticker, shares, price, "SELL_DELIST", tc))
                del self.positions[ticker]


# ── Engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-driven backtesting loop.

    Strategy is called every `rebalance_days` trading days.
    Trades fill at next-day open; NAV tracked daily at close.
    """

    def __init__(
        self,
        prices_path:     str   = PRICES_PATH,
        spy_path:        str   = SPY_PATH,
        initial_cash:    float = 1_000_000.0,
        rebalance_days:  int   = 10,
        transaction_bps: float = TRANSACTION_BP,
    ):
        self.initial_cash    = initial_cash
        self.rebalance_days  = rebalance_days
        self.transaction_bps = transaction_bps

        print("Loading price data...")
        raw = pd.read_parquet(prices_path, columns=["ticker","date","Open","High","Low","Close","Volume"])
        raw["date"] = pd.to_datetime(raw["date"])
        self.prices = raw

        spy_raw = pd.read_csv(spy_path, parse_dates=["Date"])
        spy_raw = spy_raw.rename(columns={"Date": "date", "Adj Close": "spy_close"})
        spy_raw = spy_raw[["date", "spy_close"]].sort_values("date").reset_index(drop=True)
        self.spy = spy_raw.set_index("date")["spy_close"]

        # Build fast price lookups: date -> {ticker -> price}
        self._build_price_tables()
        print(f"  Price data: {len(self.prices):,} rows, "
              f"{self.prices['ticker'].nunique()} tickers, "
              f"{self.prices['date'].min().date()} -> {self.prices['date'].max().date()}")

    def _build_price_tables(self):
        """Pivot prices into date-keyed dicts for O(1) lookup during simulation."""
        p = self.prices
        self._close = p.pivot_table(index="date", columns="ticker", values="Close")
        self._open  = p.pivot_table(index="date", columns="ticker", values="Open")
        self._all_dates = sorted(self._close.index)
        # Pre-build per-date OHLCV DataFrames so strategy calls are O(1) not O(n)
        self._daily_df = {
            date: grp[["ticker", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)
            for date, grp in p.groupby("date")
        }

    def _get_prices(self, table: pd.DataFrame, date: pd.Timestamp) -> Dict[str, float]:
        """Return {ticker: price} for a given date, dropping NaN."""
        if date not in table.index:
            return {}
        row = table.loc[date]
        return row.dropna().to_dict()

    def run(
        self,
        strategy:   Callable,
        start:      str = "2022-02-01",
        end:        str = "2026-04-22",
        verbose:    bool = True,
    ) -> "BacktestResults":
        """Run the backtest and return a BacktestResults object."""
        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)

        trading_days = [d for d in self._all_dates if start_dt <= d <= end_dt]
        if not trading_days:
            raise ValueError(f"No trading days found in [{start}, {end}]")

        portfolio = Portfolio(self.initial_cash)
        pending_weights: Optional[Dict[str, float]] = None  # to fill next open
        days_since_rebalance = self.rebalance_days  # trigger on day 1

        if verbose:
            print(f"\nBacktest: {start} -> {end}  ({len(trading_days)} trading days)")

        prev_universe: set = set()

        for i, date in enumerate(trading_days):
            close_prices = self._get_prices(self._close, date)
            open_prices  = self._get_prices(self._open,  date)

            # --- Execute pending trades at today's open ---
            if pending_weights is not None:
                # Liquidate delisted tickers (in portfolio but no open price today)
                delisted = [t for t in list(portfolio.positions)
                            if t not in open_prices and t not in close_prices]
                if delisted:
                    portfolio.liquidate_delisted(date, delisted, open_prices, self.transaction_bps)

                portfolio.rebalance(date, pending_weights, open_prices, self.transaction_bps)
                pending_weights = None

            # --- Record NAV at today's close ---
            portfolio.record_nav(date, close_prices)

            # --- Decide whether to rebalance ---
            days_since_rebalance += 1
            if days_since_rebalance >= self.rebalance_days:
                # Build available_df for strategy (uses today's close, signal for next open fill)
                ohlcv_today = self._daily_df.get(date, pd.DataFrame(
                    columns=["ticker", "Open", "High", "Low", "Close", "Volume"]))

                # Call strategy — returns target weights
                target_weights = strategy(date, ohlcv_today, portfolio)

                # Validate weights
                total = sum(target_weights.values())
                if total > 1.001:
                    target_weights = {t: w / total for t, w in target_weights.items()}

                pending_weights = target_weights
                days_since_rebalance = 0

            prev_universe = set(close_prices.keys())

        # Final liquidation at last close
        last_close = self._get_prices(self._close, trading_days[-1])
        portfolio.record_nav(trading_days[-1], last_close)

        nav_series = pd.Series(
            {d: v for d, v in portfolio.nav_series},
            name="portfolio"
        )
        nav_series.index = pd.DatetimeIndex(nav_series.index)
        nav_series = nav_series[~nav_series.index.duplicated(keep="last")]

        spy_slice = self.spy.reindex(nav_series.index, method="ffill").dropna()

        if verbose:
            analyzer = PerformanceAnalyzer(nav_series, spy_slice, self.initial_cash)
            analyzer.print_tearsheet()

        return BacktestResults(nav_series, spy_slice, portfolio, self.initial_cash)


# ── Results & Analytics ───────────────────────────────────────────────────────

@dataclass
class BacktestResults:
    nav:          pd.Series
    spy:          pd.Series
    portfolio:    Portfolio
    initial_cash: float

    @property
    def analyzer(self) -> "PerformanceAnalyzer":
        return PerformanceAnalyzer(self.nav, self.spy, self.initial_cash)


class PerformanceAnalyzer:
    """Computes performance metrics and prints a text tearsheet."""

    def __init__(
        self,
        nav:          pd.Series,
        benchmark:    Optional[pd.Series] = None,
        initial_cash: float = 1_000_000.0,
        risk_free:    float = 0.045,   # approx 10y avg fed funds
    ):
        self.nav          = nav.dropna()
        self.benchmark    = benchmark
        self.initial_cash = initial_cash
        self.rf           = risk_free

        self.returns   = self.nav.pct_change().dropna()
        self.n_years   = len(self.nav) / 252.0

    # ── Core Metrics ───────────────────────────────────────────────────────────
    def total_return(self) -> float:
        return (self.nav.iloc[-1] / self.nav.iloc[0]) - 1.0

    def annualized_return(self) -> float:
        if self.n_years <= 0:
            return 0.0
        return (1 + self.total_return()) ** (1.0 / self.n_years) - 1.0

    def max_drawdown(self) -> float:
        peak = self.nav.expanding().max()
        dd   = (self.nav - peak) / peak
        return dd.min()

    def drawdown_series(self) -> pd.Series:
        peak = self.nav.expanding().max()
        return (self.nav - peak) / peak

    def volatility(self) -> float:
        return self.returns.std() * np.sqrt(252)

    def sharpe(self) -> float:
        excess = self.returns.mean() * 252 - self.rf
        vol    = self.volatility()
        return excess / vol if vol > 0 else 0.0

    def sortino(self) -> float:
        downside = self.returns[self.returns < 0].std() * np.sqrt(252)
        excess   = self.returns.mean() * 252 - self.rf
        return excess / downside if downside > 0 else 0.0

    def calmar(self) -> float:
        mdd = self.max_drawdown()
        return self.annualized_return() / abs(mdd) if mdd != 0 else 0.0

    def win_rate_monthly(self) -> float:
        monthly = self.nav.resample("ME").last().pct_change().dropna()
        return (monthly > 0).mean()

    def alpha_beta(self) -> Tuple[float, float]:
        """CAPM alpha and beta vs benchmark (annualized alpha)."""
        if self.benchmark is None:
            return (0.0, 0.0)
        aligned = pd.concat([self.returns,
                              self.benchmark.pct_change().dropna()],
                             axis=1, join="inner").dropna()
        if len(aligned) < 30:
            return (0.0, 0.0)
        port_r, bench_r = aligned.iloc[:, 0], aligned.iloc[:, 1]
        beta  = port_r.cov(bench_r) / bench_r.var()
        alpha = (port_r.mean() - beta * bench_r.mean()) * 252
        return (alpha, beta)

    def best_month(self) -> float:
        return self.nav.resample("ME").last().pct_change().dropna().max()

    def worst_month(self) -> float:
        return self.nav.resample("ME").last().pct_change().dropna().min()

    def longest_drawdown_days(self) -> int:
        """Longest number of calendar days spent in a drawdown."""
        dd = self.drawdown_series()
        in_dd = (dd < 0)
        max_streak = 0
        streak     = 0
        for v in in_dd:
            if v:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    # ── Benchmark Metrics ──────────────────────────────────────────────────────
    def _bench_analyzer(self) -> Optional["PerformanceAnalyzer"]:
        if self.benchmark is None:
            return None
        return PerformanceAnalyzer(
            self.benchmark / self.benchmark.iloc[0] * self.initial_cash,
            risk_free=self.rf
        )

    # ── Tearsheet ──────────────────────────────────────────────────────────────
    def summary_dict(self) -> dict:
        alpha, beta = self.alpha_beta()
        return {
            "Total Return":     self.total_return(),
            "Ann. Return":      self.annualized_return(),
            "Ann. Volatility":  self.volatility(),
            "Max Drawdown":     self.max_drawdown(),
            "Sharpe Ratio":     self.sharpe(),
            "Sortino Ratio":    self.sortino(),
            "Calmar Ratio":     self.calmar(),
            "Win Rate (Mo.)":   self.win_rate_monthly(),
            "Best Month":       self.best_month(),
            "Worst Month":      self.worst_month(),
            "Alpha (ann.)":     alpha,
            "Beta":             beta,
            "Max DD Duration":  self.longest_drawdown_days(),
        }

    def print_tearsheet(self):
        bench = self._bench_analyzer()
        alpha, beta = self.alpha_beta()
        start = self.nav.index[0].date()
        end   = self.nav.index[-1].date()

        print()
        print("=" * 52)
        print(f"  Performance Tearsheet  ({start} -> {end})")
        print("=" * 52)
        print(f"  {'Metric':<26} {'Strategy':>10}  {'SPY':>10}")
        print("-" * 52)

        metrics = [
            ("Total Return",    f"{self.total_return():>9.2%}",
             f"{bench.total_return():>9.2%}" if bench else ""),
            ("Ann. Return",     f"{self.annualized_return():>9.2%}",
             f"{bench.annualized_return():>9.2%}" if bench else ""),
            ("Ann. Volatility", f"{self.volatility():>9.2%}",
             f"{bench.volatility():>9.2%}" if bench else ""),
            ("Max Drawdown",    f"{self.max_drawdown():>9.2%}",
             f"{bench.max_drawdown():>9.2%}" if bench else ""),
            ("Max DD Duration", f"{self.longest_drawdown_days():>9d}d",
             f"{bench.longest_drawdown_days():>9d}d" if bench else ""),
            ("Sharpe Ratio",    f"{self.sharpe():>9.3f}",
             f"{bench.sharpe():>9.3f}" if bench else ""),
            ("Sortino Ratio",   f"{self.sortino():>9.3f}",
             f"{bench.sortino():>9.3f}" if bench else ""),
            ("Calmar Ratio",    f"{self.calmar():>9.3f}",
             f"{bench.calmar():>9.3f}" if bench else ""),
            ("Win Rate (Mo.)",  f"{self.win_rate_monthly():>9.1%}",
             f"{bench.win_rate_monthly():>9.1%}" if bench else ""),
            ("Best Month",      f"{self.best_month():>9.2%}",
             f"{bench.best_month():>9.2%}" if bench else ""),
            ("Worst Month",     f"{self.worst_month():>9.2%}",
             f"{bench.worst_month():>9.2%}" if bench else ""),
            ("Alpha (ann.)",    f"{alpha:>+9.2%}", ""),
            ("Beta",            f"{beta:>9.3f}",   ""),
        ]

        for name, strat, spy_val in metrics:
            print(f"  {name:<26} {strat}  {spy_val}")

        print("=" * 52)
        print()

    @property
    def portfolio(self):
        # Injected by BacktestResults for tearsheet access
        return None

    def save_nav_csv(self, path: str = "backtest_nav.csv"):
        df = pd.DataFrame({"nav": self.nav})
        if self.benchmark is not None:
            bench_scaled = self.benchmark / self.benchmark.iloc[0] * self.initial_cash
            df["spy"] = bench_scaled
        df.to_csv(path)
        print(f"NAV saved to {path}")

    def monthly_returns_table(self) -> pd.DataFrame:
        monthly = self.nav.resample("ME").last().pct_change().dropna()
        monthly.index = pd.MultiIndex.from_arrays(
            [monthly.index.year, monthly.index.month_name().str[:3]],
            names=["Year", "Month"]
        )
        return monthly.unstack("Month")


# ── Built-in Strategies (for testing) ────────────────────────────────────────

def strategy_equal_weight(date, available_df, portfolio) -> Dict[str, float]:
    """Sanity check: equal-weight all available stocks every rebalance."""
    tickers = available_df["ticker"].tolist()
    if not tickers:
        return {}
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def strategy_buy_hold_spy(date, available_df, portfolio) -> Dict[str, float]:
    """
    Buy and hold a single ticker (SPY proxy not in universe — picks first ticker).
    Use as a debugging sanity check only.
    """
    tickers = available_df["ticker"].tolist()
    return {tickers[0]: 1.0} if tickers else {}


def strategy_cash(date, available_df, portfolio) -> Dict[str, float]:
    """Always holds 100% cash. Benchmark for 'do nothing'."""
    return {}


# ── Main: sanity-check run ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    strategy_name = sys.argv[1] if len(sys.argv) > 1 else "equal_weight"

    strategies = {
        "equal_weight": strategy_equal_weight,
        "cash":         strategy_cash,
    }
    strategy_fn = strategies.get(strategy_name, strategy_equal_weight)
    print(f"Running strategy: {strategy_name}")

    engine = BacktestEngine(rebalance_days=10)
    results = engine.run(strategy_fn, start="2022-02-01", end="2026-04-22")

    # Save NAV
    results.analyzer.save_nav_csv("backtest_nav.csv")

    # Trade summary
    tlog = results.portfolio.trade_log
    total_trades = len(tlog)
    total_tc     = sum(t.cost for t in tlog)
    buys  = sum(1 for t in tlog if t.side == "BUY")
    sells = sum(1 for t in tlog if "SELL" in t.side)
    print(f"\nTrade Summary:")
    print(f"  Total trades  : {total_trades:,}  ({buys:,} buys, {sells:,} sells)")
    print(f"  Total costs   : ${total_tc:,.0f}  ({total_tc/results.initial_cash:.2%} of capital)")
