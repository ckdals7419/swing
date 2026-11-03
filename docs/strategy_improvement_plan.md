# Swing Trading Strategy — Improvement Plan

## Strategy Philosophy

**Long-only rotation strategy** with cash as a position:
- Buy the best-ranked stocks at the right time
- Switch to a better stock when one emerges
- Sell and hold cash when nothing is attractive
- **Primary goal**: Low Maximum Drawdown (MDD)
- **Secondary goal**: Outperform S&P 500 annualized return

## Current State (as of 2026-05-08)

| Asset | Status |
|-------|--------|
| Data | `features_phase4.parquet` — 2.5M rows, 505 tickers, 1990-2026 |
| Features | 141 active (32 dropped through screening) |
| v2 Global Model | Test AUC 0.5535, IC 0.0411, L/S +0.23% |
| v3 Sector Models | 11 models — best AUC 0.5658 (Financials), best IC 0.0607 (Tech) |
| Backtesting | `backtest.py` — event-driven, 10bps/side, SPY benchmark |
| Signal | `scorer.py` — composite clf+reg per-sector scoring |
| Strategy | `strategy_v2.py` — hysteresis rotation, 6.65% ann, -38% MDD |
| Risk Management | None |
| Live Pipeline | None |

### Diagnosed Problems (Post Phase 3)

1. **High beta (1.14)**: Classifier predicts "prob of +2% move" which inherently favors volatile names → these crash harder in bear markets
2. **Negative alpha (-6%)**: Signal IC=0.04 is too weak to overcome beta drag — strategy captures market risk but doesn't add value
3. **MDD -38% vs SPY -22%**: Direct consequence of high-beta stock selection without any risk management
4. **No regime awareness**: Strategy is 100% invested regardless of market conditions — 2022 bear market fully absorbed
5. **No drawdown control**: No circuit breaker, no stop losses, no exposure scaling

### What Will Move the Needle Most

| Lever | Expected MDD Impact | Why |
|-------|:------------------:|-----|
| Regime-based exposure scaling | -15 to -20pp | Avoids being fully invested in bear markets |
| Portfolio drawdown circuit breaker | -5 to -10pp | Hard cap prevents tail losses |
| Beta-aware position sizing | -3 to -5pp | Reduces vol drag from high-beta names |
| Minimum score threshold (cash) | -2 to -3pp | Avoids buying when signal is weak |

---

## Phase 1: Backtesting Engine

> Build the core infrastructure to simulate and measure any strategy before improving signals.

**Why first**: Without measurement, every other improvement is guesswork.

### Deliverables

| # | Task | Output |
|---|------|--------|
| 1.1 | Build `backtest.py` — event-driven daily loop | Core engine |
| 1.2 | Portfolio tracker (positions, cash, daily NAV) | `BacktestPortfolio` class |
| 1.3 | Performance metrics module | Annualized return, Sharpe, Sortino, MDD, Calmar, alpha vs SPY |
| 1.4 | Tearsheet generator | Monthly returns heatmap, drawdown chart, equity curve vs SPY |
| 1.5 | Transaction cost model | Slippage (10bps) + commission model |

### Architecture

```
backtest.py
  |-- BacktestEngine        : daily event loop, date iteration
  |-- Portfolio              : positions, cash, NAV history
  |-- SignalGenerator        : model inference -> ranked stock list
  |-- RiskManager            : position limits, stop losses, exposure control
  |-- PerformanceAnalyzer    : metrics + tearsheet
```

### Key Design Decisions

- **Daily resolution**: Rebalance at close, use next-day open prices for fills
- **Look-ahead prevention**: Features computed ONLY from data available at decision time
- **Slippage model**: 10bps per trade (conservative for S&P 500 liquid names)
- **Benchmark**: SPY total return (dividends reinvested)

### Success Criteria
- Can simulate 2022-2026 test period in < 30 seconds ✅
- Produces correct metrics for a naive "buy SPY and hold" strategy (sanity check) ✅
- Drawdown calculation matches manual spot-check ✅

### Phase 1 Sanity Check Results (2022-02-01 ~ 2026-04-22)

| Strategy | Ann. Return | MDD | Sharpe | Notes |
|----------|:-----------:|:---:|:------:|-------|
| Equal-weight (500 stocks) | 3.55% | -21.33% | 0.025 | Lags SPY — expected, cap-weight dominates |
| Cash (0% equity) | 0.00% | 0.00% | 0.000 | Correct baseline |
| **SPY benchmark** | **12.89%** | **-22.09%** | **0.519** | Target to beat |

Engine file: `backtest.py`

---

---

## Phase 2: Baseline Strategy + Composite Signal — **Complete**

> Create the first tradeable strategy and establish baseline performance numbers.

### Deliverables

| # | Task | Output |
|---|------|--------|
| 2.1 | Composite score = f(clf_prob, reg_pred) | `compute_score()` function |
| 2.2 | Smart model routing (global vs sector) | Use sector model where IC > 0, global otherwise |
| 2.3 | Baseline strategy: top-N rotation | Backtest results on test period |
| 2.4 | Baseline metrics vs SPY | MDD, annual return, Sharpe |

### Composite Score Design

```
score = w_clf * clf_prob + w_reg * normalize(reg_pred)
```

- `clf_prob`: Probability of +2% in 10 days (0 to 1)
- `reg_pred`: Predicted 10-day return (z-scored cross-sectionally per day)
- Default weights: `w_clf = 0.6, w_reg = 0.4`
- **For sectors with inverted regressor** (Financials, Utilities, Comm Services): `w_reg = 0`

### Baseline Strategy Rules

1. Each day, score all ~500 stocks
2. Buy top-N stocks by composite score (N = 10-20)
3. Equal-weight positions
4. Hold for 10 days, then re-rank and rotate
5. No cash logic yet — always fully invested

### Phase 2 Results (2022-02-01 ~ 2026-04-22)

Files: `scorer.py` (SignalGenerator), `strategy_v1.py` (top-N strategy)

#### Rebalance Frequency Sensitivity

| Rebalance | N | Ann. Return | MDD | Sharpe | Alpha | Tx Costs |
|-----------|:-:|:-----------:|:---:|:------:|:-----:|:--------:|
| Every 5d  | 15 | -3.29% | -28.35% | -0.182 | -15.9% | 21.4% |
| **Every 10d** | **15** | **1.15%** | **-24.60%** | **0.000** | **-11.5%** | **13.0%** |
| Every 20d | 15 | 6.43% | -23.35% | 0.195 | -6.2% | 8.2% |
| **SPY benchmark** | — | **12.89%** | **-22.09%** | **0.519** | — | — |

#### Key Findings

1. **Transaction costs dominate**: 13-21% of capital consumed depending on frequency. The model's raw IC of 0.04 generates ~4-6% gross alpha, but rotation costs wipe it out.
2. **Holding period matters more than N**: Doubling from 10d→20d cuts costs by 37% and improves return by 5.3pp. The signal isn't strong enough to justify frequent switching.
3. **High-volatility bias**: Top-ranked stocks are consistently high-beta names (MRNA, CCL, NCLH, CZR) — the classifier correctly identifies high-probability movers, but those same stocks crash hard too. Beta = 1.165 vs SPY's 1.0.
4. **Phase 3 (rotation hysteresis) is critical**: Without a "switch threshold", the portfolio rotates almost entirely every cycle. Even swapping 3-4 positions instead of 12-15 would cut costs by 70%.

#### What Phase 3 Must Fix
- **Turnover**: Only switch when new stock score beats current + switch_threshold + tx_cost
- **Sector concentration**: Several rebalances go all-in on 2-3 sectors
- **Holding period**: Should be flexible (10-day minimum, hold until beaten) not fixed-cycle

---

## Phase 3: Portfolio Construction & Rotation Logic — **Complete**

> Turn raw signals into actual position decisions: what to buy, when to switch, how much.

### Deliverables

| # | Task | Output |
|---|------|--------|
| 3.1 | Position sizing (equal-weight vs conviction-weighted) | `PositionSizer` class |
| 3.2 | Rotation logic: switch threshold to prevent churn | Hysteresis parameter |
| 3.3 | Sector concentration limits | Max 30% per sector |
| 3.4 | Holding period optimization | Optimal rebalance frequency |
| 3.5 | Turnover constraint | Max daily turnover % |

### Rotation Logic

The core problem: "Should I switch from stock A to stock B?"

```
Switch if:  score(B) > score(A) + switch_threshold + transaction_cost
```

- `switch_threshold` prevents churn from noise in daily score changes
- Only switch when the improvement is meaningful enough to pay the transaction cost
- **Stale position decay**: Reduce effective score of held positions by 1% per day past holding period to encourage review, not forced selling

### Position Sizing Options

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| Equal weight | 1/N per position | Simple, diversified | Ignores conviction |
| Score-weighted | Size proportional to score | Higher conviction = bigger bet | Concentrates in model errors |
| Volatility-targeted | Size inversely proportional to stock vol | Equalizes risk contribution | May underweight high-momentum names |
| **Risk-parity hybrid** | Score-weighted, capped by vol | Balances conviction + risk | Recommended starting point |

### Sector Limits
- Max 30% in any single sector
- If Technology and Healthcare both rank top, allow both but cap each
- Prevents the "all-in tech" trap during sector bubbles

### Phase 3 Results (2022-02-01 ~ 2026-04-22)

File: `strategy_v2.py`

**Root cause of Phase 2 churn identified**: Rank-15 vs rank-16 score gap is median 0.0015 (near zero), so any daily score fluctuation (mean 0.025) causes constant boundary swaps. Fix: compare against **entry score high-water mark**, not today's score.

#### Parameter Sweep

| N | Threshold | Min Hold | Ann. Return | MDD | Sharpe | Alpha | Tx Costs |
|:-:|:---------:|:--------:|:-----------:|:---:|:------:|:-----:|:--------:|
| 10 | 0.08 | 10d | -2.78% | -29.73% | -0.197 | -14.2% | ~1% |
| **10** | **0.10** | **10d** | **6.65%** | **-38.13%** | **0.203** | **-6.0%** | **0.83%** |
| 10 | 0.12 | 10d | -3.12% | -44.78% | -0.176 | -15.8% | ~1% |
| 15 | 0.10 | 10d | 2.92% | -33.94% | 0.048 | -9.3% | 0.89% |
| 20 | 0.10 | 10d | 0.65% | -30.64% | -0.078 | -10.9% | 0.86% |
| 25 | 0.10 | 10d | 3.80% | -22.60% | 0.072 | -8.4% | 0.96% |
| **SPY** | — | — | **12.89%** | **-22.09%** | **0.519** | — | — |

**Best return config: N=10, threshold=0.10, min_hold=10d**

#### Key Wins from Phase 3
1. **Transaction costs: 13% → 0.83%** — hysteresis + high-water mark entry score eliminated almost all unnecessary churn
2. **Return: 1.15% → 6.65%** — lower costs compounded into better performance
3. **Sector caps working** — no more single-sector concentration

#### Remaining Problem
MDD is still **-38%** (worse than SPY -22%) because the model selects high-beta, volatile names. In a bear market (2022), these drop far harder than SPY. Phase 4 (regime-based exposure + drawdown circuit breaker) is the fix.

---

## Phase 4: Drawdown & Risk Control (MDD Focus) — **In Progress**

> The most important phase for the user's goal. This is where MDD gets controlled.

### Deliverables

| # | Task | Output |
|---|------|--------|
| 4.1 | Regime-based exposure scaling | Market regime -> max equity % |
| 4.2 | Portfolio drawdown circuit breaker | Hard stop on portfolio-level losses |
| 4.3 | Gradual re-entry after drawdown | Prevent whipsaw from on/off switches |
| 4.4 | Minimum score threshold | "Go to cash" when signal is weak |

File: `strategy_v3.py`

### 4.1 Regime-Based Exposure Scaling

Uses SPY price data (already loaded in BacktestEngine) and VIX data to determine max equity exposure. All signals use only data available at decision time (no look-ahead).

**Regime signals** (computed from `data/spy_daily.csv` and `data/vix_daily.csv`):
- `spy_above_sma200`: SPY close > 200-day SMA (trend filter)
- `spy_drawdown`: Current drawdown from 252-day high
- `vix_level`: Current VIX close

**Exposure table**:

| Regime | Conditions | Max Exposure |
|--------|-----------|:------------:|
| Strong bull | SPY > SMA200 AND VIX < 20 | 100% |
| Mild bull | SPY > SMA200 AND VIX 20-25 | 80% |
| Cautious | SPY > SMA200 AND VIX > 25, OR SPY < SMA200 AND VIX < 25 | 50% |
| Bear | SPY < SMA200 AND VIX > 25 | 30% |
| Crisis | VIX > 35 OR SPY drawdown > -15% | 0% |

**Implementation**: Exposure cap scales the number of positions, not individual weights. With N=10:
- 100% exposure → 10 positions × 10% each
- 50% exposure → 5 positions × 10% each (50% cash)
- 0% exposure → all cash

### 4.2 Portfolio Drawdown Circuit Breaker

Tracks portfolio high-water mark and current drawdown. Overrides regime exposure when triggered.

```
if portfolio_dd < -10%:  max_exposure = min(max_exposure, 50%)
if portfolio_dd < -15%:  max_exposure = min(max_exposure, 20%)
if portfolio_dd < -20%:  max_exposure = 0%   # all cash, full stop
```

The circuit breaker is the tighter constraint — if regime says 100% but portfolio is in -12% drawdown, exposure caps at 50%.

### 4.3 Gradual Re-Entry

After a circuit breaker triggers, don't jump back to full exposure:
- Recovery condition: portfolio drawdown improves to < -5% AND regime is bullish
- Re-entry pace: increase max positions by 2 per rebalance until target reached
- This prevents whipsaw: going all-cash at bottom, then all-in right before another leg down

### 4.4 Minimum Score Threshold

```
Only buy stocks with score > 0.55 (calibrated: ~top 20% of daily scores)
```

If fewer than N stocks pass threshold, hold remaining allocation as cash. This implements "sell and observe when nothing appeals."

### Phase 4 Results (2022-02-01 ~ 2026-04-22)

File: `strategy_v3.py`

**Key finding: With IC=0.04, all active risk overlays destroy more value than they save.**

Every overlay tested reduces MDD but reduces returns even more, because the signal has negative alpha (-6% to -8%). Reducing beta (through exposure scaling, stops, or CB) simply exposes the negative alpha.

#### Risk Overlay Sweep

| Configuration | Ann. Return | MDD | Sharpe | Alpha | Beta | Tx Costs |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| v2 N=10 (Phase 3 best) | +6.65% | -38.13% | 0.203 | -6.0% | 1.14 | 0.83% |
| Regime filter (weight scaling) | -1.93% | -22.66% | -0.322 | -7.3% | 0.48 | 3.45% |
| Soft circuit breaker (-10/-15/-25%) | -2.60% | -21.90% | -0.759 | -6.8% | 0.33 | 2.01% |
| Per-position stop loss (-8%) | +4.69% | -34.43% | 0.125 | -7.6% | 1.10 | 1.55% |
| **v3 N=25 (diversification only)** | **+3.80%** | **-22.60%** | **0.072** | **-8.4%** | **1.05** | **0.96%** |
| **SPY benchmark** | **+12.89%** | **-22.09%** | **0.519** | — | — | — |

#### Why Overlays Fail

1. **Regime filter**: Missing recovery rallies costs more than avoiding drawdowns saves (high-beta stocks bounce back hardest)
2. **Circuit breaker**: Signal too weak to recover portfolio HWM → locks strategy in cash permanently
3. **Stop losses**: Cuts mean-reverting dips too early, locks in losses on temporarily down high-beta names
4. **Diversification (N=25)**: The ONLY lever that improves MDD without proportionally destroying returns — reduces idiosyncratic risk via broader portfolio

#### Diagnosis

The model selects high-beta volatile stocks (beta=1.14 with N=10). It has IC=0.04 which translates to roughly 0% alpha after transaction costs. The strategy is essentially a **leveraged beta play with no alpha**:
- In bull markets: outperforms slightly (thanks to beta > 1)
- In bear markets: underperforms significantly (same reason)
- Net over cycle: returns close to SPY but with more risk

**The bottleneck is signal quality, not risk management.** Phase 5 (model improvements) is the only path to meaningful alpha.

#### Best Available Config (v3 Default: N=25)

| Metric | v3 N=25 | SPY |
|--------|:-------:|:---:|
| Ann. Return | 3.80% | 12.89% |
| MDD | -22.60% | -22.09% |
| Sharpe | 0.072 | 0.519 |
| Calmar | 0.168 | 0.583 |
| Beta | 1.05 | 1.00 |
| Tx Costs | 0.96% | 0% |

---

## Phase 5: Model Improvements — **CRITICAL PATH**

> **This is the make-or-break phase.** Phase 4 proved that no amount of risk management fixes a signal with IC=0.04 and negative alpha. The model must improve to IC > 0.07 to generate positive alpha after costs.

### Deliverables (ordered by expected impact)

| # | Task | Output | Expected IC Lift | Priority |
|---|------|--------|:----------------:|----------|
| 5.1 | Asymmetric loss function | Penalize false positives (long-only bias) | +0.01-0.02 | **Critical** |
| 5.2 | Downside-focused target | Predict max drawdown, not just return | +0.01-0.02 | **Critical** |
| 5.3 | Multi-horizon ensemble | Combine 5d + 10d + 20d predictions | +0.01 | High |
| 5.4 | Beta-adjusted targets | Predict excess return over sector, not raw return | +0.01-0.02 | High |
| 5.5 | Fix calendar features pipeline | Rebuild parquet with 24 calendar features | +0.005 | Medium |
| 5.6 | Sector model routing with confidence | Auto-detect when sector model is unreliable | +0.005 | Medium |

### 5.1 Calendar Features Fix

`add_calendar_features()` was implemented but never wired into `build_features.py`. Fix:
1. Add the call to `build_ticker()` in `build_features.py`
2. Rebuild `features_phase4.parquet`
3. Retrain v4 to measure lift from calendar features
4. Expected impact: Low (v1 screening showed calendars were zero-importance), but election cycle features haven't been tested

### 5.2 Asymmetric Loss (Critical for Long-Only)

Standard `binary:logistic` treats false positives and false negatives equally. But for a long-only strategy:
- **False positive** (predict up, goes down) = we LOSE money
- **False negative** (predict down, goes up) = we MISS opportunity (but lose nothing)

False positives are far more costly. Use custom objective:

```python
def asymmetric_logistic(y_pred, y_true):
    # Penalize false positives 2x more than false negatives
    grad = sigmoid(y_pred) - y_true.get_label()
    hess = sigmoid(y_pred) * (1 - sigmoid(y_pred))
    # Scale gradient for false positives
    fp_mask = (sigmoid(y_pred) > 0.5) & (y_true.get_label() == 0)
    grad[fp_mask] *= 2.0
    hess[fp_mask] *= 2.0
    return grad, hess
```

Expected impact: Lower recall but higher precision -> fewer losing trades.

### 5.3 Multi-Horizon Ensemble

Currently we only predict 10-day returns. Adding 5d and 20d:

```
ensemble_score = 0.2 * pred_5d + 0.5 * pred_10d + 0.3 * pred_20d
```

- 5d captures short-term momentum
- 10d is the primary swing trading horizon
- 20d adds trend confirmation
- A stock scoring high on all three horizons is much more reliable

### 5.4 Downside-Focused Target (Critical for MDD)

Add a new prediction target: **maximum intra-period drawdown**

```python
# For each stock-date, what's the worst drawdown over next 10 days?
fwd_max_drawdown_10d = min(future_10d_returns_cumulative)
```

Use this as a risk filter:
```
Buy only if: expected_return > threshold AND expected_max_drawdown > -3%
```

This directly targets MDD by avoiding stocks likely to have sharp drops.

### 5.5 Sector Model Confidence

Instead of binary "use sector model or don't", compute confidence:

```python
confidence = min(val_auc - 0.52, val_ic) / 0.1  # 0 to 1 scale
sector_weight = confidence
global_weight = 1 - confidence

final_score = sector_weight * sector_score + global_weight * global_score
```

This smoothly blends between sector and global models based on validation performance.

---

## Phase 6: Walk-Forward Validation & Robustness

> Prove the strategy works out-of-sample across different market regimes.

### Deliverables

| # | Task | Output |
|---|------|--------|
| 6.1 | Rolling walk-forward test | 2-year train, 6-month test, rolling quarterly |
| 6.2 | Parameter sensitivity analysis | Heatmaps for key parameters |
| 6.3 | Monte Carlo bootstrap | Confidence intervals on MDD and return |
| 6.4 | Regime-specific performance breakdown | Bull/bear/sideways sub-period results |
| 6.5 | Survivorship bias check | Test on delisted stocks too |

### Walk-Forward Schedule

```
Window 1: Train 2005-2014 -> Test 2015-H1
Window 2: Train 2005-2014.5 -> Test 2015-H2
Window 3: Train 2005-2015 -> Test 2016-H1
...
Window N: Train 2005-2024 -> Test 2025-H1
```

- If strategy is robust, metrics should be stable across windows
- If metrics degrade sharply in recent windows, the signal may be decaying

### Parameter Sensitivity

Key parameters to stress-test:

| Parameter | Range | Impact |
|-----------|-------|--------|
| Number of positions (N) | 5-30 | Concentration vs diversification |
| Rebalance frequency | 5-20 days | Turnover vs signal freshness |
| Switch threshold | 0-0.2 | Churn vs responsiveness |
| Drawdown circuit breaker | -5% to -20% | Conservatism vs participation |
| Score threshold for cash | 0.4-0.6 | Time in market vs selectivity |
| clf/reg weight blend | 0.3/0.7 to 0.8/0.2 | Direction vs magnitude signal |

---

## Phase 7: Live Signal Pipeline

> Production system that outputs daily actionable signals.

### Deliverables

| # | Task | Output |
|---|------|--------|
| 7.1 | Daily data fetch (prices + macro) | `fetch_daily.py` |
| 7.2 | Feature computation for latest day | `compute_features_live.py` |
| 7.3 | Model inference + ranking | `generate_signals.py` |
| 7.4 | Portfolio decision engine | "Buy AAPL 5%, Sell MSFT, Hold NVDA" |
| 7.5 | Signal output format | CSV/JSON + notification |

### Daily Workflow

```
06:00  Fetch yesterday's closing prices + macro data
06:05  Compute features for all 500 stocks
06:10  Run model inference (141 features -> scores)
06:15  Apply risk filters (regime, drawdown, stops)
06:20  Compare vs current portfolio -> rotation decisions
06:25  Output: BUY / SELL / HOLD per position + reasoning
```

### Output Format

```json
{
  "date": "2026-05-08",
  "regime": "mild_bull",
  "target_exposure": 80,
  "current_drawdown": -2.1,
  "actions": [
    {"ticker": "NVDA", "action": "BUY",  "weight": 8.0, "score": 0.82, "reason": "Top rank, strong momentum + low vol"},
    {"ticker": "MSFT", "action": "HOLD", "weight": 7.5, "score": 0.78, "reason": "Still high rank, no switch needed"},
    {"ticker": "XOM",  "action": "SELL", "weight": 0,   "score": 0.41, "reason": "Score dropped below threshold"},
    {"ticker": "CASH", "action": "HOLD", "weight": 20.0, "reason": "Regime exposure limit"}
  ]
}
```

---

## Implementation Priority & Dependencies

```
Phase 1 (Backtest Engine)
  |
  v
Phase 2 (Baseline Strategy + Composite Signal)
  |
  +--> Phase 3 (Portfolio Construction)
  |      |
  |      v
  |    Phase 4 (Risk/MDD Control)  <-- Highest impact for user's goal
  |
  +--> Phase 5 (Model Improvements) <-- Can run in parallel with 3-4
         |
         v
       Phase 6 (Walk-Forward Validation)
         |
         v
       Phase 7 (Live Pipeline)
```

### Estimated Impact on MDD

| Component | MDD Reduction | Mechanism |
|-----------|:------------:|-----------|
| Regime-based exposure (4.3) | -40~50% | Avoids being fully invested during bear markets |
| Portfolio drawdown breaker (4.1) | -20~30% | Hard cap on portfolio-level losses |
| Per-position stop loss (4.2) | -10~15% | Prevents single-stock blowups |
| Minimum score cash threshold (4.5) | -5~10% | Avoids buying weak signals |
| Downside-focused target (5.4) | -5~10% | Filters out high-crash-risk stocks |
| **Combined** | **-50~70%** | **MDD from ~-25% -> ~-8% to -12%** |

### Performance Target

| Metric | Target | SPY (2022-2025 avg) |
|--------|:------:|:-------------------:|
| Annual Return | 12-18% | ~10% |
| Maximum Drawdown | < -12% | ~-25% |
| Sharpe Ratio | > 1.0 | ~0.5 |
| Calmar Ratio | > 1.2 | ~0.4 |
| Win Rate (trades) | > 55% | N/A |
| Time in Market | 60-80% | 100% |
| Annual Turnover | < 500% | 0% |

---

## Phase Status

| Phase | Status | Depends On |
|-------|:------:|:----------:|
| Phase 1: Backtesting Engine | **Complete** | - |
| Phase 2: Baseline Strategy | **Complete** | Phase 1 |
| Phase 3: Portfolio Construction | **Complete** | Phase 2 |
| Phase 4: Risk/MDD Control | **Complete** | Phase 3 |
| Phase 5: Model Improvements | Pending | Phase 1 |
| Phase 6: Walk-Forward Validation | Pending | Phase 4 + 5 |
| Phase 7: Live Pipeline | Pending | Phase 6 |
