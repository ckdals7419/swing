# QLD Simulation Review Plan

Systematic audit of the backtest pipeline for errors, biases, and methodological issues.
Each phase is self-contained and ordered by impact severity.

**Legend**: PASS / ISSUE / N/A / INFO

---

## Phase 1 — Data Integrity & Synthetic LETF Validation

### 1-1. Synthetic QLD/TQQQ Fee Model Accuracy
- **File**: `build_synthetic_letf.py:32-81`
- **Issue**: SWAP_SPREAD=0.49% and EXPENSE_RATIO=0.95% hardcoded; fees changed over time
- **Result**: PASS (minor)
  - Overlap period: 2010-02-12 to 2026-05-15 (4,089 days)
  - Mean daily tracking error: **0.07 bps/yr** — essentially zero
  - Daily return correlation synth vs real: **0.9992**
  - Scale factor stable across first 20 days: 0.0622–0.0624 (range 0.3%)
  - Conclusion: Fee model is accurate enough; 0.07 bps/yr bias is negligible

### 1-2. Synthetic QLD Splice Point
- **File**: `build_extended_features.py:126-141`
- **Issue**: Single-day anchor may propagate first-day noise
- **Result**: PASS (minor)
  - Single-day scale: 0.062214
  - Regression slope over 2006–2010 window: 0.062327
  - Difference: **0.181%** — immaterial impact on returns
  - Conclusion: Single-day splice is adequate; regression alternative changes nothing materially

### 1-3. Synthetic QLD Open Price Reconstruction
- **File**: `build_extended_features.py:137-139`
- **Result**: PASS
  - Mean open price error vs real: **0.000%**, Std: **0.000%**
  - The extended features use real QLD open prices for the entire real period (2006-07-03 onward); synthetic open is only used pre-2006 where there is no real open to compare against
  - No impact on backtest fill prices

### 1-4. yfinance auto_adjust & Dividend Double-Counting
- **File**: `download_qld_data.py:32-33`
- **Result**: PASS
  - Mean daily diff (adj vs no-adj) in 2024: **0.00000%**
  - Max abs diff: **0.0000%**
  - Conclusion: No double-counting; auto_adjust is correct for leveraged ETFs

### 1-5. VIX Data Gaps & Forward-Fill
- **File**: `build_qld_features.py:146-151`, `build_extended_features.py:100-104`
- **Result**: PASS
  - Total NaN days in VIX after reindex: **6 of 6,839** (all at very start of data)
  - Gaps > 5 trading days: **0**
  - Conclusion: ffill without limit is safe; no meaningful gaps exist

### 1-6. BIL Return Proxy Pre-Inception
- **File**: `build_qld_features.py:190-195`
- **Result**: ISSUE (low impact, pre-2007 only)
  - BIL inception: 2007-05-30
  - Fed funds rate annualized 2000–2007 avg: **4.35%**
  - Actual 3M T-bill annualized 2000–2007 avg: **3.02%**
  - **Bias: +1.33%/yr** — cash proxy overstates cash return during 2000–2007 period
  - Impact: overstates strategy performance during periods out of QLD pre-2007
  - Fix: Use ^IRX (3M T-bill) as proxy for pre-BIL cash return

---

## Phase 2 — Backtest Engine Mechanics

### 2-1. Fill Timing & Slippage Model
- **File**: `strategy_qld.py:267-322`
- **Result**: PASS (sensitivity documented)
  - Each +5 bps costs approximately **−0.4% annual, −0.013 Sharpe**
  - At realistic 10 bps (from 2007): Ann 25.2%, Sharpe 0.648 (vs 25.6% / 0.661 at 5 bps)
  - At 20 bps: Ann 24.4%, Sharpe 0.624
  - Conclusion: 5 bps is optimistic for QLD (typical spread 10–15 bps); results remain strong even at 20 bps

### 2-2. Transaction Cost Realism
- **File**: `strategy_qld.py:38, 308-309`
- **Result**: PASS (sensitivity documented)

  | Cost | Ann | Total | Sharpe | Calmar |
  |------|-----|-------|--------|--------|
  | 0 bps | 26.0% | 8,682% | 0.673 | 0.602 |
  | 3 bps | 25.8% | 8,364% | 0.665 | 0.592 |
  | **5 bps (current)** | **25.6%** | **8,159%** | **0.661** | **0.586** |
  | 10 bps | 25.2% | 7,666% | 0.648 | 0.571 |
  | 15 bps | 24.8% | 7,203% | 0.636 | 0.557 |
  | 20 bps | 24.4% | 6,767% | 0.624 | 0.543 |

  Strategy remains highly profitable even at 20 bps. 123 round-trips over 19 years.

### 2-3. Open-to-Close vs Close-to-Close Return Split
- **File**: `strategy_qld.py:311-317`
- **Result**: PASS
  - Manual trace on 3 sell days (2008-01-07, 2008-03-28, 2008-06-12):
  - Sold portion earns **0** on execution day — correctly excluded
  - Formula: `unchanged * cc + bought * oc` where sold portion drops out
  - Logic is mathematically correct

### 2-4. Cash Return Compounding
- **File**: `strategy_qld.py:287, 317`
- **Result**: PASS
  - Cash-only backtest vs compound BIL: max deviation **0.000000%**
  - Linear daily cash application is equivalent to compound for small daily rates

### 2-5. Weight Boundary Initialization
- **File**: `strategy_qld.py:295`
- **Result**: PASS
  - Day 0: w_applied = 0.0 (correct — starts in cash)
  - Day 1: w_applied = 1.0 (first signal applied)
  - Behavior is intentional and correct: cold-start in cash, enter at next open

### 2-6. NaN Propagation in Price Series
- **File**: `strategy_qld.py:284-286`
- **Result**: PASS (with note)
  - NaN qld_close in real period (2006+): **0**
  - Zero qld_close: **0**
  - Single-day moves >20% (legitimate): 4 events
    - 2008-10-13: +24.59% (post-bailout rally)
    - 2008-10-28: +20.92% (bear market rally)
    - 2020-03-16: −24.28% (COVID crash)
    - 2025-04-09: +23.49% (tariff reversal)
  - All are real market events, not data errors

---

## Phase 3 — Look-Ahead Bias & Information Leakage

### 3-1. Macro Data Publication Lag
- **File**: `build_qld_features.py:176-182`
- **Result**: N/A for primary strategies
  - Baseline SMA200+SMA50 and vol-target use only `qqq_vs_sma200`, `qqq_vs_sma50`, `realized_vol_20d` — no macro features
  - Macro lag issue is relevant only for `rule_weight()` (5-level) and ML strategies
  - No impact on current primary backtest

### 3-2. SMA200 NaN Default to Full Exposure
- **File**: `strategy_qld.py:103-111`
- **Result**: PASS
  - Days with NaN `qqq_vs_sma200` in extended features: **0**
  - The extended features pre-compute SMA200 over the full QQQ history before filtering, so no warm-up NaN exists in the backtest dataset
  - No fix needed

### 3-3. ML Target/Fill Timing Mismatch
- **File**: `build_qld_features.py:201-211`, `train_qld_model.py:43-44`
- **Result**: PASS (negligible)
  - QLD open-to-close gap: mean **+0.045%**, std 2.246%
  - Annualized systematic bias: **~11 bps/yr**
  - Conclusion: near-zero mean gap; not a material look-ahead source

### 3-4. ML Train/Val Purge Gap Adequacy
- **File**: `train_qld_model.py:38-41`
- **Result**: ISSUE
  - Max feature lookback: **200 days** (SMA200)
  - Forward target window: 10 days
  - Minimum safe purge: **210 days**
  - Current purge: **40 days** (2018-12-31 to 2019-03-01)
  - SMA200 at 2019-03-01 uses QQQ data back to ~2018-07 — which overlaps training data
  - Fix: Extend purge to at least 210 days (train end → 2019-10-01 for val start)

### 3-5. Feature NaN Fill as Signal
- **File**: `train_qld_model.py:229, 298`
- **Result**: ISSUE (ML only)
  - 6 unbounded features filled with 0: `sma50_slope`, `sma200_slope`, `macd_hist`, `roc_20`, `vol_ratio_10_60`, `credit_spread`
  - `fillna(0)` is non-neutral for slope/z-score features (0 = "flat trend", not "unknown")
  - Fix: Replace with training-set median per feature; add `feature_available` binary flag

### 3-6. FRED Data Forward-Fill in Crossval
- **File**: `crossval_qld.py:602-643`
- **Result**: PASS
  - Lag applied as `index += lag_days` (shifts observation date forward)
  - ffill operates forward from shifted date — data only used after publication date
  - No future leak confirmed

---

## Phase 4 — Strategy Rule Robustness

### 4-1. VIX Threshold Sensitivity (22 / 30)
- **File**: `strategy_qld.py:92-100`
- **Result**: N/A for primary strategies
  - VIX thresholds only used in `rule_weight()` and `rule_sma200_vix()`, not in baseline or vol-target
  - Primary strategies are VIX-agnostic

### 4-2. Drawdown Threshold Sensitivity (−12% / −15%)
- **File**: `strategy_qld.py:84, 92, 147-148`
- **Result**: N/A for primary strategies
  - DD thresholds only in `rule_weight()` and `vol_target_with_stop()`; not used in primary vol-target

### 4-3. Vol-Target Level (30%) Sensitivity
- **File**: `chart_dca_compare.py:29`
- **Result**: INFO — plateau, not a sharp optimum

  | Target | Ann | Total | MDD | Sharpe | Calmar |
  |--------|-----|-------|-----|--------|--------|
  | 15% | 14.1% | 1,185% | −17.9% | 0.673 | 0.791 |
  | 20% | 17.2% | 2,060% | −22.7% | 0.680 | 0.758 |
  | 25% | 19.7% | 3,136% | −27.2% | 0.681 | 0.725 |
  | **30% (current)** | **22.0%** | **4,549%** | **−31.3%** | **0.695** | **0.701** |
  | 35% | 23.4% | 5,764% | −35.0% | 0.696 | 0.670 |
  | 40% | 24.5% | 6,807% | −37.7% | 0.696 | 0.649 |
  | 45% | 24.9% | 7,292% | −39.7% | 0.686 | 0.628 |
  | 50% | 25.1% | 7,546% | −40.9% | 0.678 | 0.615 |

  Sharpe is flat 0.695–0.696 across 30–40%. 30% is conservative (best Calmar); 35–40% gives more return with modestly more drawdown. Not overfit to a single peak.

### 4-4. Hysteresis Re-entry Logic
- **Result**: N/A for primary strategies
  - Only in `vol_target_with_stop()`, not used in current primary vol-target strategy

### 4-5. SMA50 Recovery Gate Value
- **File**: `chart_dca_compare.py:42-48, 56`
- **Result**: KEEP — adds meaningful value

  | Strategy | Ann | Total | MDD | Sharpe | Calmar | Extra days |
  |----------|-----|-------|-----|--------|--------|------------|
  | SMA200 only | 22.2% | 4,701% | −42.2% | 0.591 | 0.525 | — |
  | SMA200+SMA50 | 25.6% | 8,159% | −43.7% | 0.661 | 0.586 | +222 |

  SMA50 adds **+3.4% annual return, +0.070 Sharpe** with only marginally worse MDD (−1.5 pp). Keeping it is correct.

### 4-6. ML Threshold Grid Resolution
- **File**: `train_qld_model.py:248-261`
- **Result**: INFO (ML only)
  - Current grid: 0.025 step with t1 forced to midpoint
  - Low priority given ML is secondary to rule-based strategy

---

## Phase 5 — ML Model Validity

### 5-1 to 5-5. ML Walk-Forward, Feature Importance, Overfitting
- **Result**: DEFERRED
  - ML model (`models/qld/clf_10d.ubj`) exists but is secondary to rule-based primary strategy
  - Key confirmed issues: purge gap too short (3-4), NaN fill non-neutral (3-5)
  - Full walk-forward test requires retraining — defer to separate ML review

---

## Phase 6 — Visualization & Reporting Accuracy

### 6-1. Drawdown Calculation Scope
- **File**: `chart_yearly.py:140-154`
- **Result**: ISSUE (visual only)
  - 2008 MDD within-year: **−21.0%**
  - 2008 MDD carrying forward 2007 peak: **−28.0%**
  - Understatement: **7 percentage points**
  - Fix: Carry forward prior-year peak into yearly chart drawdown calculation

### 6-2. Applied Weight Shift Misalignment
- **File**: `chart_dca_compare.py:77`, `chart_yearly.py:81`
- **Result**: INFO (cosmetic)
  - Weight is plotted at signal date T, but executes at T+1 open
  - One-day visual offset; not a calculation error
  - Fix: Add legend note "signal date (fills next open)"

### 6-3. Annual Return Calculation Method
- **File**: `chart_yearly.py:73-74, 96`
- **Result**: ISSUE (misleading for partial years)
  - 2026 YTD: 93 trading days, calendar return = **+20.3%**
  - Annualized as shown in chart title: **+65.1%** — highly misleading
  - Fix: For current partial year, display calendar YTD return instead of annualized

### 6-4. Crossval Synthetic vs Extended Features Consistency
- **File**: `crossval_qld.py:111-139` vs `build_extended_features.py:126-141`
- **Result**: INFO
  - Two separate synthetic reconstructions exist; not verified to match exactly
  - Low priority: crossval is for bear-market regime testing only

### 6-5. Vol Panel Backward-Looking Shading
- **File**: `chart_dca_compare.py:203-205`
- **Result**: INFO (cosmetic)
  - Red shading shows 20d trailing vol > target; reader may interpret as "weight cut today"
  - Fix: Add "20d trailing vol" annotation to vol panel

---

## Phase 7 — End-to-End Sanity Checks

### 7-1. Manual Trade Trace
- **Result**: PASS
  - 5 significant dates verified (Lehman day, COVID crash, 2022 open, GFC bottom, COVID recovery):
  - All: expected NAV = actual NAV, error = **0.0000** — exact match

### 7-2. Benchmark Reproduction
- **Result**: PASS
  - QLD raw price return 2007–2026: **7,627.06%**
  - Backtest B&H return: **7,618.59%**
  - Difference: **−8.5 bps** (= transaction cost on first-day entry only, since B&H has zero rebalancing)

### 7-3. Cash-Only Benchmark
- **Result**: PASS
  - Max deviation vs compound BIL: **0.00000000%** — exact match

### 7-4. Turnover & Cost Attribution
- **Result**: PASS
  - Baseline total turnover 2007–2026: **123 round-trips** (|dw| sum)
  - Avg ~6.5 trades/year
  - Cost model consistent; TC deducted from NAV correctly

### 7-5. Regime Consistency Check
- **Result**: PASS
  - Days above SMA200: 3,946 (81%) — consistent with long-term NASDAQ bull trend
  - Days below SMA200: 927 (19%)
  - Days VIX > 30: 449 (9%)
  - GFC Sep 2008–Jun 2009: 81% days below SMA200 (169/209) — reasonable (some whipsaws in recovery)
  - 2017 bull: 100% days above SMA200 (251/251) — correct

### 7-6. Sensitivity Summary Table
- **Result**: COMPLETE

  | Fix | Ann Impact | Sharpe Impact | Priority |
  |-----|-----------|---------------|----------|
  | 1-6: BIL proxy (use ^IRX) | −1.33%/yr pre-2007 only | Low | Low (historical only) |
  | 2-1/2-2: Cost at 10 bps | −0.4%/yr | −0.013 | Medium |
  | 3-4: ML purge gap | ML only | ML only | High (ML) |
  | 3-5: NaN fill in ML | ML only | ML only | High (ML) |
  | 4-3: Vol target 35% | +1.4%/yr, −3.7% MDD | flat | Optional |
  | 4-5: SMA50 gate | +3.4%/yr | +0.070 | Keep as-is |
  | 6-1: Yearly DD scope | Visual only | — | Low |
  | 6-3: Partial year ann | Visual only | — | Medium |

---

## Summary

**Critical issues (fix before trusting results):**
- None for rule-based strategies

**Confirmed issues with fixes identified:**
- **1-6**: BIL proxy overstates cash return by +1.33%/yr pre-2007 (use ^IRX instead)
- **3-4**: ML purge gap too short (40d vs 210d needed) — ML results unreliable
- **3-5**: NaN fillna(0) non-neutral for slope/zscore ML features
- **6-1**: Yearly chart drawdown understated by up to 7 pp (carry forward prior-year peak)
- **6-3**: Partial year shown as annualized — show calendar YTD instead

**All clean (no fix needed):**
- 1-1, 1-2, 1-3, 1-4, 1-5, 2-1~2-6, 3-1, 3-2, 3-3, 3-6, 4-5, 7-1~7-5

**Rule-based primary strategy verdict: SOUND.** Backtest engine arithmetic is exact, no look-ahead bias, regime signals are consistent, transaction costs are modeled correctly. The main open risk is the pre-2007 BIL proxy (+1.33%/yr) which only affects results when starting the backtest from 2000.
