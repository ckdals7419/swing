# QLD Dynamic Weight Strategy — Implementation Plan

## Objective

Build a model that dynamically adjusts allocation between **QLD** (ProShares Ultra QQQ, 2× leveraged NASDAQ-100) and **cash** on each rebalance day, targeting:

- **Higher total / annualized return** than QLD buy-and-hold
- **Lower maximum drawdown** than QLD buy-and-hold

QLD buy-and-hold delivers extreme returns in bull markets but suffers catastrophic drawdowns (−60 % COVID, −70 % 2022 bear) and ongoing volatility decay. The edge comes from **avoiding those drawdowns** — even a simple regime filter that sidesteps the worst months dramatically improves the risk/return profile.

---

## Why a Single-Asset Weight Model (Not Stock Rotation)

| Dimension | Stock rotation (Phases 1–4) | QLD weight adjustment |
|-----------|---------------------------|----------------------|
| Universe | ~500 stocks | 1 ETF |
| Signal task | Cross-sectional ranking (hard) | Time-series regime (easier) |
| Alpha source | Stock selection IC | Market timing — avoid drawdowns |
| Data requirement | Huge panel | Single time-series + macro |
| Overfitting risk | High (many tickers × features) | Lower (fewer degrees of freedom) |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Feature Engine                                 │
│  QQQ OHLCV + VIX + Macro  →  ~30 features      │
└──────────────────┬──────────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │  Regime Classifier  │
         │  (XGBoost / rules)  │
         │  → regime label     │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  Weight Mapper      │
         │  regime → QLD wt    │
         │  (0 % – 100 %)     │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  Backtest Engine    │
         │  Signal T close     │
         │  Fill T+1 open      │
         │  10 bps per side    │
         └────────────────────┘
```

### Signal is generated on QQQ (the 1× underlying)

QQQ has a cleaner price series — no leverage decay, no daily rebalance noise. All trend, momentum, and volatility features are computed on QQQ. The **weight** is then applied to QLD in the portfolio.

---

## Phases

### Phase A — Data Download

Download daily OHLCV + adjusted close for:

| Ticker | Purpose |
|--------|---------|
| **QLD** | Trade target (2× NASDAQ-100) |
| **QQQ** | Signal source (1× NASDAQ-100) |
| **TQQQ** | Optional comparison (3× NASDAQ-100) |
| **SPY** | Benchmark |
| **^VIX** | Volatility regime |
| **^VXN** | NASDAQ volatility (if available) |
| **SHY / BIL** | Cash proxy (short-term treasuries) — for interest-on-cash modeling |

Source: `yfinance`. Period: 2007-01-01 → present (QLD inception: 2006-06-19).

Also reuse existing macro data from `data/` (fed funds, yield curve, credit spreads, DXY, oil, gold) already downloaded in Phase 4.

**Output:** `data/qld_universe.parquet`

### Phase B — Feature Engineering (~30 features)

All features computed on **QQQ** daily data (not QLD) unless otherwise noted.

#### B1. Trend (8 features)
| Feature | Definition |
|---------|-----------|
| `qqq_vs_sma50` | QQQ close / SMA(50) − 1 |
| `qqq_vs_sma200` | QQQ close / SMA(200) − 1 |
| `sma50_slope` | 5-day rate of change of SMA(50) |
| `sma200_slope` | 20-day rate of change of SMA(200) |
| `sma_cross` | SMA(50) > SMA(200) → 1, else 0 |
| `price_vs_20d_high` | close / rolling-20d-high − 1 |
| `price_vs_60d_high` | close / rolling-60d-high − 1 |
| `adx_14` | ADX(14) — trend strength |

#### B2. Momentum (6 features)
| Feature | Definition |
|---------|-----------|
| `return_5d` | 5-day return |
| `return_20d` | 20-day return |
| `return_60d` | 60-day return |
| `rsi_14` | RSI(14) |
| `macd_hist` | MACD histogram |
| `roc_20` | Rate of change 20d |

#### B3. Volatility (7 features)
| Feature | Definition |
|---------|-----------|
| `realized_vol_10d` | 10-day annualized realized vol of QQQ |
| `realized_vol_20d` | 20-day annualized realized vol of QQQ |
| `vol_ratio_10_60` | realized_vol_10d / realized_vol_60d |
| `vix_close` | VIX daily close |
| `vix_sma20` | SMA(20) of VIX |
| `vix_percentile` | VIX percentile rank over trailing 252d |
| `vix_term_struct` | VIX vs VIX3M (or VIX 1-month vs 3-month) if available, else VIX change 5d |

#### B4. Drawdown (3 features)
| Feature | Definition |
|---------|-----------|
| `qqq_drawdown` | QQQ drawdown from trailing 252d high |
| `qld_drawdown` | QLD drawdown from trailing 252d high |
| `days_in_drawdown` | Trading days since QQQ last all-time high |

#### B5. Breadth / Macro (6 features)
| Feature | Definition |
|---------|-----------|
| `yield_curve_10y2y` | 10Y − 2Y treasury spread |
| `fed_funds_rate` | Fed funds rate |
| `credit_spread` | HY credit spread (from existing macro data) |
| `dxy_change_20d` | 20d change in DXY |
| `spy_vs_qqq_20d` | SPY 20d return − QQQ 20d return (rotation signal) |
| `gold_change_20d` | 20d change in gold (risk-off proxy) |

**Output:** `data/qld_features.parquet`

### Phase C — Target Definition

Two possible targets (try both, pick the one that works better):

#### C1. Binary: "Good period to hold QLD"
```
fwd_return_10d = QLD return over next 10 trading days
fwd_up = 1 if fwd_return_10d > 0, else 0
```

#### C2. Regression: forward return
```
fwd_return_10d = raw forward 10d return of QLD
```

Also compute for evaluation:
```
fwd_return_5d, fwd_return_20d (sensitivity check on horizon)
```

### Phase D — Model Training

#### D1. Train / Validation / Test Split

| Set | Period | Purpose |
|-----|--------|---------|
| Train | 2007-01-01 → 2018-12-31 | Model fitting (12 years, includes GFC + recovery + bull) |
| Validation | 2019-03-01 → 2021-12-31 | Early stopping, hyperparameter selection (includes COVID crash + recovery) |
| Test | 2022-02-01 → 2026-05-06 | Final evaluation (includes 2022 bear + 2023-2025 bull) |

20-trading-day purge gap between each set.

#### D2. Model: XGBoost Classifier + Regressor

Same architecture as the stock rotation model — one classifier (P(up)) and one regressor (expected return), combined into a composite signal.

Hyperparameters (starting point):
```python
n_estimators = 500
learning_rate = 0.03
max_depth = 4          # shallower than stock model — less data
min_child_weight = 30
subsample = 0.8
colsample_bytree = 0.7
early_stopping_rounds = 30
```

Shallower trees and fewer estimators than the stock model because we have ~4,500 training samples (one per trading day) vs ~1.5M for stocks.

#### D3. Composite Score → Weight Mapping

```
raw_score = 0.6 * clf_prob + 0.4 * sigmoid(reg_z)
```

Map score to QLD weight:
```
weight = clip(raw_score - 0.3, 0, 1) / 0.4   # maps [0.3, 0.7] → [0, 1]
```

Thresholds (0.3, 0.7) tuned on validation set. Discretize to 4 levels to reduce turnover:

| Score Range | QLD Weight | Meaning |
|-------------|-----------|---------|
| < 0.35 | 0% | Risk-off — all cash |
| 0.35 – 0.45 | 33% | Cautious |
| 0.45 – 0.55 | 67% | Neutral-bullish |
| ≥ 0.55 | 100% | Full allocation |

#### D4. Fallback: Rule-Based Regime Model

If ML overfits (likely with ~4,500 samples), fall back to a transparent rule-based model:

```
IF vix > 30 OR qqq_drawdown < -15%:
    weight = 0%      (crisis)
ELIF qqq < SMA200 AND vix > 22:
    weight = 25%     (bear)
ELIF qqq < SMA200 OR vix > 22:
    weight = 50%     (cautious)
ELIF qqq < SMA50:
    weight = 75%     (mild pullback)
ELSE:
    weight = 100%    (bull)
```

This is the primary candidate. The ML model serves as a validation — if ML and rules agree, we have confidence. If ML beats rules on val but not test, rules win.

### Phase E-X — Cross-Fold Validation (`crossval_qld.py`)

Six test windows, each covering a major bear market + ~5 years of recovery:

| Window | Period | Bear Type |
|--------|--------|-----------|
| 2000 dot-com | 2000-01-03 → 2007-10-31 | tech crash |
| 2008 GFC | 2007-10-01 → 2014-03-31 | systemic crisis |
| 2015 chop | 2015-01-01 → 2021-02-28 | sideways |
| 2018 QT | 2018-01-01 → 2023-12-31 | fast correction |
| 2020 COVID | 2020-01-01 → 2025-03-31 | volatility shock |
| 2022 inflation | 2022-01-03 → present | rate regime |

**CV methodology:** True k-fold — for each test window, the ML model trains on ALL other data (past AND future), excluding only the test dates with a 30-day purge gap on each side. This provides a consistent training set size regardless of when the test window falls.

**Synthetic QLD:** Pre-2006 dates use 2× QQQ daily returns, anchored to actual QLD inception price.

#### Cross-Fold Summary (ML-MTF vs SMA200 vs QLD B&H)

| Window | ML Ann | ML MDD | SMA Ann | SMA MDD | B&H Ann | B&H MDD | Cal ML | Cal SMA | Cal B&H |
|--------|--------|--------|---------|---------|---------|---------|--------|---------|---------|
| 2000 dot-com | -19.6% | -96.8% | -4.8% | -83.1% | -22.4% | -98.5% | -0.202 | -0.058 | -0.227 |
| 2008 GFC | 1.0% | -83.6% | 9.5% | -43.5% | 10.5% | -83.1% | 0.012 | 0.218 | 0.126 |
| 2015 chop | 28.8% | -24.9% | 22.0% | -40.9% | 36.7% | -51.7% | **1.160** | 0.537 | 0.709 |
| 2018 QT | 19.7% | -49.1% | 25.0% | -40.9% | 26.2% | -63.7% | 0.401 | 0.610 | 0.412 |
| 2020 COVID | 39.1% | -33.4% | 31.3% | -38.3% | 22.8% | -63.7% | **1.170** | 0.817 | 0.358 |
| 2022 inflation | 20.6% | -42.3% | 23.7% | -34.3% | 17.8% | -63.2% | 0.486 | 0.689 | 0.282 |
| **Average Calmar** | | | | | | | **0.504** | **0.469** | **0.277** |

- ML-MTF beats B&H (return AND MDD): **3/6** windows
- SMA200 beats B&H (return AND MDD): **3/6** windows
- ML excels in sideways/vol-shock regimes (2015, 2020); SMA200 excels in trending regimes (2008, 2018)

#### ML-MTF Feature Importance (avg across folds, top 20)

| Rank | Feature | Importance | Timeframe |
|------|---------|-----------|-----------|
| 1 | `realized_vol_20d` | 4.78% | Daily |
| 2 | `realized_vol_8w` | 4.50% | **Weekly** |
| 3 | `vix_sma20` | 4.02% | Daily |
| 4 | `return_60d` | 3.88% | Daily |
| 5 | `return_13w` | 3.73% | **Weekly** |
| 6 | `price_vs_60d_high` | 3.71% | Daily |
| 7 | `qqq_vs_sma10m` | 3.65% | **Monthly** |
| 8 | `return_6m` | 3.48% | **Monthly** |
| 9 | `vix_close` | 3.47% | Daily |
| 10 | `qqq_drawdown` | 3.34% | Daily |
| 11 | `return_26w` | 3.26% | **Weekly** |
| 12 | `sma10m_slope` | 3.23% | **Monthly** |
| 13 | `return_12m` | 3.22% | **Monthly** |
| 14 | `days_in_drawdown` | 3.22% | Daily |
| 15 | `rsi_14m` | 3.11% | **Monthly** |
| 16 | `vol_ratio_10_60` | 3.11% | Daily |
| 17 | `qqq_vs_sma200` | 3.11% | Daily |
| 18 | `qqq_vs_sma40w` | 3.09% | **Weekly** |
| 19 | `return_3m` | 3.08% | **Monthly** |
| 20 | `macd_hist_m` | 3.00% | **Monthly** |

**Importance by timeframe:**
| Timeframe | Share |
|-----------|-------|
| Daily | 53.7% |
| Weekly | 26.7% |
| Monthly | 19.5% |

**Key insight:** Despite daily features dominating by count, weekly vol (`realized_vol_8w`) and 13-week momentum (`return_13w`) rank in the top 5. Monthly features collectively hold 19.5% — particularly monthly RSI, 6m/12m momentum, and SMA slope. The model genuinely uses all three timeframes rather than ignoring longer-horizon signals.

### Phase E — Backtest

#### E1. Engine

Reuse existing `BacktestEngine` with modifications:

- Universe: just QLD (1 ticker)
- Strategy callback returns `{"QLD": weight}` where weight ∈ [0, 1]
- Rebalance frequency: daily (check signal every day, but only trade if weight bucket changes → reduces turnover)
- Transaction cost: 5 bps per side (QLD is highly liquid, tighter spreads than individual stocks)
- Cash earns risk-free rate (approximate with SHY return or fixed 4% annualized for 2022-2025 period)

#### E2. Benchmark

**QLD buy-and-hold**: 100% allocation from day 1, no rebalancing.

#### E3. Metrics

Primary:
- Total return (must beat QLD B&H)
- Max drawdown (must beat QLD B&H)
- Sharpe ratio
- Calmar ratio (ann. return / |MDD|)

Secondary:
- Win rate (monthly)
- Time in market (% of days with weight > 0)
- Number of round-trip trades
- Transaction costs as % of capital

### Phase F — Iteration & Tuning

Priority order of experiments (run on validation set, confirm on test):

1. **Weight thresholds** — tune the score-to-weight mapping breakpoints
2. **Rebalance frequency** — daily vs weekly (reduce turnover)
3. **Hysteresis on weight changes** — require score to move by X before changing weight level (same idea as stock rotation switch_threshold)
4. **Alternative horizons** — try 5d and 20d targets instead of 10d
5. **Alternative cash proxy** — SHY (1-3Y treasuries) instead of pure cash, for yield pickup during risk-off

---

## File Structure

```
download_qld_data.py       # Phase A: download QLD/QQQ/VIX data
build_qld_features.py      # Phase B: feature engineering
train_qld_model.py         # Phase D: XGBoost + rule-based training
strategy_qld.py            # Phase E: strategy + backtest runner
data/qld_universe.parquet  # raw price data
data/qld_features.parquet  # engineered features + targets
models/qld/                # saved models + results
```

---

## Execution Order

```bash
python download_qld_data.py       # ~1 min
python build_qld_features.py      # ~10 sec
python train_qld_model.py         # ~1 min (single asset, small dataset)
python strategy_qld.py            # ~30 sec (backtest)
python strategy_qld.py --compare  # rule-based vs ML vs buy-and-hold
```

---

## Risk Notes

1. **Survivorship bias**: QLD has existed since 2006. Pre-2006 data would need to be synthetically constructed from QQQ × 2 daily returns (minus fees). Not critical for 2022-2026 test period.

2. **Leverage decay**: QLD resets daily — in sideways/volatile markets, the 2× return is less than 2× the period return. The weight model implicitly captures this because features include QLD drawdown and vol regime.

3. **Overfitting risk**: ~4,500 daily samples is thin for ML. The rule-based model is the safety net. If ML validation Sharpe < rule-based validation Sharpe, ship rules.

4. **Transaction costs**: With 4 weight levels and hysteresis, expect ~20-40 round trips per year. At 10 bps round-trip on $1M, that's $2,000-4,000/year — negligible vs the return magnitude of a 2× leveraged ETF.
