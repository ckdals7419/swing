# XGBoost Swing Trading Model — Feature Engineering Plan

## Objective

Build an XGBoost classifier/regressor for swing trading on S&P 500 stocks.

- **Holding period**: 5–20 trading days (typical swing)
- **Target variable**: Forward N-day return (e.g., 10-day) or binary label (up/down beyond threshold)
- **Training / Test split**: Time-based cutoff (e.g., train ≤ 2023, test 2024–2026)

---

## Data Available

| File | Content |
|------|---------|
| `sp500_clean_ohlcv.parquet` | 2.56M rows, 505 tickers, 1990–2026 |
| `clean_tickers.csv` | 512 clean tickers with membership dates |
| `sp500_membership_intervals.csv` | Historical S&P 500 membership periods |

**Missing (need to download)**:
- Sector / industry classification (GICS)
- Market-wide index data (SPY, VIX)
- Macroeconomic data (Fed funds rate, CPI, GDP)

---

## Target Definition

| Option | Formula | Use Case |
|--------|---------|----------|
| `fwd_return_N` | `(Close[t+N] / Close[t]) - 1` | Regression target |
| `fwd_label_N` | `1 if fwd_return_N > threshold else 0` | Classification target |
| `risk_adjusted` | `fwd_return_N / realized_vol` | Sharpe-like ranking |

Recommended: Start with `fwd_return_10d` as regression, then binarize at ±2% for classification.

---

## Feature Categories

### 1. Price Action & Candle Patterns

| Feature | Description |
|---------|-------------|
| `body_ratio` | `(Close - Open) / (High - Low)` — candle body vs range |
| `upper_shadow` | `(High - max(Open, Close)) / (High - Low)` |
| `lower_shadow` | `(min(Open, Close) - Low) / (High - Low)` |
| `gap_pct` | `(Open[t] - Close[t-1]) / Close[t-1]` — overnight gap |
| `intraday_range` | `(High - Low) / Close` |
| `close_position` | `(Close - Low) / (High - Low)` — where close sits in range |

### 2. Technical Indicators — Trend

| Feature | Window | Description |
|---------|--------|-------------|
| `sma_5, 10, 20, 50, 200` | 5–200d | Simple moving average of Close |
| `ema_5, 10, 20, 50` | 5–50d | Exponential moving average |
| `price_vs_sma_X` | — | `(Close - SMA_X) / SMA_X` — distance from MA |
| `sma_cross_5_20` | — | `1` if SMA5 > SMA20, else `0` |
| `sma_cross_50_200` | — | Golden/death cross signal |
| `sma_slope_X` | 5d | Rate of change of SMA_X |
| `adx_14` | 14d | Average Directional Index — trend strength |
| `aroon_up_25, aroon_down_25` | 25d | Aroon oscillator |
| `ichimoku_*` | — | Tenkan-sen, Kijun-sen, cloud boundaries |

### 3. Technical Indicators — Momentum

| Feature | Window | Description |
|---------|--------|-------------|
| `rsi_14` | 14d | Relative Strength Index |
| `rsi_7` | 7d | Short-term RSI |
| `stoch_k_14, stoch_d_14` | 14d | Stochastic oscillator %K, %D |
| `macd, macd_signal, macd_hist` | 12/26/9 | MACD line, signal, histogram |
| `roc_5, 10, 20` | 5–20d | Rate of change |
| `williams_r_14` | 14d | Williams %R |
| `cci_20` | 20d | Commodity Channel Index |
| `tsi` | 25/13 | True Strength Index |
| `momentum_N` | 1–20d | Simple `Close[t] / Close[t-N] - 1` returns |

### 4. Technical Indicators — Volatility

| Feature | Window | Description |
|---------|--------|-------------|
| `atr_14` | 14d | Average True Range |
| `atr_pct_14` | 14d | `ATR / Close` — normalized ATR |
| `bb_upper, bb_lower, bb_width` | 20d, 2σ | Bollinger Bands |
| `bb_pct` | 20d | `(Close - BB_lower) / BB_width` — position in band |
| `keltner_upper, keltner_lower` | 20d | Keltner Channel |
| `realized_vol_N` | 10, 20, 60d | Standard deviation of log returns |
| `vol_ratio` | — | `realized_vol_10 / realized_vol_60` — vol regime shift |
| `parkinson_vol` | 20d | High-Low based volatility estimator |
| `garman_klass_vol` | 20d | OHLC-based volatility estimator |

### 5. Volume Features

| Feature | Window | Description |
|---------|--------|-------------|
| `volume_sma_20` | 20d | Average volume |
| `volume_ratio` | — | `Volume / volume_sma_20` — relative volume |
| `obv` | — | On-Balance Volume |
| `obv_slope_5` | 5d | OBV trend |
| `vwap_deviation` | — | `(Close - VWAP) / VWAP` |
| `volume_price_trend` | — | VPT indicator |
| `mfi_14` | 14d | Money Flow Index (volume-weighted RSI) |
| `ad_line` | — | Accumulation/Distribution line |
| `chaikin_mf_20` | 20d | Chaikin Money Flow |
| `volume_volatility` | 20d | Std dev of volume — is volume itself unstable? |

### 6. Multi-Timeframe Features

Compute key indicators at multiple windows to capture different scales:

| Base Feature | Windows |
|-------------|---------|
| `return` | 1d, 3d, 5d, 10d, 20d, 60d |
| `rsi` | 7d, 14d, 21d |
| `volatility` | 5d, 10d, 20d, 60d |
| `volume_ratio` | 5d, 10d, 20d |
| `drawdown_from_high` | 20d, 60d, 252d — max drawdown from trailing high |

### 7. Calendar & Seasonality

| Feature | Description |
|---------|-------------|
| `day_of_week` | 0=Mon ... 4=Fri (known day-of-week effects) |
| `month` | 1–12 (January effect, sell-in-May) |
| `quarter` | 1–4 |
| `week_of_year` | 1–52 |
| `is_month_start` | First 3 trading days of month |
| `is_month_end` | Last 3 trading days of month (window dressing) |
| `is_quarter_end` | Last 5 trading days of quarter (rebalancing flows) |
| `is_year_end` | December effect / tax-loss harvesting |
| `is_january` | January effect |
| `trading_day_of_month` | Ordinal position in month |
| `days_to_quarter_end` | Countdown to quarter close |
| `is_opex_week` | Options expiration week (3rd Friday) |
| `is_fomc_week` | FOMC meeting week (need external calendar) |

### 8. Presidential / Political Cycle

| Feature | Description |
|---------|-------------|
| `election_year` | Year mod 4: 0=election, 1=post, 2=midterm, 3=pre-election |
| `election_cycle_phase` | Encoded as 1–4 (pre-election year historically strongest) |
| `is_election_year` | Binary |
| `is_midterm_year` | Binary (midterm year historically weak H1, strong H2) |
| `months_to_election` | Countdown to November of election year |
| `is_inauguration_quarter` | Q1 of post-election year |
| `party_in_power` | Encode incumbent party (0/1) — requires external mapping |
| `is_lame_duck` | Post-election Nov–Jan period |

**US Presidential election years**: 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024

### 9. Sector & Cross-Sectional Features

**Requires**: GICS sector/industry data (download from Yahoo Finance or Wikipedia)

| Feature | Description |
|---------|-------------|
| `sector` | GICS sector (label-encoded or one-hot) |
| `industry` | GICS sub-industry |
| `return_vs_sector` | Stock return minus sector average return |
| `return_vs_market` | Stock return minus S&P 500 return |
| `sector_momentum_20d` | Average 20d return of stocks in same sector |
| `sector_rotation_signal` | Sector rank by 20d performance (rotating money) |
| `cross_sectional_rank` | Percentile rank of stock vs all S&P 500 stocks |
| `relative_strength_vs_spy` | Rolling ratio of stock to SPY |
| `sector_vol_rank` | Sector volatility relative to market |

### 10. Market Regime Features

**Requires**: SPY / S&P 500 index data, VIX data

| Feature | Description |
|---------|-------------|
| `spy_return_Nd` | S&P 500 index 5d, 20d, 60d return |
| `spy_above_sma200` | Market above or below 200d SMA (bull/bear regime) |
| `spy_drawdown` | Drawdown from 252d high |
| `vix_level` | VIX index value |
| `vix_sma_20` | Smoothed VIX |
| `vix_percentile` | VIX percentile rank over 252d |
| `vix_term_structure` | VIX - VIX3M (contango/backwardation) |
| `market_breadth` | % of S&P 500 stocks above SMA50 |
| `advance_decline` | Net advancing vs declining stocks |
| `correlation_to_spy_60d` | Rolling beta/correlation to market |
| `dispersion` | Cross-sectional std dev of returns (high = stock-picking market) |

### 11. Macroeconomic Features

**Requires**: External data (FRED API or similar)

| Feature | Description |
|---------|-------------|
| `fed_funds_rate` | Current Fed funds rate |
| `rate_change_direction` | Hiking / cutting / paused |
| `yield_curve_10y2y` | 10Y minus 2Y Treasury (inversion = recession signal) |
| `yield_curve_10y3m` | 10Y minus 3M Treasury |
| `cpi_yoy` | CPI year-over-year change |
| `unemployment_rate` | Unemployment rate |
| `ism_pmi` | ISM Manufacturing PMI |
| `credit_spread` | HY - IG spread (risk appetite) |
| `dxy` | US Dollar Index |
| `oil_price_change_20d` | Crude oil momentum |
| `gold_price_change_20d` | Gold momentum (risk-off proxy) |

### 12. Statistical / Derived Features

| Feature | Description |
|---------|-------------|
| `hurst_exponent` | Mean-reverting (<0.5) vs trending (>0.5) |
| `autocorrelation_lag1` | Serial correlation of returns |
| `skewness_20d` | Return distribution asymmetry |
| `kurtosis_20d` | Tail risk |
| `entropy_20d` | Information entropy of return distribution |
| `mean_reversion_score` | Z-score of close vs 20d mean |
| `breakout_signal` | Close vs N-day high/low channel |
| `consecutive_up_days` | Run length of positive days |
| `consecutive_down_days` | Run length of negative days |
| `days_since_52w_high` | Recency of new high |
| `pct_from_52w_high` | Distance from 52-week high |
| `pct_from_52w_low` | Distance from 52-week low |

---

## Implementation Progress

### Phase 1 — Core (OHLCV-only) `[COMPLETE]`

| Module | File | Status |
|--------|------|--------|
| Price action & candle patterns | `features/price_action.py` | ✅ Done |
| Trend indicators (SMA, EMA, ADX, Aroon, Ichimoku) | `features/trend.py` | ✅ Done |
| Momentum (RSI, MACD, Stoch, CCI, TSI, ROC) | `features/momentum.py` | ✅ Done |
| Volatility (ATR, Bollinger, realized vol) | `features/volatility.py` | ✅ Done |
| Volume (OBV, MFI, Chaikin, VWAP) | `features/volume.py` | ✅ Done |
| Multi-timeframe returns & drawdowns | `features/multi_timeframe.py` | ✅ Done |
| Calendar & seasonality | `features/calendar_features.py` | ✅ Done |
| Presidential / political cycle | `features/calendar_features.py` | ✅ Done |
| Statistical (Hurst, autocorr, skew, breakout) | `features/statistical.py` | ✅ Done |
| Target variable generation | `features/targets.py` | ✅ Done |
| **Phase 1 orchestrator** | `build_features.py` | ✅ Done |
| **Phase 1 output** | `data/features_phase1.parquet` | ✅ Done |

### Phase 2 — Cross-sectional (sector data) `[COMPLETE]`

| Module | File | Status |
|--------|------|--------|
| Download GICS sector data | `download_sectors.py` | ✅ Done (506/512 tickers) |
| Sector & relative performance features | `features/sector_features.py` | ✅ Done |
| Phase 2 output | `data/features_phase2.parquet` | ✅ Done (2,563,655 rows × 154 cols) |

### Phase 3 — Market context (SPY, VIX) `[COMPLETE]`

| Module | File | Status |
|--------|------|--------|
| Download SPY, VIX data | `download_market_data.py` | ✅ Done (SPY 1993-2026, VIX 1990-2026) |
| Market regime features | `features/market_regime.py` | ✅ Done |
| Phase 3 output | `data/features_phase3.parquet` | ✅ Done (2,563,655 rows x 174 cols) |

### Phase 4 — Macro (FRED) `[COMPLETE]`

| Module | File | Status |
|--------|------|--------|
| Download FRED macro data | `download_macro.py` | ✅ Done (9 series, 1990-2026) |
| Macro features | `features/macro_features.py` | ✅ Done |
| Phase 4 output | `data/features_phase4.parquet` | ✅ Done (2,563,655 rows x 193 cols) |

### Model Training `[COMPLETE]`

| Task | File | Status |
|------|------|--------|
| Train/val/test split + purge | `train_model.py` | ✅ Done |
| XGBoost classifier (fwd_up_10d) | `models/clf_10d.ubj` | ✅ Done |
| XGBoost regressor (fwd_return_10d) | `models/reg_10d.ubj` | ✅ Done |
| Feature importance | `models/importance_clf.csv`, `importance_reg.csv` | ✅ Done |
| Results | `models/results.json` | ✅ Done |

#### Results

| Metric | Validation | Test |
|--------|-----------|------|
| Classifier AUC-ROC | 0.5780 | 0.5454 |
| Regressor IC (Spearman) | 0.0894 | 0.0793 |
| Long-short spread (10d) | 0.10% | 0.18% |

#### Top Features (by gain)
`sma_10`, `obv`, `ema_20`, `relative_strength_vs_mkt`, `volume_price_trend`, `atr_pct_14`, `bb_upper`, `aroon_down_25`, `gold_price`, `vix_percentile_252d`

---

## Train / Test Split Strategy

```
|-------- Training ---------|--- Validation ---|---- Test ----|
|   1990-01-01 ~ 2021-12-31 | 2022-01 ~ 2023-12 | 2024-01 ~ 2026-05 |
```

- **Time-based split only** — no random shuffle (prevents look-ahead bias)
- **Purge gap**: Drop 20 trading days between train/val and val/test to avoid label leakage
- **Cross-validation**: Use expanding-window or sliding-window time-series CV during training
- **Walk-forward**: Retrain monthly on expanding window for robustness check

---

## XGBoost Considerations

- **Handle correlated features**: XGBoost is tree-based, so correlated features don't break it but may dilute importance. Still include — the model selects what matters.
- **No scaling needed**: Tree-based models are invariant to monotonic transformations.
- **Missing values**: XGBoost handles natively — no imputation needed for early windows where indicators aren't yet defined.
- **Category encoding**: Use integer-encoded sector/industry (XGBoost supports `enable_categorical`).
- **Class imbalance**: If binary classification, tune `scale_pos_weight` or use custom objective.
- **Regularization**: Tune `max_depth`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree` to prevent overfitting with 100+ features.

---

## Estimated Feature Count

| Category | Approx. Count |
|----------|---------------|
| Price action | ~6 |
| Trend indicators | ~20 |
| Momentum indicators | ~15 |
| Volatility indicators | ~15 |
| Volume features | ~12 |
| Multi-timeframe | ~25 |
| Calendar & seasonality | ~12 |
| Presidential cycle | ~6 |
| Sector & cross-sectional | ~10 |
| Market regime | ~12 |
| Macro | ~12 |
| Statistical / derived | ~15 |
| **Total** | **~160** |
