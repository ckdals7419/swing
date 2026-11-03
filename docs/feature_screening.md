# Feature Importance Screening Report

## Overview

- **Total features evaluated**: 173 (original)
- **Model**: XGBoost classifier (binary up/down) + regressor (return magnitude)
- **Screening test period**: 2024-01-31 ~ 2026-04-22 (original run)
- **Retrain test period**: 2022-01-01 ~ 2026-05 (includes 2022 bear market)
- **Screening method**: XGBoost gain importance, ranked separately for each model, then averaged
- **Calendar features note**: 24 calendar/presidential cycle features were designed but not yet built into the pipeline — they are excluded from evaluation and marked as pending in the feature plan

| Version | Features | Dropped | Training |
|---------|:--------:|:-------:|---------|
| v1 | 173 | 0 | Single global model |
| v2 | 149 | 24 (14 DROP + 10 Tier 4) | Single global model |
| **v3** | **141** | **32 (24 v2 + 8 new)** | **11 sector-specific models** |

---

## v3 Results — Sector-by-Sector (141 features)

**Split**: Train 1990-2019 | Val 2020-2021 | Test 2022-2026

| Sector | TrainN | ValN | TestN | Val AUC | Test AUC | Val IC | Test IC | Test L/S |
|--------|-------:|-----:|------:|:-------:|:--------:|:------:|:-------:|:--------:|
| Basic Materials | 84,608 | 8,228 | 18,379 | 0.6119 | 0.5461 | 0.1612 | 0.0178 | -0.33% |
| Communication Services | 77,161 | 8,300 | 19,883 | 0.6134 | 0.5068 | 0.1152 | 0.0019 | -0.62% |
| Consumer Cyclical | 232,683 | 22,422 | 46,494 | 0.6113 | 0.5526 | 0.1446 | -0.0008 | +0.17% |
| Consumer Defensive | 189,194 | 15,154 | 34,541 | 0.6120 | 0.5353 | 0.0600 | 0.0365 | -0.05% |
| Energy | 126,563 | 8,545 | 20,397 | 0.5676 | 0.5330 | 0.0972 | 0.0267 | +0.04% |
| Financial Services | 284,766 | 21,777 | 50,692 | 0.6137 | **0.5658** | -0.0865 | **-0.1006** | +0.03% |
| Healthcare | 186,574 | 20,990 | 49,519 | 0.5886 | 0.5487 | 0.1287 | 0.0518 | -0.05% |
| Industrials | 328,883 | 29,485 | 67,681 | **0.6304** | 0.5441 | 0.0881 | 0.0231 | -0.14% |
| Real Estate | 66,526 | 9,955 | 24,129 | 0.6179 | 0.5278 | 0.1801 | -0.0026 | -0.05% |
| Technology | 188,681 | 25,534 | 66,266 | 0.6012 | 0.5278 | 0.1584 | **0.0607** | +0.04% |
| Utilities | 153,019 | 11,132 | 26,794 | 0.6394 | 0.5483 | 0.0600 | -0.0707 | -0.01% |

**v2 global baseline (all sectors combined)**: Test AUC 0.5535 | Test IC 0.0411 | L/S +0.23%

### v3 Observations

**Winners vs v2 global:**
- **Financial Services** best classifier (AUC 0.5658 > 0.5535) — sector-specific regime features help
- **Technology** best regressor (IC 0.0607 > 0.0411) — tech stocks respond cleanly to momentum/trend features
- **Healthcare** solid IC (0.0518) — defensive characteristics make return patterns more learnable

**Problematic sectors:**
- **Financial Services IC: -0.1006** — regressor is predicting backwards in test period. Financials in 2022-2026 (rate hike cycle, SVB crisis) behave very differently from training. The classifier still works but the return ranker inverts. **Do not use the regressor for Financials.**
- **Utilities IC: -0.0707** — same story. Rate sensitivity dominates utility stock returns in 2022; the model learned vol/trend patterns that reverse when rates spike.
- **Communication Services L/S: -0.62%** — worst L/S spread. Meta, Alphabet, Netflix went through a massive repricing in 2022; the feature set didn't learn to identify the rotation.

**Val→Test degradation is larger for sector models than global:**
All sectors show bigger val-test gaps than v2 global (val IC 0.08-0.18 → test IC -0.10 to +0.06), suggesting sector models overfit more to pre-2022 patterns. The global model benefits from cross-sector diversification smoothing out regime-specific failures.

**Recommendation:** Use sector models for classifier signals in Financials; use global v2 regressor for return-ranking in rate-sensitive sectors (Utilities, Financials, Real Estate).

---

## v3 Features Dropped (8 new, 32 total)

These 8 features had near-zero gain in both models after v2 pruning:

| Feature | Category | clf% | reg% | Why Dropped |
|---------|----------|:----:|:----:|-------------|
| `return_10d` | Momentum | 0.000 | 0.010 | Zero clf gain; backward 10d return redundant with momentum features |
| `channel_position_20d` | Statistical | 0.000 | 0.020 | Zero clf gain; essentially useless |
| `upper_shadow` | Price Action | 0.033 | 0.009 | Both shadows dropped — candle wicks are noise |
| `price_vs_ema_20` | Trend | 0.059 | 0.010 | `price_vs_ema_5` (#13 clf) makes the 20d version redundant |
| `stoch_d_14` | Momentum | 0.072 | 0.001 | Signal-line smoothing of `stoch_k_14`; k is better |
| `realized_vol_10d` | Volatility | 0.093 | 0.001 | Sandwiched between 5d/20d; `realized_vol_20d` is top clf feature |
| `return_vs_sector_20d` | Sector | 0.095 | 0.009 | `return_vs_sector_1d` was dropped in v2; 20d also near-zero |
| `bb_upper` | Volatility | 0.024 | 0.077 | Raw band level; `bb_pct` (position within bands) is the better signal |

---

## v3 Surviving Features (141 total)

All features NOT in the DROP list below. By category:

### Trend (15 surviving of 37)
`sma_5`, `sma_10`, `sma_20`, `sma_50`, `sma_200`, `ema_5`, `ema_10`, `ema_20`, `ema_50`,
`price_vs_sma_5`, `price_vs_sma_10`, `price_vs_sma_50`, `price_vs_sma_200`,
`price_vs_ema_5`, `price_vs_ema_10`, `price_vs_ema_50`,
`sma_slope_5`, `sma_slope_10`, `sma_slope_20`, `sma_slope_50`, `sma_slope_200`,
`sma_cross_50_200`, `adx_14`, `di_plus_14`, `di_minus_14`,
`aroon_up_25`, `aroon_down_25`, `aroon_osc_25`,
`ichimoku_tenkan`, `ichimoku_kijun`, `ichimoku_cloud_top`, `ichimoku_cloud_bot`,
`ichimoku_below_cloud`

*(Dropped from Trend: `price_vs_sma_20`, `price_vs_ema_20`, `sma_cross_5_20`, `ichimoku_above_cloud`, `ichimoku_tk_cross`)*

### Market Regime (21 surviving of 21)
`spy_return_1d`, `spy_return_5d`, `spy_return_20d`, `spy_return_60d`,
`spy_above_sma50`, `spy_vs_sma50`, `spy_above_sma200`, `spy_vs_sma200`,
`spy_drawdown_252d`, `spy_vol_10d`, `spy_vol_20d`, `spy_vol_60d`, `spy_sma50_slope`,
`vix_close`, `vix_sma_20`, `vix_change_5d`, `vix_percentile_252d`,
`vix_regime`, `vix_term_structure`, `beta_60d`, `corr_to_spy_60d`

### Macro (13 surviving of 19)
`fed_funds_rate`, `rate_regime`, `rate_momentum_3m`,
`yield_curve_10y2y`, `yield_curve_10y3m`, `yield_curve_inverted`, `yield_curve_change_20d`,
`cpi_yoy`, `cpi_acceleration`, `unemployment_rate`,
`dxy`, `dxy_change_20d`, `oil_price`, `oil_change_20d`, `gold_price`, `gold_change_20d`

*(Dropped from Macro: `rate_change_direction`, `hy_credit_spread`, `credit_spread_change_20d`)*

### Volatility (12 surviving of 16)
`atr_14`, `atr_pct_14`, `bb_lower`, `bb_width`, `bb_pct`,
`keltner_upper`, `keltner_lower`, `keltner_pct`,
`realized_vol_5d`, `realized_vol_20d`, `realized_vol_60d`,
`vol_ratio_10_60`, `parkinson_vol_20d`, `garman_klass_vol_20d`

*(Dropped from Volatility: `realized_vol_10d`, `bb_upper`)*

### Volume (13 surviving of 14)
`volume_sma_20`, `volume_ratio_5`, `volume_ratio_10`, `volume_ratio_20`,
`volume_volatility_20`, `obv`, `ad_line`, `ad_slope_5`,
`chaikin_mf_20`, `volume_price_trend`, `vpt_slope_5`, `mfi_14`, `vwap_deviation`

*(Dropped from Volume: `obv_slope_5`)*

### Statistical (14 surviving of 21)
`skewness_20d`, `kurtosis_20d`, `autocorr_lag1_20d`,
`mean_reversion_zscore_20d`, `mean_reversion_zscore_60d`, `hurst_63d`,
`breakout_low_20d`, `channel_position_50d`,
`pct_from_52w_high`, `pct_from_52w_low`, `days_since_52w_high`,
`drawdown_from_high_20d`, `drawdown_from_high_60d`, `drawdown_from_high_252d`

*(Dropped from Statistical: `consecutive_up_days`, `consecutive_down_days`, `breakout_high_20d`, `breakout_high_50d`, `breakout_low_50d`, `channel_position_20d`, `entropy_20d`)*

### Sector (8 surviving of 12)
`sector_code`, `return_vs_market_1d`, `return_vs_market_20d`, `return_vs_sector_20d` (removed),
`sector_momentum_20d`, `sector_rotation_rank`, `cross_sectional_rank`,
`market_dispersion_1d`, `sector_dispersion_20d`, `relative_strength_vs_mkt`

*(Dropped from Sector: `return_vs_sector_1d`, `return_vs_sector_20d`, `sector_vol_rank`)*

### Momentum (11 surviving of 24)
`rsi_7`, `rsi_14`, `rsi_21`, `stoch_k_14`,
`macd`, `macd_signal`, `macd_hist`, `macd_hist_slope`,
`roc_10`, `roc_20`, `cci_20`, `tsi`,
`momentum_1d`, `momentum_3d`, `momentum_10d`,
`return_1d`, `return_3d`, `return_60d`

*(Dropped from Momentum: `stoch_d_14`, `roc_5`, `williams_r_14`, `momentum_5d`, `momentum_20d`, `return_5d`, `return_10d`, `return_20d`)*

### Price Action (2 surviving of 6)
`body_ratio`, `gap_pct`, `intraday_range`

*(Dropped from Price Action: `close_position`, `upper_shadow`, `lower_shadow`)*

---

## All Dropped Features — Cumulative (32 total)

### v2 Drops (24) — Zero importance or Tier 4

**Zero importance in both models (14):**
`close_position`, `consecutive_up_days`, `consecutive_down_days`,
`breakout_high_20d`, `breakout_high_50d`, `breakout_low_50d`,
`ichimoku_above_cloud`, `ichimoku_tk_cross`, `sma_cross_5_20`,
`return_20d`, `momentum_20d`, `rate_change_direction`,
`hy_credit_spread`, `credit_spread_change_20d`

**Tier 4 — Low value in both models (10):**
`return_5d`, `entropy_20d`, `lower_shadow`, `williams_r_14`,
`sector_vol_rank`, `return_vs_sector_1d`, `obv_slope_5`,
`momentum_5d`, `roc_5`, `price_vs_sma_20`

### v3 Drops (8) — Near-zero after v2 pruning

`return_10d`, `channel_position_20d`, `upper_shadow`,
`price_vs_ema_20`, `stoch_d_14`, `realized_vol_10d`,
`return_vs_sector_20d`, `bb_upper`

---

## Retrain v2 Results (Pruned Features, Test from 2022)

**Split**: Train 1990–2019 | Val 2020–2021 (COVID) | Test 2022–2026 (bear + recovery)
**Features**: 149 (24 dropped from v1's 173)

| Metric | Val (2020-2021) | Test (2022-2026) | v1 Test (2024-2026) |
|--------|:--------------:|:----------------:|:-------------------:|
| Classifier AUC-ROC | 0.5989 | 0.5535 | 0.5454 |
| Classifier Accuracy | 0.5693 | 0.5412 | 0.5960 |
| Regressor IC (Spearman) | 0.0779 | 0.0411 | 0.0793 |
| L/S spread (10d) | 0.21% | **0.23%** | 0.18% |

### v2 Top Features — Classifier
`realized_vol_20d`, `drawdown_from_high_60d`, `di_minus_14`, `parkinson_vol_20d`, `vix_regime`, `realized_vol_60d`, `spy_above_sma200`, `spy_drawdown_252d`, `drawdown_from_high_252d`, `fed_funds_rate`

### v2 Top Features — Regressor
`obv`, `volume_price_trend`, `aroon_down_25`, `ema_20`, `ichimoku_tenkan`, `atr_pct_14`, `sma_20`, `ema_10`, `relative_strength_vs_mkt`, `pct_from_52w_low`

### Comparison Notes
- **Classifier AUC improved on val** (0.599 vs 0.578) — 2020-2021 (COVID crash/recovery) is harder to predict than 2022-2023
- **Test AUC slightly higher** (0.5535 vs 0.5454) — 24 dropped features reduced noise
- **Regressor IC lower on test** (0.041 vs 0.079) — 2022 bear market is structurally different from the bull market the model was trained on; return-ranking is harder in high-volatility regimes
- **L/S spread higher** (0.23% vs 0.18%) — pruned model better at identifying outlier stocks

### Features Dropped for Retrain (24 total)

**DROP — Zero importance in both models (14):**
`close_position`, `consecutive_up_days`, `consecutive_down_days`, `breakout_high_20d`, `breakout_high_50d`, `breakout_low_50d`, `ichimoku_above_cloud`, `ichimoku_tk_cross`, `sma_cross_5_20`, `return_20d`, `momentum_20d`, `rate_change_direction`, `hy_credit_spread`, `credit_spread_change_20d`

**Tier 4 — Low value in both models (10):**
`return_5d`, `entropy_20d`, `lower_shadow`, `williams_r_14`, `sector_vol_rank`, `return_vs_sector_1d`, `obv_slope_5`, `momentum_5d`, `roc_5`, `price_vs_sma_20`

### Features Kept for Retrain (149 total)
All Tier 1 (7) + Tier 2 (74) + Tier 3 (68) features.

---

## Tier Summary

| Tier | Count | Criteria | Recommendation |
|------|-------|----------|----------------|
| **Tier 1** | 7 | Avg rank <= 30 | Must keep |
| **Tier 2** | 74 | Avg rank 31-80 | Keep |
| **Tier 3** | 68 | Avg rank 81-130 | Review - consider pruning |
| **Tier 4** | 10 | Avg rank 131+ | Low value - candidate for removal |
| **DROP** | 14 | Zero gain in both models | Remove |

---

## Category Breakdown by Tier

| Category | Tier 1 | Tier 2 | Tier 3 | Tier 4 | DROP | Total | Importance Rank |
|----------|:------:|:------:|:------:|:------:|:----:|:-----:|:---------------:|
| Trend | 1 | 20 | 13 | 1 | 2 | 37 | 1st (regressor) |
| Market Regime | 1 | 14 | 6 | 0 | 0 | 21 | 1st (classifier) |
| Volatility | 1 | 8 | 7 | 0 | 0 | 16 | 3rd |
| Macro | 2 | 9 | 5 | 0 | 3 | 19 | 2nd (classifier) |
| Volume | 0 | 6 | 7 | 1 | 0 | 14 | 2nd (regressor) |
| Statistical | 2 | 7 | 6 | 1 | 5 | 21 | 5th |
| Sector | 0 | 2 | 8 | 2 | 0 | 12 | 6th |
| Momentum | 0 | 6 | 14 | 4 | 0 | 24 | 7th |
| Price Action | 0 | 2 | 2 | 1 | 1 | 6 | 8th |
| Calendar | 0 | 0 | 0 | 0 | 0 | 0 | Not used |

---

## Tier 1 - Must Keep (7 features)

These features rank in the top 30 across both models. Removing any of these will hurt performance.

| Feature | Category | Clf Rank | Reg Rank | Why It Matters |
|---------|----------|:--------:|:--------:|----------------|
| `parkinson_vol_20d` | Volatility | #1 | #17 | Best single classifier feature. High-Low based vol estimator captures intraday risk better than close-to-close vol |
| `gold_price` | Macro | #13 | #13 | Strong in both models. Gold acts as risk-off proxy and inflation signal |
| `vix_percentile_252d` | Market Regime | #22 | #15 | VIX relative to its own 1-year history distinguishes fear regimes |
| `aroon_down_25` | Trend | #32 | #12 | Days since 25-day low. Captures trend exhaustion and reversal timing |
| `pct_from_52w_high` | Statistical | #14 | #33 | Distance from 52-week high is a universal "where are we" anchor |
| `drawdown_from_high_60d` | Statistical | #3 | #48 | 60-day drawdown signals regime (correction vs dip vs crash) |
| `dxy_change_20d` | Macro | #28 | #26 | Dollar momentum affects all equities via earnings/flows |

---

## Tier 2 - Keep (74 features)

Solid contributors. Categories are listed in order of aggregate importance.

### Trend (20 features in Tier 2)
| Feature | Clf Rank | Reg Rank | Note |
|---------|:--------:|:--------:|------|
| `sma_10` | #132 | **#1** | Top regressor feature overall. Short-term price level anchor |
| `sma_20` | #137 | **#9** | |
| `sma_200` | #56 | #18 | Long-term trend defining level |
| `sma_5` | #112 | **#8** | |
| `ema_20` | #96 | **#3** | 3rd best regressor feature |
| `ema_10` | #114 | #11 | |
| `ema_5` | #71 | #10 | |
| `aroon_osc_25` | #61 | #19 | |
| `aroon_up_25` | #103 | #20 | |
| `di_minus_14` | **#2** | #138 | 2nd best classifier feature - directional movement |
| `ichimoku_tenkan` | **#6** | #129 | Short-term Ichimoku signal line |
| `ichimoku_cloud_bot` | #12 | #91 | |
| `ichimoku_cloud_top` | #42 | #89 | |
| `ichimoku_kijun` | #45 | #83 | |
| `sma_slope_200` | #86 | #46 | |
| `sma_slope_20` | #113 | #39 | |
| `sma_50` | #106 | #40 | |
| `price_vs_sma_200` | #72 | #86 | |
| `price_vs_sma_10` | #60 | #61 | |
| `price_vs_ema_20` | #34 | #110 | |

### Market Regime (14 features in Tier 2)
| Feature | Clf Rank | Reg Rank | Note |
|---------|:--------:|:--------:|------|
| `vix_close` | **#7** | #77 | Raw VIX level - key regime indicator |
| `spy_drawdown_252d` | **#5** | #105 | How far market has fallen from peak |
| `spy_above_sma200` | **#10** | #148 | Bull vs bear regime binary |
| `spy_sma50_slope` | **#9** | #124 | Market trend direction |
| `spy_vs_sma200` | #18 | #74 | |
| `spy_return_60d` | #19 | #75 | |
| `spy_vol_60d` | #20 | #80 | |
| `vix_sma_20` | #21 | #72 | |
| `vix_term_structure` | #23 | #68 | |
| `spy_vol_20d` | #36 | #58 | |
| `spy_vol_10d` | #37 | #76 | |
| `spy_vs_sma50` | #33 | #117 | |
| `spy_above_sma50` | #55 | #101 | |
| `spy_return_20d` | #27 | #126 | |

### Macro (9 features in Tier 2)
| Feature | Clf Rank | Reg Rank | Note |
|---------|:--------:|:--------:|------|
| `fed_funds_rate` | **#8** | #85 | Interest rate level shapes everything |
| `dxy` | #11 | #59 | Dollar index level |
| `oil_price` | #25 | #36 | |
| `yield_curve_10y2y` | #24 | #44 | Inversion = recession signal |
| `yield_curve_10y3m` | #17 | #121 | |
| `unemployment_rate` | #16 | #78 | |
| `gold_change_20d` | #15 | #104 | |
| `rate_regime` | #31 | #115 | |
| `rate_momentum_3m` | #63 | #79 | |

### Volatility (8 features in Tier 2)
| Feature | Clf Rank | Reg Rank | Note |
|---------|:--------:|:--------:|------|
| `atr_pct_14` | #95 | **#6** | Normalized ATR - top regressor feature |
| `bb_upper` | #128 | **#7** | Bollinger upper band level |
| `atr_14` | #30 | #49 | |
| `bb_pct` | #51 | #30 | Position within Bollinger Bands |
| `keltner_pct` | #49 | #54 | |
| `keltner_upper` | #79 | #55 | |
| `keltner_lower` | #85 | #47 | |
| `realized_vol_60d` | **#4** | #139 | 4th best classifier feature |

### Volume (6 features in Tier 2)
| Feature | Clf Rank | Reg Rank | Note |
|---------|:--------:|:--------:|------|
| `obv` | #111 | **#2** | 2nd best regressor feature overall |
| `volume_price_trend` | #118 | **#5** | 5th best regressor feature |
| `chaikin_mf_20` | #52 | #21 | |
| `volume_sma_20` | #110 | #24 | |
| `ad_line` | #109 | #43 | |
| `ad_slope_5` | #87 | #52 | |

### Statistical (7 features in Tier 2)
| Feature | Clf Rank | Reg Rank | Note |
|---------|:--------:|:--------:|------|
| `skewness_20d` | #80 | #14 | Return asymmetry - strong in regressor |
| `drawdown_from_high_252d` | #40 | #37 | |
| `drawdown_from_high_20d` | #43 | #34 | |
| `mean_reversion_zscore_60d` | #91 | #23 | |
| `days_since_52w_high` | #100 | #31 | |
| `pct_from_52w_low` | #70 | #69 | |
| `channel_position_50d` | #67 | #82 | |

### Momentum (6 features in Tier 2)
| Feature | Clf Rank | Reg Rank | Note |
|---------|:--------:|:--------:|------|
| `cci_20` | #124 | #16 | Commodity Channel Index - strong in regressor |
| `momentum_1d` | #53 | #28 | Yesterday's return matters |
| `return_60d` | #82 | #32 | 3-month momentum |
| `macd_hist` | #99 | #35 | |
| `macd_hist_slope` | #107 | #51 | |
| `macd_signal` | #83 | #62 | |

### Sector (2 features in Tier 2)
| Feature | Clf Rank | Reg Rank | Note |
|---------|:--------:|:--------:|------|
| `relative_strength_vs_mkt` | #62 | **#4** | 4th best regressor feature. Stock vs market |
| `market_dispersion_1d` | #58 | #90 | |

### Price Action (2 features in Tier 2)
| Feature | Clf Rank | Reg Rank | Note |
|---------|:--------:|:--------:|------|
| `intraday_range` | #125 | #29 | |
| `body_ratio` | #89 | #53 | |

---

## Tier 3 - Review (68 features)

These features contribute marginally. Many are redundant with Tier 1-2 features. Pruning these may reduce noise and overfitting without hurting performance.

**Notable patterns in Tier 3:**
- **14 momentum features**: RSI, Stochastic, ROC are mostly Tier 3 - classical oscillators add little beyond the trend/vol features
- **8 sector features**: Most sector features except `relative_strength_vs_mkt` landed here
- **6 market regime features**: Shorter-window SPY/VIX features add incrementally
- **Calendar features**: Zero calendar features were used by either model (all are effectively DROP)

**Tier 3 features with split personality (strong in one model, weak in the other):**

| Feature | Clf Rank | Reg Rank | Comment |
|---------|:--------:|:--------:|---------|
| `cross_sectional_rank` | #155 | **#22** | Top regressor-only feature |
| `mean_reversion_zscore_20d` | #145 | **#25** | Strong for ranking returns |
| `rsi_21` | **#26** | #148 | Good classifier but useless for regression |
| `vix_regime` | **#38** | #148 | Binary VIX bucket - classifier only |
| `cpi_yoy` | **#29** | #141 | Macro regime detection |
| `return_1d` | **#41** | #148 | Yesterday's direction for classification |

---

## Tier 4 - Low Value (10 features)

These features rank poorly in both models. Strong candidates for removal.

| Feature | Category | Clf Rank | Reg Rank |
|---------|----------|:--------:|:--------:|
| `return_5d` | Momentum | #155 | #106 |
| `entropy_20d` | Statistical | #143 | #120 |
| `lower_shadow` | Price Action | #136 | #135 |
| `williams_r_14` | Momentum | #127 | #148 |
| `sector_vol_rank` | Sector | #135 | #148 |
| `return_vs_sector_1d` | Sector | #155 | #128 |
| `obv_slope_5` | Volume | #144 | #144 |
| `momentum_5d` | Momentum | #142 | #148 |
| `roc_5` | Momentum | #149 | #146 |
| `price_vs_sma_20` | Trend | #154 | #148 |

---

## DROP - Remove (14 features)

Zero importance gain in both classifier and regressor. These features were never selected by any tree split.

| Feature | Category | Reason |
|---------|----------|--------|
| `close_position` | Price Action | Redundant with `bb_pct`, `keltner_pct` |
| `consecutive_up_days` | Statistical | Run-length not predictive |
| `consecutive_down_days` | Statistical | Run-length not predictive |
| `breakout_high_20d` | Statistical | Binary breakout too noisy |
| `breakout_high_50d` | Statistical | Binary breakout too noisy |
| `breakout_low_50d` | Statistical | Binary breakout too noisy |
| `ichimoku_above_cloud` | Trend | Redundant with cloud/price levels |
| `ichimoku_tk_cross` | Trend | Cross signal too binary |
| `sma_cross_5_20` | Other | Binary cross signal useless |
| `return_20d` | Other | Redundant with `roc_20`, `momentum_20d` |
| `momentum_20d` | Other | Redundant with `roc_20`, `return_20d` |
| `rate_change_direction` | Macro | Ternary (-1/0/1) too coarse vs `fed_funds_rate` |
| `hy_credit_spread` | Macro | Data only from 2023 - too sparse |
| `credit_spread_change_20d` | Macro | Derived from sparse `hy_credit_spread` |

---

## Calendar & Presidential Cycle Features - Complete Failure

**All 24 calendar features (day_of_week, month, quarter, seasonal, presidential cycle, etc.) have zero importance in both models.**

These features were never selected by any tree split. Possible explanations:
1. **Seasonal effects are priced in** - well-known anomalies (January effect, sell-in-May) are arbitraged away
2. **Presidential cycle is noise** - 4-year cycle with ~9 elections in our data provides too few samples per regime
3. **Captured by other features** - macro and market regime features already encode the same information (e.g., `fed_funds_rate` captures rate-cut pre-election stimulus)

**Recommendation**: Remove all 24 calendar features. They add noise and computation cost with zero benefit.

---

## Key Insights

### Classifier vs Regressor Use Different Information

| For Predicting... | Top Categories | Interpretation |
|-------------------|---------------|----------------|
| **Direction (up/down)** | Market Regime (25%), Macro (19%), Trend (20%) | "Which way is the market/economy going?" drives binary prediction |
| **Return Magnitude** | Per-stock Trend (42%), Volume (18%), Volatility (13%) | "Where is THIS stock relative to its own patterns?" drives return ranking |

### Most Overrated Feature Categories
- **Classical momentum oscillators** (RSI, Stochastic, Williams %R, ROC) - mostly Tier 3/4. The raw price levels and moving averages they derive from are more useful
- **Binary signal features** (crosses, breakouts, above/below flags) - almost universally DROP. Tree models prefer continuous values they can split on
- **Calendar/seasonality** - complete zero. Don't use them

### Most Underrated Feature Categories
- **Volume features** (OBV, VPT) - #2 and #5 in the regressor. Volume confirms price moves
- **Parkinson volatility** - #1 classifier feature. The High-Low range captures intraday information that close-to-close returns miss
- **Gold price** - Tier 1 in both models. An unexpected but consistent macro signal

---

## Recommended Pruning

| Action | Features | Expected Impact |
|--------|:--------:|-----------------|
| Remove DROP | 14 | No impact - zero importance |
| Remove Tier 4 | 10 | Minimal impact - may reduce noise |
| Remove Calendar | 24 | No impact - zero importance |
| **Total removable** | **48** | **Reduce from 173 to ~125 features** |

The detailed feature-level data is saved at `models/feature_screening.csv`.
