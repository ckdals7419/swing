"""Analyze feature importance and generate screening data."""
import pandas as pd
import numpy as np

clf = pd.read_csv('models/importance_clf.csv')
reg = pd.read_csv('models/importance_reg.csv')

all_cols = pd.read_parquet('data/features_phase4.parquet').columns.tolist()
meta = ['ticker','date','Open','High','Low','Close','Volume','sector']
tgt = [c for c in all_cols if c.startswith('fwd_')]
feats = [c for c in all_cols if c not in meta and c not in tgt]

merged = pd.DataFrame({'feature': feats})
merged = merged.merge(clf.rename(columns={'importance_gain':'clf_gain'}), on='feature', how='left')
merged = merged.merge(reg.rename(columns={'importance_gain':'reg_gain'}), on='feature', how='left')
merged = merged.fillna(0)

clf_total = max(merged['clf_gain'].sum(), 1)
reg_total = max(merged['reg_gain'].sum(), 1)
merged['clf_pct'] = merged['clf_gain'] / clf_total * 100
merged['reg_pct'] = merged['reg_gain'] / reg_total * 100
merged['avg_pct'] = (merged['clf_pct'] + merged['reg_pct']) / 2
merged['clf_rank'] = merged['clf_gain'].rank(ascending=False, method='min').astype(int)
merged['reg_rank'] = merged['reg_gain'].rank(ascending=False, method='min').astype(int)
merged['avg_rank'] = ((merged['clf_rank'] + merged['reg_rank']) / 2)

# Category map
CAT = {}
for f in ['sma_5','sma_10','sma_20','sma_50','sma_200','ema_5','ema_10','ema_20','ema_50',
    'price_vs_sma_5','price_vs_sma_10','price_vs_sma_20','price_vs_sma_50','price_vs_sma_200',
    'price_vs_ema_5','price_vs_ema_10','price_vs_ema_20','price_vs_ema_50',
    'sma_slope_5','sma_slope_10','sma_slope_20','sma_slope_50','sma_slope_200',
    'sma_cross_50_200','adx_14','di_plus_14','di_minus_14',
    'aroon_up_25','aroon_down_25','aroon_osc_25',
    'ichimoku_tenkan','ichimoku_kijun','ichimoku_cloud_top','ichimoku_cloud_bot',
    'ichimoku_above_cloud','ichimoku_below_cloud','ichimoku_tk_cross']: CAT[f]='Trend'
for f in ['rsi_7','rsi_14','rsi_21','stoch_k_14','stoch_d_14',
    'macd','macd_signal','macd_hist','macd_hist_slope',
    'roc_5','roc_10','roc_20','williams_r_14','cci_20','tsi',
    'momentum_1d','momentum_3d','momentum_5d','momentum_10d',
    'return_1d','return_3d','return_5d','return_10d','return_60d']: CAT[f]='Momentum'
for f in ['atr_14','atr_pct_14','bb_upper','bb_lower','bb_width','bb_pct',
    'keltner_upper','keltner_lower','keltner_pct',
    'realized_vol_5d','realized_vol_10d','realized_vol_20d','realized_vol_60d',
    'vol_ratio_10_60','parkinson_vol_20d','garman_klass_vol_20d']: CAT[f]='Volatility'
for f in ['volume_sma_20','volume_ratio_5','volume_ratio_10','volume_ratio_20',
    'volume_volatility_20','obv','obv_slope_5','ad_line','ad_slope_5',
    'chaikin_mf_20','volume_price_trend','vpt_slope_5','mfi_14','vwap_deviation']: CAT[f]='Volume'
for f in ['body_ratio','upper_shadow','lower_shadow','gap_pct','intraday_range','close_position']: CAT[f]='Price Action'
for f in ['skewness_20d','kurtosis_20d','autocorr_lag1_20d','entropy_20d',
    'mean_reversion_zscore_20d','mean_reversion_zscore_60d','hurst_63d',
    'breakout_high_20d','breakout_high_50d','breakout_low_20d','breakout_low_50d',
    'channel_position_20d','channel_position_50d',
    'consecutive_up_days','consecutive_down_days',
    'pct_from_52w_high','pct_from_52w_low','days_since_52w_high',
    'drawdown_from_high_20d','drawdown_from_high_60d','drawdown_from_high_252d']: CAT[f]='Statistical'
for f in ['day_of_week','month','quarter','week_of_year','year',
    'trading_day_of_month','is_month_start','is_month_end',
    'is_quarter_end','trading_day_of_quarter','days_to_quarter_end',
    'is_year_end','is_january','is_opex_week','sell_in_may',
    'election_cycle_phase','is_election_year','is_midterm_year',
    'is_pre_election_year','is_post_election_year','months_to_election',
    'is_inauguration_quarter','is_lame_duck','party_in_power']: CAT[f]='Calendar'
for f in ['sector_code','return_vs_market_1d','return_vs_market_20d',
    'return_vs_sector_1d','return_vs_sector_20d','sector_momentum_20d',
    'sector_rotation_rank','cross_sectional_rank','market_dispersion_1d',
    'sector_dispersion_20d','sector_vol_rank','relative_strength_vs_mkt']: CAT[f]='Sector'
for f in ['spy_return_1d','spy_return_5d','spy_return_20d','spy_return_60d',
    'spy_above_sma50','spy_vs_sma50','spy_above_sma200','spy_vs_sma200',
    'spy_drawdown_252d','spy_vol_10d','spy_vol_20d','spy_vol_60d','spy_sma50_slope',
    'vix_close','vix_sma_20','vix_change_5d','vix_percentile_252d',
    'vix_regime','vix_term_structure','beta_60d','corr_to_spy_60d']: CAT[f]='Market Regime'
for f in ['fed_funds_rate','rate_regime','rate_momentum_3m','rate_change_direction',
    'yield_curve_10y2y','yield_curve_10y3m','yield_curve_inverted','yield_curve_change_20d',
    'cpi_yoy','cpi_acceleration','unemployment_rate',
    'hy_credit_spread','credit_spread_change_20d',
    'dxy','dxy_change_20d','oil_price','oil_change_20d','gold_price','gold_change_20d']: CAT[f]='Macro'

merged['category'] = merged['feature'].map(CAT).fillna('Other')

def assign_tier(r):
    if r['clf_gain'] == 0 and r['reg_gain'] == 0:
        return 'DROP'
    if r['avg_rank'] <= 30:
        return 'Tier 1'
    elif r['avg_rank'] <= 80:
        return 'Tier 2'
    elif r['avg_rank'] <= 130:
        return 'Tier 3'
    else:
        return 'Tier 4'

merged['tier'] = merged.apply(assign_tier, axis=1)
merged = merged.sort_values('avg_rank').reset_index(drop=True)
merged.to_csv('models/feature_screening.csv', index=False)

# Print results
for t in ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'DROP']:
    sub = merged[merged['tier'] == t]
    print(f"\n{t}: {len(sub)} features")
    for _, r in sub.iterrows():
        print(f"  {r['feature']:<35} {r['category']:<16} clf#{int(r['clf_rank']):<4} reg#{int(r['reg_rank']):<4} avg={r['avg_pct']:.3f}%")

# Category summary by tier
print("\n=== CATEGORY x TIER SUMMARY ===")
ct = merged.groupby(['category','tier']).size().unstack(fill_value=0)
for col_order in [['Tier 1','Tier 2','Tier 3','Tier 4','DROP']]:
    cols = [c for c in col_order if c in ct.columns]
    ct = ct[cols]
print(ct.to_string())
