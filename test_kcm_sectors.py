"""KCM strategy test across all sector ETFs."""
import warnings, numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")
from strategy_qld import backtest, _nav_metrics

ETFS = [
    ('XLK', 'Technology'),
    ('XLF', 'Financials'),
    ('XLE', 'Energy'),
    ('XLV', 'Healthcare'),
    ('XLI', 'Industrials'),
    ('XLP', 'Staples'),
    ('XLY', 'Discretionary'),
    ('XLU', 'Utilities'),
    ('XLB', 'Materials'),
    ('QQQ', 'NASDAQ 100'),
    ('SPY', 'S&P 500'),
    ('IWM', 'Russell 2000'),
    ('SMH', 'Semiconductors'),
    ('IBB', 'Biotech'),
]

# Load VIX
vix_df = pd.read_csv('data/vix_daily.csv', parse_dates=['Date'])
vix_raw = vix_df.set_index('Date')['vix_close']

print("Downloading ETF data...")
raw = {}
for ticker, name in ETFS:
    d = yf.download(ticker, start='1993-01-01', end='2026-06-01', progress=False)
    d.columns = d.columns.get_level_values(0)
    raw[ticker] = d
    print(f"  {ticker} ({name}): {d.index[0].date()} -> {d.index[-1].date()}")


def build_and_test(ticker, name):
    d = raw[ticker]
    close = d['Close'].squeeze()
    opn   = d['Open'].squeeze()

    sma50  = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()

    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi14 = 100 - 100 / (1 + rs)

    vix = vix_raw.reindex(close.index, method='ffill')
    vix_sma10 = vix.rolling(10).mean()

    df = pd.DataFrame(index=close.index)
    df.index.name = 'date'
    df['vs200'] = close / sma200 - 1
    df['vs50']  = close / sma50 - 1
    df['rsi']   = rsi14
    df['vix']   = vix
    df['vix_sma10'] = vix_sma10
    df['bil_return_1d'] = 0.0325 / 252.0
    df['etf_open']  = opn
    df['etf_close'] = close
    df = df.dropna(subset=['vs200'])

    vs200 = df['vs200'].fillna(0).values
    vs50  = df['vs50'].fillna(0).values
    vix_v = df['vix'].values
    vix_s = df['vix_sma10'].values
    rsi_v = df['rsi'].ffill().fillna(50).values
    n = len(df)
    idx = df.index

    # Buy & Hold
    w_bh = pd.Series(np.ones(n), index=idx)

    # KCM hold
    w_h = np.zeros(n)
    inp = vs200[0] >= 0
    w_h[0] = 1.0 if inp else 0.0
    for i in range(1, n):
        has_vix = not np.isnan(vix_v[i]) and not np.isnan(vix_s[i])
        vr = has_vix and vix_v[i] > 35 and vix_v[i] < vix_s[i]
        if inp:
            if vs200[i] < -0.01 and vs50[i] < 0:
                inp = False
                if vr or rsi_v[i] < 30: w_h[i] = 1.0
                else: w_h[i] = 0.0
            else:
                w_h[i] = 1.0
        else:
            if vs200[i] >= 0.01: inp = True; w_h[i] = 1.0
            elif vs50[i] >= 0: inp = True; w_h[i] = 1.0
            elif vr or rsi_v[i] < 30: w_h[i] = 1.0
            else: w_h[i] = 0.0
    w_hold = pd.Series(w_h, index=idx)

    # KCM + RSI>80/50 swing
    w_s = np.zeros(n)
    in_regime = vs200[0] >= 0
    swing_out = False
    w_s[0] = 1.0 if in_regime else 0.0
    for i in range(1, n):
        has_vix = not np.isnan(vix_v[i]) and not np.isnan(vix_s[i])
        vr = has_vix and vix_v[i] > 35 and vix_v[i] < vix_s[i]
        if in_regime:
            if vs200[i] < -0.01 and vs50[i] < 0:
                in_regime = False; swing_out = False
                if vr or rsi_v[i] < 30: w_s[i] = 1.0
                else: w_s[i] = 0.0
            elif swing_out:
                if rsi_v[i] < 50: swing_out = False; w_s[i] = 1.0
                else: w_s[i] = 0.0
            else:
                if rsi_v[i] > 80: swing_out = True; w_s[i] = 0.0
                else: w_s[i] = 1.0
        else:
            if vs200[i] >= 0.01: in_regime = True; swing_out = False; w_s[i] = 1.0
            elif vs50[i] >= 0: in_regime = True; swing_out = False; w_s[i] = 1.0
            elif vr or rsi_v[i] < 30: w_s[i] = 1.0
            else: w_s[i] = 0.0
    w_swing = pd.Series(w_s, index=idx)

    # Run backtests
    nav_bh = backtest(df, w_bh, vehicle_open=df['etf_open'], vehicle_close=df['etf_close'])
    nav_h  = backtest(df, w_hold, vehicle_open=df['etf_open'], vehicle_close=df['etf_close'])
    nav_sw = backtest(df, w_swing, vehicle_open=df['etf_open'], vehicle_close=df['etf_close'])

    m_bh = _nav_metrics(nav_bh, w_bh)
    m_h  = _nav_metrics(nav_h, w_hold)
    m_sw = _nav_metrics(nav_sw, w_swing)

    years = (idx[-1] - idx[0]).days / 365.25
    swing_delta = m_sw['ann'] - m_h['ann']

    return {
        'ticker': ticker, 'name': name,
        'years': years,
        'bh_ann': m_bh['ann'], 'bh_mdd': m_bh['mdd'], 'bh_sharpe': m_bh['sharpe'],
        'kcm_ann': m_h['ann'], 'kcm_mdd': m_h['mdd'], 'kcm_sharpe': m_h['sharpe'], 'kcm_calmar': m_h['calmar'],
        'sw_ann': m_sw['ann'], 'sw_mdd': m_sw['mdd'], 'sw_sharpe': m_sw['sharpe'], 'sw_calmar': m_sw['calmar'],
        'swing_delta': swing_delta,
    }


# ── Run all ──
print("\nRunning backtests...\n")
results = []
for ticker, name in ETFS:
    r = build_and_test(ticker, name)
    results.append(r)

# ── Print results ──
print(f"\n{'':=<140}")
print(f"  {'ETF':<5} {'Sector':<15} {'Yrs':>4} {'|':>2}"
      f" {'B&H Ann':>8} {'B&H MDD':>8} {'|':>2}"
      f" {'KCM Ann':>8} {'KCM MDD':>8} {'KCM Shp':>8} {'KCM Cal':>8} {'|':>2}"
      f" {'Swing Ann':>9} {'Swing MDD':>9} {'Delta':>7}")
print(f"{'':=<140}")

for r in sorted(results, key=lambda x: x['kcm_ann'], reverse=True):
    print(f"  {r['ticker']:<5} {r['name']:<15} {r['years']:>4.0f} {'|':>2}"
          f" {r['bh_ann']:>8.2%} {r['bh_mdd']:>8.2%} {'|':>2}"
          f" {r['kcm_ann']:>8.2%} {r['kcm_mdd']:>8.2%} {r['kcm_sharpe']:>8.3f} {r['kcm_calmar']:>8.3f} {'|':>2}"
          f" {r['sw_ann']:>9.2%} {r['sw_mdd']:>9.2%} {r['swing_delta']:>+7.2%}")

# Summary
print(f"\n{'':=<140}")
wins = sum(1 for r in results if r['swing_delta'] > 0)
losses = sum(1 for r in results if r['swing_delta'] < 0)
ties = sum(1 for r in results if r['swing_delta'] == 0)
avg_delta = np.mean([r['swing_delta'] for r in results])
print(f"  RSI>80/50 swing vs KCM hold: {wins} wins, {losses} losses, {ties} ties  |  Avg delta: {avg_delta:+.2%}")

# KCM vs B&H
kcm_wins = sum(1 for r in results if r['kcm_ann'] > r['bh_ann'])
kcm_sharpe_wins = sum(1 for r in results if r['kcm_sharpe'] > r['bh_sharpe'])
print(f"  KCM vs Buy&Hold: {kcm_wins}/{len(results)} higher return, {kcm_sharpe_wins}/{len(results)} higher Sharpe")
