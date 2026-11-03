"""KCM strategy test across representative individual stocks."""
import warnings, numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")
from strategy_qld import backtest, _nav_metrics

STOCKS = [
    # Mega-cap tech (M7)
    ('AAPL', 'Apple',          'Mega Tech'),
    ('MSFT', 'Microsoft',      'Mega Tech'),
    ('GOOG', 'Google',         'Mega Tech'),
    ('AMZN', 'Amazon',         'Mega Tech'),
    ('NVDA', 'NVIDIA',         'Mega Tech'),
    ('META', 'Meta',           'Mega Tech'),
    ('TSLA', 'Tesla',          'Mega Tech'),
    # Large-cap tech
    ('ADBE', 'Adobe',          'Large Tech'),
    ('CRM',  'Salesforce',     'Large Tech'),
    ('NFLX', 'Netflix',        'Large Tech'),
    ('AMD',  'AMD',            'Large Tech'),
    ('AVGO', 'Broadcom',       'Large Tech'),
    # Non-tech blue chips
    ('JNJ',  'J&J',            'Healthcare'),
    ('UNH',  'UnitedHealth',   'Healthcare'),
    ('PG',   'P&G',            'Staples'),
    ('KO',   'Coca-Cola',      'Staples'),
    ('WMT',  'Walmart',        'Retail'),
    ('HD',   'Home Depot',     'Retail'),
    ('JPM',  'JPMorgan',       'Finance'),
    ('V',    'Visa',           'Finance'),
    # Energy / Industrial / Other
    ('XOM',  'ExxonMobil',     'Energy'),
    ('CVX',  'Chevron',        'Energy'),
    ('CAT',  'Caterpillar',    'Industrial'),
    ('BA',   'Boeing',         'Industrial'),
    ('DIS',  'Disney',         'Media'),
    ('NKE',  'Nike',           'Consumer'),
    ('COST', 'Costco',         'Retail'),
    ('LLY',  'Eli Lilly',      'Pharma'),
]

vix_df = pd.read_csv('data/vix_daily.csv', parse_dates=['Date'])
vix_raw = vix_df.set_index('Date')['vix_close']

print(f"Downloading {len(STOCKS)} stocks...")
raw = {}
for ticker, name, sector in STOCKS:
    d = yf.download(ticker, start='1980-01-01', end='2026-06-01', progress=False)
    d.columns = d.columns.get_level_values(0)
    raw[ticker] = d

print("Running backtests...\n")


def test_stock(ticker, name, sector):
    d = raw[ticker]
    if len(d) < 300:
        return None

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
    df['stk_open']  = opn
    df['stk_close'] = close
    df = df.dropna(subset=['vs200'])

    vs200 = df['vs200'].fillna(0).values
    vs50  = df['vs50'].fillna(0).values
    vix_v = df['vix'].values
    vix_s = df['vix_sma10'].values
    rsi_v = df['rsi'].ffill().fillna(50).values
    n = len(df)
    idx = df.index

    # B&H
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

    nav_bh = backtest(df, w_bh, vehicle_open=df['stk_open'], vehicle_close=df['stk_close'])
    nav_h  = backtest(df, w_hold, vehicle_open=df['stk_open'], vehicle_close=df['stk_close'])
    nav_sw = backtest(df, w_swing, vehicle_open=df['stk_open'], vehicle_close=df['stk_close'])

    m_bh = _nav_metrics(nav_bh, w_bh)
    m_h  = _nav_metrics(nav_h, w_hold)
    m_sw = _nav_metrics(nav_sw, w_swing)

    years = (idx[-1] - idx[0]).days / 365.25
    return {
        'ticker': ticker, 'name': name, 'sector': sector, 'years': years,
        'bh_ann': m_bh['ann'], 'bh_mdd': m_bh['mdd'], 'bh_sharpe': m_bh['sharpe'],
        'kcm_ann': m_h['ann'], 'kcm_mdd': m_h['mdd'], 'kcm_sharpe': m_h['sharpe'], 'kcm_calmar': m_h['calmar'],
        'sw_ann': m_sw['ann'], 'sw_mdd': m_sw['mdd'],
        'kcm_vs_bh': m_h['ann'] - m_bh['ann'],
        'swing_delta': m_sw['ann'] - m_h['ann'],
    }


results = []
for ticker, name, sector in STOCKS:
    r = test_stock(ticker, name, sector)
    if r:
        results.append(r)

# ── Print by sector ──
print(f"\n{'':=<145}")
print(f"  {'Ticker':<6} {'Name':<14} {'Sector':<12} {'Yrs':>4} {'|':>1}"
      f" {'B&H Ann':>8} {'B&H MDD':>8} {'|':>1}"
      f" {'KCM Ann':>8} {'KCM MDD':>8} {'KCM Shp':>8} {'vs B&H':>7} {'|':>1}"
      f" {'Swing':>8} {'SwDelta':>8}")
print(f"{'':=<145}")

current_sector = None
for r in results:
    if r['sector'] != current_sector:
        current_sector = r['sector']
        print(f"  --- {current_sector} ---")
    print(f"  {r['ticker']:<6} {r['name']:<14} {r['sector']:<12} {r['years']:>4.0f} |"
          f" {r['bh_ann']:>8.2%} {r['bh_mdd']:>8.2%} |"
          f" {r['kcm_ann']:>8.2%} {r['kcm_mdd']:>8.2%} {r['kcm_sharpe']:>8.3f} {r['kcm_vs_bh']:>+7.2%} |"
          f" {r['sw_ann']:>8.2%} {r['swing_delta']:>+8.2%}")

# ── Summary ──
print(f"\n{'':=<145}")
kcm_wins = sum(1 for r in results if r['kcm_vs_bh'] > 0)
kcm_sharpe_wins = sum(1 for r in results if r['kcm_sharpe'] > r['bh_sharpe'])
swing_wins = sum(1 for r in results if r['swing_delta'] > 0)

print(f"  KCM vs Buy&Hold return:  {kcm_wins}/{len(results)} wins  (avg delta: {np.mean([r['kcm_vs_bh'] for r in results]):+.2%})")
print(f"  KCM vs Buy&Hold Sharpe:  {kcm_sharpe_wins}/{len(results)} wins")
print(f"  RSI swing vs KCM hold:   {swing_wins}/{len(results)} wins  (avg delta: {np.mean([r['swing_delta'] for r in results]):+.2%})")

# By sector group
print(f"\n  By sector:")
for sector in ['Mega Tech', 'Large Tech', 'Healthcare', 'Staples', 'Retail', 'Finance', 'Energy', 'Industrial', 'Media', 'Consumer', 'Pharma']:
    sr = [r for r in results if r['sector'] == sector]
    if not sr:
        continue
    avg_kcm = np.mean([r['kcm_vs_bh'] for r in sr])
    avg_sw  = np.mean([r['swing_delta'] for r in sr])
    w = sum(1 for r in sr if r['kcm_vs_bh'] > 0)
    print(f"    {sector:<14}: KCM vs B&H {avg_kcm:+.2%} ({w}/{len(sr)} win)  |  Swing delta {avg_sw:+.2%}")
