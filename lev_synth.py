"""
Synthetic leveraged-ETF builder (reusable for any index: NDX→TQQQ, SPX→UPRO, ...).

Daily model (close-to-close, calendar-day financing accrual):

  r_etf = L·r_idx − [ER + (L−1)·(rf + spread)] · days/365 + c2·r_idx²

  r_idx   index daily return (price index; dividends absorbed by spread fit)
  rf      short rate (^IRX 13-week T-bill, %/100), previous day's value
  days    calendar days since previous trading day (weekend financing)
  ER      fund expense ratio
  spread  swap financing spread over rf (fit on real ETF overlap)
  c2      rebalancing/vol drag coefficient (fit; 0 = pure model)

Open price: prev synth close × (1 + L·(idx_open/idx_prev_close − 1)) — overnight gap only.

Usage:
  from lev_synth import load_inputs, calibrate, synth, splice
  idx, rf = load_inputs("^NDX")
  p = calibrate(idx, real_etf_close, rf, lev=3, er=0.0084)
  syn = synth(idx, rf, lev=3, **p)          # DataFrame Open/Close
  veh = splice(syn, real_etf_ohlc)           # synthetic before, real after listing

Calibrated presets live in PRESETS (see README "3배 지수 합성").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

# Fit on real ETF overlap; see `python lev_synth.py` for the calibration report.
# index      : price index for signals / overnight gap
# tr_etf     : ETF whose adjusted close gives the total-return index (+ its ER)
# pre_div    : assumed dividend yield before tr_etf exists
# spread, c2 : calibrate_drift(..., fit_c2=True) over the full real-ETF history
PRESETS: dict[str, dict] = {
    "TQQQ": {
        "index": "^NDX", "tr_etf": "QQQ", "tr_er": 0.0020, "pre_div": 0.005,
        "lev": 3.0, "er": 0.0084, "spread": 0.003348, "c2": -0.2531,
        "real": "TQQQ", "fit": "2010-02-11~2026-11",
    },
}


def _dl(t: str, start: str = "1985-01-01") -> pd.DataFrame:
    d = yf.download(t, start=start, auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d


def load_inputs(index_ticker: str, start: str = "1985-01-01") -> tuple[pd.DataFrame, pd.Series]:
    idx = _dl(index_ticker, start)[["Open", "High", "Low", "Close"]].astype(float)
    irx = _dl("^IRX", start)["Close"].astype(float) / 100.0
    rf = irx.reindex(idx.index).ffill().bfill()
    return idx, rf


def _features(idx_close: pd.Series, rf: pd.Series) -> pd.DataFrame:
    r = idx_close.pct_change()
    days = idx_close.index.to_series().diff().dt.days.fillna(1).astype(float)
    rf_prev = rf.reindex(idx_close.index).ffill().shift(1).bfill()
    return pd.DataFrame({"r": r, "days": days, "rf": rf_prev})


def synth_returns(
    idx_close: pd.Series,
    rf: pd.Series,
    *,
    lev: float,
    er: float,
    spread: float = 0.0,
    c2: float = 0.0,
    rf_mult: float | None = None,
) -> pd.Series:
    f = _features(idx_close, rf)
    m = (lev - 1) if rf_mult is None else rf_mult
    fin = (er + m * f["rf"] + (lev - 1) * spread) * f["days"] / 365.0
    return (lev * f["r"] - fin + c2 * f["r"] ** 2).fillna(0.0)


def synth(
    idx: pd.DataFrame,
    rf: pd.Series,
    *,
    lev: float,
    er: float,
    spread: float = 0.0,
    c2: float = 0.0,
    rf_mult: float | None = None,
    base: float = 100.0,
) -> pd.DataFrame:
    c = idx["Close"].astype(float)
    ret = synth_returns(c, rf, lev=lev, er=er, spread=spread, c2=c2, rf_mult=rf_mult)
    close = base * (1 + ret).cumprod()
    gap = (idx["Open"].astype(float) / c.shift(1) - 1).fillna(0.0)
    open_ = close.shift(1) * (1 + lev * gap)
    open_.iloc[0] = base
    return pd.DataFrame({"Open": open_, "Close": close})


def calibrate(
    idx_close: pd.Series,
    etf_close: pd.Series,
    rf: pd.Series,
    *,
    lev: float,
    er: float,
    start: str | None = None,
    end: str | None = None,
    fit_c2: bool = True,
    fit_rf_mult: bool = False,
) -> dict:
    """OLS on y = r_etf − L·r_idx + ER·days/365 against financing (and r²) terms."""
    f = _features(idx_close, rf)
    y = etf_close.reindex(f.index).pct_change()
    df = f.assign(y=y).loc[start:end].dropna()
    resid = df["y"] - lev * df["r"] + er * df["days"] / 365.0
    cols = {"spread": -(lev - 1) * df["days"] / 365.0}
    if fit_rf_mult:
        cols["rf_mult"] = -df["rf"] * df["days"] / 365.0
    else:
        resid = resid + (lev - 1) * df["rf"] * df["days"] / 365.0
    if fit_c2:
        cols["c2"] = df["r"] ** 2
    X = pd.DataFrame(cols)
    beta, *_ = np.linalg.lstsq(X.values, resid.values, rcond=None)
    out = {"er": er, "spread": 0.0, "c2": 0.0}
    out.update(dict(zip(X.columns, map(float, beta))))
    return out


def calibrate_drift(
    idx: pd.DataFrame,
    etf_close: pd.Series,
    rf: pd.Series,
    *,
    lev: float,
    er: float,
    start: str | None = None,
    end: str | None = None,
    fit_c2: bool = True,
) -> dict:
    """c2 from daily OLS (vol drag), then spread so that cumulative synth == real."""
    c2 = 0.0
    if fit_c2:
        c2 = calibrate(idx["Close"], etf_close, rf, lev=lev, er=er, start=start, end=end)["c2"]
    lo, hi = -0.05, 0.05
    for _ in range(60):
        mid = (lo + hi) / 2
        s = synth(idx, rf, lev=lev, er=er, spread=mid, c2=c2)["Close"]
        d = tracking_report(s, etf_close, start, end)["drift_ann"]
        if d > 0:
            lo = mid
        else:
            hi = mid
    return {"er": er, "spread": (lo + hi) / 2, "c2": c2}


def load_tr_index(
    price_ticker: str,
    tr_etf: str,
    tr_etf_er: float,
    *,
    pre_div_yield: float = 0.0,
    start: str = "1985-01-01",
) -> tuple[pd.DataFrame, pd.Series]:
    """Total-return index: ETF adjusted close + its ER where available,
    else price index + pre_div_yield. Opens from the price index overnight gap."""
    px, rf = load_inputs(price_ticker, start)
    etf = _dl(tr_etf, start)["Close"].astype(float).reindex(px.index)
    days = px.index.to_series().diff().dt.days.fillna(1).astype(float)
    r_px = px["Close"].pct_change() + pre_div_yield * days / 365.0
    r_etf = etf.pct_change() + tr_etf_er * days / 365.0
    r = r_etf.where(r_etf.notna() & etf.shift(1).notna(), r_px).fillna(0.0)
    close = 100 * (1 + r).cumprod()
    gap = (px["Open"] / px["Close"].shift(1)).fillna(1.0)
    hi = (px["High"] / px["Close"]).fillna(1.0)
    lo = (px["Low"] / px["Close"]).fillna(1.0)
    out = pd.DataFrame({
        "Open": close.shift(1) * gap,
        "High": close * hi,
        "Low": close * lo,
        "Close": close,
    })
    out.iloc[0, 0] = close.iloc[0]
    return out, rf


def tracking_report(syn_close: pd.Series, etf_close: pd.Series, start=None, end=None) -> dict:
    s = syn_close.loc[start:end]
    e = etf_close.reindex(s.index).dropna()
    s = s.reindex(e.index)
    rs, re = s.pct_change().dropna(), e.pct_change().dropna()
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    ratio = (s.iloc[-1] / s.iloc[0]) / (e.iloc[-1] / e.iloc[0])
    path = np.log((s / s.iloc[0]) / (e / e.iloc[0]))
    return {
        "te_ann": float((rs - re).std() * np.sqrt(252)),
        "drift_ann": float(ratio ** (1 / yrs) - 1),
        "terminal_ratio": float(ratio),
        "max_path_dev": float(path.abs().max()),
        "corr": float(np.corrcoef(rs, re.reindex(rs.index))[0, 1]),
    }


def splice(syn: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    """Synthetic before first real bar, real (rescaled to synth level) after."""
    real = real[["Open", "Close"]].astype(float).dropna()
    t0 = real.index[0]
    k = float(syn["Close"].loc[t0]) / float(real["Close"].iloc[0])
    out = syn.copy()
    cm = real.index.intersection(out.index)
    out.loc[cm, "Close"] = real.loc[cm, "Close"] * k
    out.loc[cm, "Open"] = real.loc[cm, "Open"] * k
    return out.ffill()


def build_preset(name: str, start: str = "1985-01-01") -> dict:
    """Return signal OHLC (price index), TR index, rf, synthetic and spliced vehicle."""
    p = PRESETS[name]
    px, rf = load_inputs(p["index"], start)
    tri, _ = load_tr_index(p["index"], p["tr_etf"], p["tr_er"], pre_div_yield=p["pre_div"], start=start)
    syn = synth(tri, rf, lev=p["lev"], er=p["er"], spread=p["spread"], c2=p["c2"])
    real = _dl(p["real"], start)[["Open", "Close"]].astype(float).dropna()
    return {"signal": px, "tr": tri, "rf": rf, "synth": syn, "vehicle": splice(syn, real), "real": real}


def _report(name: str = "TQQQ") -> None:
    """Compare synthesis models on the real-ETF overlap (train / test / full)."""
    p = PRESETS[name]
    lev, er = p["lev"], p["er"]
    px, rf = load_inputs(p["index"])
    tri, _ = load_tr_index(p["index"], p["tr_etf"], p["tr_er"], pre_div_yield=p["pre_div"])
    real = _dl(p["real"], "1990-01-01")["Close"].astype(float).dropna()
    t0 = str(real.index[0].date())
    split = str((real.index[0] + (real.index[-1] - real.index[0]) / 2).date())
    tr, te_ = (t0, split), (split, None)

    def legacy(base):
        r = lev * base["Close"].pct_change().fillna(0) - 1e-4
        return pd.DataFrame({"Close": 100 * (1 + r).cumprod()})

    models = [
        ("A 가격지수 −1bp/일 (구)", lambda: legacy(px), "-"),
        ("B 가격지수 ER+(L−1)×금리", lambda: synth(px, rf, lev=lev, er=er), "-"),
        ("B' TR ER+(L−1)×금리", lambda: synth(tri, rf, lev=lev, er=er), "-"),
    ]
    g_tr = calibrate_drift(tri, real, rf, lev=lev, er=er, start=tr[0], end=tr[1])
    g_all = calibrate_drift(tri, real, rf, lev=lev, er=er, start=t0)
    models += [
        ("G TR+스프레드+r² (학습만 보정)", lambda: synth(tri, rf, lev=lev, **g_tr), g_tr),
        ("G* TR+스프레드+r² (전체 보정)", lambda: synth(tri, rf, lev=lev, **g_all), g_all),
    ]
    print(f"{name} 합성 보정 | 학습 {tr[0]}~{tr[1]} | 검증 {te_[0]}~")
    print(f"{'모델':<30}{'학습drift':>10}{'검증drift':>10}{'검증최대괴리':>12}{'전체drift':>10}{'TE':>7}  파라미터")
    for nm, fn, prm in models:
        s = fn()["Close"]
        a = tracking_report(s, real, *tr)
        b = tracking_report(s, real, *te_)
        f = tracking_report(s, real, t0, None)
        ptxt = prm if isinstance(prm, str) else ", ".join(f"{k}={v:.4g}" for k, v in prm.items())
        print(f"{nm:<30}{a['drift_ann']:>+10.2%}{b['drift_ann']:>+10.2%}{b['max_path_dev']:>12.1%}"
              f"{f['drift_ann']:>+10.2%}{f['te_ann']:>7.2%}  {ptxt}")


if __name__ == "__main__":
    _report()
