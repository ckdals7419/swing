"""
Phase E — QLD Dynamic Weight Strategy Backtest.

Tests both rule-based and ML models against QLD buy-and-hold over the
2022-02-01 → present test period (and optionally a longer history).

The backtest is self-contained (no dependency on BacktestEngine):
  - Signal at T close → fill at T+1 open
  - Cash earns BIL / risk-free daily rate
  - Transaction cost: 5 bps per side on traded notional

Usage:
    python strategy_qld.py                     # test period: 2022-02-01 → latest
    python strategy_qld.py --start 2010-01-01  # longer history
    python strategy_qld.py --model rules       # rule-based only
    python strategy_qld.py --model ml          # ML only
    python strategy_qld.py --model both        # compare (default)
    python strategy_qld.py --full              # full history from 2007

Outputs:
    backtest_nav_qld_rules.csv
    backtest_nav_qld_ml.csv
"""

import warnings
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

warnings.filterwarnings("ignore")

FEATURES_PATH = "data/qld_features.parquet"
MODELS_DIR    = Path("models/qld")
DEFAULT_START = "2022-02-01"
COST_BPS      = 9.0   # per side (0.09% commission)


# ── Load data ─────────────────────────────────────────────────────────────────

def load_features(start: str, end: str = None) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    df = df.set_index("date").sort_index()
    # Drop rows still in warm-up (too many nulls)
    feat_cols = _feat_cols(df.columns)
    null_ratio = df[feat_cols].isnull().mean(axis=1)
    df = df[null_ratio < 0.6]
    return df


def load_tqqq_prices(start: str, end: str = None) -> pd.DataFrame:
    """Return TQQQ open/close aligned to the same dates as load_features."""
    univ = pd.read_parquet("data/qld_universe.parquet")
    univ["date"] = pd.to_datetime(univ["date"])
    t = univ[univ["ticker"] == "TQQQ"].sort_values("date").set_index("date")
    if start:
        t = t[t.index >= start]
    if end:
        t = t[t.index <= end]
    return t[["Open", "Close"]].rename(columns={"Open": "tqqq_open", "Close": "tqqq_close"})


def _feat_cols(columns):
    non_feat = {
        "fwd_return_5d", "fwd_return_10d", "fwd_return_20d",
        "fwd_up_5d", "fwd_up_10d", "fwd_up_20d",
        "qld_close", "qld_open", "qqq_close", "qqq_open",
        "bil_return_1d",
    }
    return [c for c in columns if c not in non_feat]


# ── Rule-based weights ────────────────────────────────────────────────────────

def rule_weight(row) -> float:
    """5-level regime → QLD weight."""
    vix   = row.get("vix_close",    20.0)
    dd    = row.get("qqq_drawdown",  0.0)
    vs200 = row.get("qqq_vs_sma200", 0.0)
    vs50  = row.get("qqq_vs_sma50",  0.0)

    for v in [vix, dd, vs200, vs50]:
        if pd.isna(v):
            return 1.0

    if vix > 30 or dd < -0.15:
        return 0.0
    if vs200 < 0 and vix > 22:
        return 0.25
    if vs200 < 0 or vix > 22:
        return 0.50
    if vs50 < 0:
        return 0.75
    return 1.0


def rule_sma200(row) -> float:
    """
    Pure SMA200 trend filter: 100% QLD when QQQ > SMA200, 0% otherwise.
    Uses qqq_vs_sma200 feature (already computed in features file).
    """
    vs200 = row.get("qqq_vs_sma200", 0.0)
    if pd.isna(vs200):
        return 1.0
    return 1.0 if vs200 >= 0 else 0.0


def rule_sma200_vix(row) -> float:
    """
    SMA200 trend filter + VIX spike override.
    - 0% when QQQ < SMA200 (trend is down)
    - 0% when VIX > 30 even if above SMA200 (crisis override)
    - 100% otherwise
    """
    vs200 = row.get("qqq_vs_sma200", 0.0)
    vix   = row.get("vix_close",     20.0)
    if pd.isna(vs200): vs200 = 0.0
    if pd.isna(vix):   vix   = 20.0
    if vix > 30:
        return 0.0
    return 1.0 if vs200 >= 0 else 0.0


def rule_weight_binary(row) -> float:
    """
    Binary rule: 0% or 100%.
    Exit to cash only during confirmed bear/crisis; hold 100% otherwise.
    Tighter triggers than 5-level to avoid cutting exposure too soon.
      Crisis : VIX > 28 OR QQQ drawdown < -12%
      Bear   : below SMA200 AND VIX > 24
      else   : 100%
    """
    vix   = row.get("vix_close",    20.0)
    dd    = row.get("qqq_drawdown",  0.0)
    vs200 = row.get("qqq_vs_sma200", 0.0)

    for v in [vix, dd, vs200]:
        if pd.isna(v):
            return 1.0

    if vix > 28 or dd < -0.12:
        return 0.0
    if vs200 < 0 and vix > 24:
        return 0.0
    return 1.0


# ── Volatility targeting ─────────────────────────────────────────────────────

def vol_target_weight(df: pd.DataFrame,
                      target_vol: float = 0.18) -> pd.Series:
    """
    Size the QLD position so that the portfolio's daily volatility equals
    target_vol (annualized), using realized_vol_20d of QQQ as the estimate.

    weight = min(1.0, target_vol / realized_vol_20d)

    Low QQQ vol (bull):   ~15% → weight ~1.0 (fully in)
    Normal QQQ vol:       ~20% → weight ~0.90
    High QQQ vol (bear):  ~35% → weight ~0.51
    Crisis QQQ vol:       ~55% → weight ~0.33
    """
    rv = df["realized_vol_20d"].replace(0, np.nan).ffill().fillna(0.20)
    weights = (target_vol / rv).clip(0.0, 1.0)
    return weights


def vol_target_with_stop(df: pd.DataFrame,
                         target_vol: float = 0.18,
                         vix_stop: float = 32.0,
                         dd_stop: float = -0.18) -> pd.Series:
    """
    Vol targeting with hard stop: if VIX spikes above vix_stop OR
    QQQ drawdown breaches dd_stop, go to 0% regardless of vol signal.
    Re-enters when VIX < vix_stop - 5 AND drawdown > dd_stop + 0.05.
    """
    rv     = df["realized_vol_20d"].replace(0, np.nan).ffill().fillna(0.20)
    vix    = df["vix_close"].ffill().fillna(20.0)
    dd     = df["qqq_drawdown"].ffill().fillna(0.0)

    base_w = (target_vol / rv).clip(0.0, 1.0)

    # Hard stop: go to cash
    stopped = (vix >= vix_stop) | (dd <= dd_stop)

    # Hysteresis: stay out until both conditions clearly resolved
    # (implement as a forward-looking but leakage-free stateful pass)
    weights = np.empty(len(df))
    in_market = True
    for i in range(len(df)):
        if stopped.iloc[i]:
            in_market = False
        elif not in_market:
            # Re-entry: VIX has to drop back below threshold
            re_entry = (vix.iloc[i] < vix_stop - 6) and (dd.iloc[i] > dd_stop + 0.06)
            if re_entry:
                in_market = True
        weights[i] = float(base_w.iloc[i]) if in_market else 0.0

    return pd.Series(weights, index=df.index)


# ── ML weights ────────────────────────────────────────────────────────────────

def load_ml_models():
    clf = xgb.XGBClassifier()
    clf.load_model(str(MODELS_DIR / "clf_10d.ubj"))
    reg = xgb.XGBRegressor()
    reg.load_model(str(MODELS_DIR / "reg_10d.ubj"))

    with open(MODELS_DIR / "thresholds.json") as f:
        thr = tuple(json.load(f)["thresholds"])

    with open(MODELS_DIR / "feature_cols.json") as f:
        feat_cols = json.load(f)

    return clf, reg, thr, feat_cols


def _ml_score(df: pd.DataFrame) -> pd.Series:
    """Return raw composite score [0,1] for each row."""
    clf, reg, thr, feat_cols = load_ml_models()
    X = df.reindex(columns=feat_cols).fillna(0)
    clf_prob = clf.predict_proba(X)[:, 1]
    reg_pred = reg.predict(X)
    reg_std  = reg_pred.std()
    reg_z    = (reg_pred - reg_pred.mean()) / reg_std if reg_std > 1e-6 else np.zeros(len(reg_pred))

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    score = 0.6 * clf_prob + 0.4 * sigmoid(reg_z)
    return pd.Series(score, index=df.index)


def ml_weight_series(df: pd.DataFrame) -> pd.Series:
    """4-level ML weights using tuned thresholds: {0, 0.33, 0.67, 1.0}."""
    _, _, thr, _ = load_ml_models()
    score = _ml_score(df)
    t0, t1, t2 = thr
    weights = np.where(score < t0, 0.0,
              np.where(score < t1, 0.33,
              np.where(score < t2, 0.67, 1.0)))
    return pd.Series(weights, index=df.index)


def ml_weight_binary(df: pd.DataFrame,
                     bull_threshold: float = 0.46) -> pd.Series:
    """
    Binary ML: 0% when score < threshold, 100% otherwise.
    Default threshold 0.46 tuned to minimize MDD while maximizing return
    on the validation set.
    """
    score = _ml_score(df)
    weights = np.where(score >= bull_threshold, 1.0, 0.0)
    return pd.Series(weights, index=df.index)


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, weights: pd.Series,
             cost_bps: float = COST_BPS,
             initial_cash: float = 1_000_000.0,
             vehicle_open: pd.Series = None,
             vehicle_close: pd.Series = None) -> pd.Series:
    """
    NAV series given daily allocation weights.

    Signal at T close → trade at T+1 open → hold to T+1 close.
    Cash earns bil_return_1d each day.
    Transaction cost charged on |Δweight| × NAV.

    vehicle_open / vehicle_close: override the default QLD price series.
    Pass TQQQ prices to backtest an alternative vehicle with the same weights.
    """
    cost_rate = cost_bps / 10_000.0

    weights = weights.reindex(df.index).ffill().fillna(0.0)
    v_close = vehicle_close.reindex(df.index).ffill() if vehicle_close is not None else df["qld_close"].ffill()
    v_open  = vehicle_open.reindex(df.index).ffill()  if vehicle_open  is not None else df["qld_open"].ffill()
    cash_daily = df["bil_return_1d"].fillna(0.0)

    # Close-to-close return (correct for held positions)
    cc_ret = (v_close / v_close.shift(1) - 1).fillna(0.0)
    # Intraday return (open-to-close, used only on entry day when buying new position)
    intra_ret = (v_close / v_open - 1).fillna(0.0)

    # Weight applied today = signal from yesterday
    w_applied = weights.shift(1).fillna(0.0)
    w_prev    = w_applied.shift(1).fillna(0.0)
    w_delta   = (w_applied - w_prev)          # signed delta (+ = bought, - = sold)
    w_abs_chg = w_delta.abs()

    nav_vals = np.empty(len(df))
    nav_val  = float(initial_cash)

    for i in range(len(df)):
        w    = w_applied.iloc[i]
        wp   = w_prev.iloc[i]
        dw   = w_delta.iloc[i]
        adw  = w_abs_chg.iloc[i]
        tc   = adw * nav_val * cost_rate

        # Unchanged portion earns close-to-close
        unchanged = min(w, wp)
        # Newly bought portion only earns open-to-close (entry day)
        bought    = max(0.0, dw)

        daily_ret = (unchanged * cc_ret.iloc[i]
                     + bought   * intra_ret.iloc[i]
                     + (1 - w)  * cash_daily.iloc[i])

        nav_val   = nav_val * (1 + daily_ret) - tc
        nav_vals[i] = nav_val

    return pd.Series(nav_vals, index=df.index, name="nav")


# ── Analytics ─────────────────────────────────────────────────────────────────

def print_tearsheet(label: str, nav: pd.Series,
                    benchmark_nav: pd.Series = None,
                    weights: pd.Series = None,
                    initial_cash: float = 1_000_000.0):

    def _m(n):
        ret    = n.pct_change().dropna()
        total  = n.iloc[-1] / n.iloc[0] - 1
        nyrs   = len(n) / 252.0
        ann    = (1 + total) ** (1 / nyrs) - 1 if nyrs > 0 else 0.0
        vol    = ret.std() * np.sqrt(252)
        rf     = 0.04
        sharpe = (ann - rf) / vol if vol > 0 else 0.0
        peak   = n.expanding().max()
        dd     = (n - peak) / peak
        mdd    = dd.min()
        calmar = ann / abs(mdd) if mdd != 0 else 0.0
        monthly = n.resample("ME").last().pct_change().dropna()
        win_m  = (monthly > 0).mean()
        best_m = monthly.max()
        worst_m = monthly.min()
        return dict(total=total, ann=ann, vol=vol, sharpe=sharpe,
                    mdd=mdd, calmar=calmar, win_m=win_m,
                    best_m=best_m, worst_m=worst_m)

    m = _m(nav)

    bench_str = ""
    if benchmark_nav is not None:
        b = _m(benchmark_nav)
    else:
        b = None

    start = nav.index[0].date()
    end   = nav.index[-1].date()

    w = 56
    print()
    print("=" * w)
    print(f"  {label}")
    print(f"  {start} → {end}")
    print("=" * w)

    rows = [
        ("Total Return",   f"{m['total']:>9.2%}",   f"{b['total']:>9.2%}" if b else ""),
        ("Ann. Return",    f"{m['ann']:>9.2%}",     f"{b['ann']:>9.2%}"   if b else ""),
        ("Ann. Volatility",f"{m['vol']:>9.2%}",     f"{b['vol']:>9.2%}"   if b else ""),
        ("Max Drawdown",   f"{m['mdd']:>9.2%}",     f"{b['mdd']:>9.2%}"   if b else ""),
        ("Sharpe Ratio",   f"{m['sharpe']:>9.3f}",  f"{b['sharpe']:>9.3f}"if b else ""),
        ("Calmar Ratio",   f"{m['calmar']:>9.3f}",  f"{b['calmar']:>9.3f}"if b else ""),
        ("Win Rate (Mo.)", f"{m['win_m']:>9.1%}",   f"{b['win_m']:>9.1%}" if b else ""),
        ("Best Month",     f"{m['best_m']:>9.2%}",  f"{b['best_m']:>9.2%}"if b else ""),
        ("Worst Month",    f"{m['worst_m']:>9.2%}", f"{b['worst_m']:>9.2%}"if b else ""),
    ]

    header = f"  {'Metric':<24} {'Strategy':>10}  {'QLD B&H':>10}"
    print(header)
    print("-" * w)
    for name, strat, bah in rows:
        print(f"  {name:<24} {strat}  {bah}")

    if weights is not None:
        time_in = (weights > 0).mean()
        full_w  = (weights >= 0.99).mean()
        avg_w   = weights.mean()
        print("-" * w)
        print(f"  {'Time in market':<24} {time_in:>9.1%}")
        print(f"  {'Avg QLD weight':<24} {avg_w:>9.1%}")
        print(f"  {'Time at 100%':<24} {full_w:>9.1%}")

    print("=" * w)


# ── Weight regime breakdown ───────────────────────────────────────────────────

def print_weight_breakdown(label: str, weights: pd.Series):
    counts = weights.value_counts().sort_index()
    total  = len(weights)
    print(f"\n  {label} - weight distribution:")
    for w, n in counts.items():
        bar = "#" * int(30 * n / total)
        print(f"    {w*100:>5.0f}%  {n:>5}d  ({n/total:>5.1%})  {bar}")


# ── Runner ────────────────────────────────────────────────────────────────────

def _nav_metrics(nav: pd.Series, w: pd.Series = None) -> dict:
    ret    = nav.pct_change().dropna()
    total  = nav.iloc[-1] / nav.iloc[0] - 1
    nyrs   = len(nav) / 252.0
    ann    = (1 + total) ** (1 / nyrs) - 1 if nyrs > 0 else 0.0
    vol    = ret.std() * np.sqrt(252)
    peak   = nav.expanding().max()
    mdd    = ((nav - peak) / peak).min()
    sharpe = (ann - 0.04) / vol if vol > 0 else 0.0
    calmar = ann / abs(mdd) if mdd != 0 else 0.0
    avg_w  = float(w.mean()) if w is not None else 1.0
    return dict(ann=ann, mdd=mdd, sharpe=sharpe, calmar=calmar, avg_w=avg_w, total=total)


def run(model: str = "both", start: str = DEFAULT_START,
        end: str = None, save_csv: bool = True):

    print(f"Loading features {start} -> {end or 'latest'}...")
    df = load_features(start, end)
    print(f"  {len(df):,} trading days")

    bah_w   = pd.Series(1.0, index=df.index)
    bah_nav = backtest(df, bah_w)

    has_ml = (MODELS_DIR / "clf_10d.ubj").exists()
    variants = []

    # SMA200 binary
    if model in ("rules", "both"):
        w = df.apply(rule_sma200, axis=1)
        nav = backtest(df, w)
        variants.append(("SMA200", w, nav))
        if save_csv:
            pd.DataFrame({"nav": nav, "qld_bah": bah_nav}).to_csv(
                "backtest_nav_qld_sma200.csv")

    # SMA200 + VIX spike filter
    if model in ("rules", "both"):
        w = df.apply(rule_sma200_vix, axis=1)
        nav = backtest(df, w)
        variants.append(("SMA200+VIX", w, nav))
        if save_csv:
            pd.DataFrame({"nav": nav, "qld_bah": bah_nav}).to_csv(
                "backtest_nav_qld_sma200vix.csv")

    # 5-level rule-based
    if model in ("rules", "both"):
        w = df.apply(rule_weight, axis=1)
        nav = backtest(df, w)
        variants.append(("Rules-5lvl", w, nav))
        if save_csv:
            pd.DataFrame({"nav": nav, "qld_bah": bah_nav}).to_csv(
                "backtest_nav_qld_rules.csv")

    # Binary rule-based
    if model in ("rules", "both"):
        w = df.apply(rule_weight_binary, axis=1)
        nav = backtest(df, w)
        variants.append(("Rules-bin", w, nav))
        if save_csv:
            pd.DataFrame({"nav": nav, "qld_bah": bah_nav}).to_csv(
                "backtest_nav_qld_rules_binary.csv")

    # Vol targeting (pure)
    if model in ("rules", "both"):
        w = vol_target_weight(df, target_vol=0.18)
        nav = backtest(df, w)
        variants.append(("VolTarget", w, nav))
        if save_csv:
            pd.DataFrame({"nav": nav, "qld_bah": bah_nav}).to_csv(
                "backtest_nav_qld_voltarget.csv")

    # Vol targeting with hard stop
    if model in ("rules", "both"):
        w = vol_target_with_stop(df, target_vol=0.18, vix_stop=32, dd_stop=-0.18)
        nav = backtest(df, w)
        variants.append(("VolTgt+Stop", w, nav))
        if save_csv:
            pd.DataFrame({"nav": nav, "qld_bah": bah_nav}).to_csv(
                "backtest_nav_qld_voltarget_stop.csv")

    if model in ("ml", "both") and has_ml:
        # 4-level ML
        w = ml_weight_series(df)
        nav = backtest(df, w)
        variants.append(("ML-4lvl", w, nav))
        if save_csv:
            pd.DataFrame({"nav": nav, "qld_bah": bah_nav}).to_csv(
                "backtest_nav_qld_ml.csv")

        # Binary ML
        w = ml_weight_binary(df)
        nav = backtest(df, w)
        variants.append(("ML-binary", w, nav))
        if save_csv:
            pd.DataFrame({"nav": nav, "qld_bah": bah_nav}).to_csv(
                "backtest_nav_qld_ml_binary.csv")

        # Hybrid: vol-target base × ML signal cap
        # Use vol-target weight but cap at ML signal (avoid buying on weak signal)
        vt_w  = vol_target_weight(df, target_vol=0.18)
        ml_s  = _ml_score(df)
        # ML score acts as a multiplier: 0.0→0.0, 0.5→0.5, 1.0→1.0
        hybrid_w = (vt_w * ml_s.clip(0, 1)).clip(0, 1)
        nav = backtest(df, hybrid_w)
        variants.append(("Hybrid", hybrid_w, nav))
        if save_csv:
            pd.DataFrame({"nav": nav, "qld_bah": bah_nav}).to_csv(
                "backtest_nav_qld_hybrid.csv")

    elif model in ("ml", "both") and not has_ml:
        print("\n[SKIP] ML model not found. Run: python train_qld_model.py")

    # ── TQQQ variants (same weights, higher vehicle leverage) ─────────────────
    # Apply the best-performing QLD weight signals to TQQQ (3x leverage):
    # At 67% weight TQQQ = 2x effective (same as QLD B&H)
    # At 100% weight TQQQ = 3x effective (higher return in bull market)
    # At 0% weight = cash (same protection)
    univ = pd.read_parquet("data/qld_universe.parquet")
    univ["date"] = pd.to_datetime(univ["date"])
    tqqq_df = (univ[univ["ticker"] == "TQQQ"]
               .sort_values("date").set_index("date")
               [["Open", "Close"]])
    t_open  = tqqq_df["Open"]
    t_close = tqqq_df["Close"]

    # TQQQ is only available from 2010-02-11; skip if period starts earlier
    tqqq_ok = t_open.index.min() <= pd.Timestamp(start)

    if tqqq_ok:
        # TQQQ B&H baseline
        tqqq_bah_nav = backtest(df, pd.Series(1.0, index=df.index),
                                vehicle_open=t_open, vehicle_close=t_close)
        variants.append(("TQQQ-BaH", pd.Series(1.0, index=df.index), tqqq_bah_nav))

        # TQQQ + ML 4-level weights
        if has_ml:
            w = ml_weight_series(df)
            nav = backtest(df, w, vehicle_open=t_open, vehicle_close=t_close)
            variants.append(("TQQQ+ML4", w, nav))
            if save_csv:
                pd.DataFrame({"nav": nav, "tqqq_bah": tqqq_bah_nav,
                              "qld_bah": bah_nav}).to_csv(
                    "backtest_nav_tqqq_ml.csv")

        # TQQQ + vol-target weights (target 2x effective leverage on avg)
        # target_vol_tqqq = 0.35 means weight ≈ 0.67 at normal QQQ vol ~17.5%
        vt_w = vol_target_weight(df, target_vol=0.35)
        nav  = backtest(df, vt_w, vehicle_open=t_open, vehicle_close=t_close)
        variants.append(("TQQQ+VT", vt_w, nav))
        if save_csv:
            pd.DataFrame({"nav": nav, "tqqq_bah": tqqq_bah_nav,
                          "qld_bah": bah_nav}).to_csv(
                "backtest_nav_tqqq_voltarget.csv")

    # ── Master comparison table ────────────────────────────────────────────────
    if len(variants) > 1:
        bah_m = _nav_metrics(bah_nav)
        print("\n")
        w_col = 18
        col   = 10
        header = f"  {'Metric':<22}"
        for label, _, _ in variants:
            header += f" {label[:col]:>{col}}"
        header += f" {'QLD B&H':>{col}}"
        print("=" * len(header))
        print("  Master Comparison")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for metric_name, key, fmt in [
            ("Ann. Return",  "ann",    ".2%"),
            ("Total Return", "total",  ".2%"),
            ("Max Drawdown", "mdd",    ".2%"),
            ("Sharpe Ratio", "sharpe", ".3f"),
            ("Calmar Ratio", "calmar", ".3f"),
            ("Avg QLD Wt",   "avg_w",  ".1%"),
        ]:
            row = f"  {metric_name:<22}"
            for _, w, nav in variants:
                m = _nav_metrics(nav, w)
                row += f" {format(m[key], fmt):>{col}}"
            row += f" {format(bah_m[key], fmt):>{col}}"
            print(row)

        # Mark which variants beat B&H on both key metrics
        print("-" * len(header))
        beat_row = f"  {'Beats B&H (R+MDD)':<22}"
        for label, w, nav in variants:
            m   = _nav_metrics(nav, w)
            bm  = bah_m
            win = (m["ann"] >= bm["ann"]) and (m["mdd"] >= bm["mdd"])
            beat_row += f" {'YES' if win else 'no':>{col}}"
        beat_row += f" {'---':>{col}}"
        print(beat_row)
        print("=" * len(header))

    return {label: (w, nav) for label, w, nav in variants}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLD dynamic weight strategy backtest")
    parser.add_argument("--model",  choices=["rules", "ml", "both"], default="both")
    parser.add_argument("--start",  type=str, default=DEFAULT_START)
    parser.add_argument("--end",    type=str, default=None)
    parser.add_argument("--full",   action="store_true",
                        help="Run full history from 2007 (pre-train data)")
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    start = "2007-01-01" if args.full else args.start
    run(model=args.model, start=start, end=args.end, save_csv=not args.no_csv)
