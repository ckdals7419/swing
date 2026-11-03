"""
Multi-asset rotation strategy using ML spread signal.
Instead of binary QLD/cash, allocates across:
  TQQQ (3x)  - very bullish
  QLD  (2x)  - bullish
  QQQ  (1x)  - mild bull
  SCHD       - defensive equity
  SGOV/BIL   - treasuries (risk-off)

Walk-forward annual retraining, 2015-2026.
"""
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from pathlib import Path

warnings.filterwarnings("ignore")

from crossval_qld import (
    build_full_dataset,
    ML_FEATURES_BASE, CROSS_ASSET_ADDITIONS, REGIME_ADDITIONS,
    BTC_XGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOP_ROUNDS,
    w_sma200, backtest, compute_metrics, COST_BPS,
)

FEAT_LIST   = ML_FEATURES_BASE + CROSS_ASSET_ADDITIONS + REGIME_ADDITIONS
CHART_START = "2015-01-01"
ROTATION_CACHE = Path("data/rotation_assets.parquet")

# Walk-forward retraining schedule
RETRAIN_SCHEDULE = [
    ("2015-01-01", "2014-11-30"),
    ("2016-01-01", "2015-11-30"),
    ("2017-01-01", "2016-11-30"),
    ("2018-01-01", "2017-11-30"),
    ("2019-01-01", "2018-11-30"),
    ("2020-01-01", "2019-11-30"),
    ("2021-01-01", "2020-11-30"),
    ("2022-01-01", "2021-11-30"),
    ("2023-01-01", "2022-11-30"),
    ("2024-01-01", "2023-11-30"),
    ("2025-01-01", "2024-11-30"),
    ("2025-07-01", "2025-05-31"),
    ("2026-01-01", "2025-11-30"),
]

# ── Allocation bands: spread -> asset ──────────────────────────────────────
# Spread is bull_prob - bear_prob. Higher = more bullish.
BANDS = [
    # (spread_min, asset_name, colour, label)
    # QLD is the default. Only rotate out on strong conviction.
    ( 0.35,  "TQQQ",  "#ff6622", "TQQQ (3x)"),       # very strong bull
    (-0.10,  "QLD",   "#f0c040", "QLD  (2x)"),        # default: mild bull to mild bear
    (-0.20,  "QQQ",   "#4a9eda", "QQQ  (1x)"),        # moderate bear: de-lever
    (-0.30,  "SCHD",  "#88bb44", "SCHD (div)"),        # bearish: defensive equity
    (-9.99,  "SGOV",  "#aa66cc", "SGOV (T-bill)"),     # strong bear: risk-off
]

ASSET_COLORS = {b[1]: b[2] for b in BANDS}
ASSET_LABELS = {b[1]: b[3] for b in BANDS}


def get_asset(spread_val: float) -> str:
    for thresh, name, _, _ in BANDS:
        if spread_val >= thresh:
            return name
    return "SGOV"


# ── Asset data download ────────────────────────────────────────────────────

def download_rotation_assets(full_df, force=False):
    """
    Download/build daily close for TQQQ, QQQ, QLD, SCHD, SGOV.
    TQQQ is synthetic (3x QQQ) before inception.
    SGOV uses BIL before inception.
    Returns DataFrame with daily RETURNS (not prices).
    """
    if ROTATION_CACHE.exists() and not force:
        df = pd.read_parquet(ROTATION_CACHE)
        df = df.reindex(full_df.index)
        print(f"Rotation asset cache loaded ({len(df.columns)} assets)")
        return df

    print("Downloading rotation assets...")
    idx = full_df.index

    def _dl(ticker, start):
        raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        raw.index = pd.to_datetime(raw.index)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        if hasattr(raw.index, "tz") and raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)
        return raw["Close"].reindex(idx).ffill()

    qqq_close  = full_df["qqq_close"]
    qld_close  = full_df["qld_close"]
    qqq_ret1   = qqq_close.pct_change().fillna(0)

    # TQQQ: real from 2010-02, synthetic 3x before that
    tqqq_real = _dl("TQQQ", "2010-02-01")
    tqqq_inception = tqqq_real.first_valid_index()
    # Synthetic: 3x daily QQQ return, anchored to real TQQQ at inception
    synth_3x = (1 + qqq_ret1 * 3.0).cumprod()
    if tqqq_inception and tqqq_inception in synth_3x.index:
        scale = float(tqqq_real.loc[tqqq_inception]) / float(synth_3x.loc[tqqq_inception])
        synth_3x = synth_3x * scale
    tqqq_close = pd.concat([
        synth_3x[synth_3x.index < tqqq_inception],
        tqqq_real[tqqq_real.index >= tqqq_inception].dropna()
    ]).sort_index().reindex(idx).ffill()

    # SCHD: real from 2011-10
    schd_close = _dl("SCHD", "2011-10-01")
    # Before inception, use DVY as proxy
    dvy_close = _dl("DVY", "2003-11-01")
    schd_inception = schd_close.first_valid_index()
    if schd_inception:
        # Scale DVY to match SCHD at inception
        scale = float(schd_close.loc[schd_inception]) / float(dvy_close.loc[schd_inception])
        dvy_scaled = dvy_close * scale
        schd_close = pd.concat([
            dvy_scaled[dvy_scaled.index < schd_inception],
            schd_close[schd_close.index >= schd_inception].dropna()
        ]).sort_index().reindex(idx).ffill()

    # SGOV: real from 2020-06, BIL before that
    sgov_close = _dl("SGOV", "2020-06-01")
    bil_close  = _dl("BIL",  "2007-05-01")
    sgov_inception = sgov_close.first_valid_index()
    if sgov_inception and sgov_inception in bil_close.index:
        scale = float(sgov_close.loc[sgov_inception]) / float(bil_close.loc[sgov_inception])
        bil_scaled = bil_close * scale
        sgov_close = pd.concat([
            bil_scaled[bil_scaled.index < sgov_inception],
            sgov_close[sgov_close.index >= sgov_inception].dropna()
        ]).sort_index().reindex(idx).ffill()

    # Build daily return DataFrame
    ret = pd.DataFrame(index=idx)
    ret["TQQQ"] = tqqq_close.pct_change().fillna(0)
    ret["QLD"]  = qld_close.pct_change().fillna(0)
    ret["QQQ"]  = qqq_ret1
    ret["SCHD"] = schd_close.pct_change().fillna(0)
    ret["SGOV"] = sgov_close.pct_change().fillna(0)

    # Also store close prices for reference
    ret["TQQQ_close"] = tqqq_close
    ret["QLD_close"]  = qld_close
    ret["QQQ_close"]  = qqq_close
    ret["SCHD_close"] = schd_close
    ret["SGOV_close"] = sgov_close

    ret.to_parquet(ROTATION_CACHE)
    print(f"  Saved rotation assets ({len(ret.columns)} cols)")
    return ret


# ── Training (reuse from chart_full) ────────────────────────────────────────

def train_for_period(full_df, train_end, feat_list):
    feat_cols   = [c for c in feat_list if c in full_df.columns]
    target_cols = ["target_bull", "target_bear"]
    train_df = full_df.loc[:train_end].dropna(subset=feat_cols + target_cols)
    n_val    = max(60, int(len(train_df) * 0.10))
    val_df   = train_df.iloc[-n_val:]
    tr_df    = train_df.iloc[:-n_val]
    X_tr  = tr_df[feat_cols].ffill().fillna(0.0)
    X_val = val_df[feat_cols].ffill().fillna(0.0)
    boosters = {}
    for target in ("target_bull", "target_bear"):
        y_tr  = tr_df[target]
        y_val = val_df[target]
        spw   = float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum()))
        params = {**BTC_XGB_PARAMS, "scale_pos_weight": spw}
        dtrain = xgb.DMatrix(X_tr,  label=y_tr,  feature_names=feat_cols)
        dval   = xgb.DMatrix(X_val, label=y_val, feature_names=feat_cols)
        booster = xgb.train(
            params, dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dval, "eval")],
            early_stopping_rounds=EARLY_STOP_ROUNDS,
            verbose_eval=False,
        )
        boosters[target] = booster
    return boosters["target_bull"], boosters["target_bear"], feat_cols


def compute_spread(df, bull_clf, bear_clf, feat_cols):
    X = df[[c for c in feat_cols if c in df.columns]].copy()
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_cols].ffill().fillna(0.0)
    dmat = xgb.DMatrix(X, feature_names=feat_cols)
    bp   = pd.Series(bull_clf.predict(dmat), index=df.index)
    brp  = pd.Series(bear_clf.predict(dmat), index=df.index)
    return bp, brp, bp - brp


# ── Rotation backtest ──────────────────────────────────────────────────────

def backtest_rotation(spread: pd.Series, asset_returns: pd.DataFrame,
                      cost_bps: float = COST_BPS) -> tuple:
    """
    For each day, pick asset based on spread, compute NAV.
    Signal at close T -> allocation at T+1.
    Returns (nav_series, allocation_series).
    """
    cost_rate = cost_bps / 10_000.0
    dates = spread.index
    nav = np.ones(len(dates)) * 1_000_000.0
    allocs = []

    prev_asset = get_asset(float(spread.iloc[0]))
    allocs.append(prev_asset)

    for i in range(1, len(dates)):
        # Signal from yesterday's close
        today_asset = get_asset(float(spread.iloc[i-1]))
        allocs.append(today_asset)

        # Daily return of chosen asset
        daily_ret = float(asset_returns.loc[dates[i], today_asset])

        # Transaction cost on switch
        tc = cost_rate * nav[i-1] if today_asset != prev_asset else 0.0

        nav[i] = nav[i-1] * (1 + daily_ret) - tc
        prev_asset = today_asset

    return (pd.Series(nav, index=dates, name="nav"),
            pd.Series(allocs, index=dates, name="alloc"))


# ── Friction parameters ───────────────────────────────────────────────────
EMA_SPAN       = 5     # Decay memory: EMA smoothing on raw spread
HYSTERESIS     = 0.05  # Extra margin spread must exceed to trigger switch
MIN_HOLD_DAYS  = 10    # Minimum days before allowing another switch


def get_asset_with_hysteresis(spread_val: float, current_asset: str,
                              hysteresis: float = HYSTERESIS) -> str:
    """
    To switch away from current_asset, spread must move further past
    the band boundary by `hysteresis`.
    - To move UP (more aggressive): spread must exceed upper boundary + hysteresis
    - To move DOWN (more defensive): spread must drop below lower boundary - hysteresis
    If neither condition met, stay in current asset.
    """
    # Find current asset's band boundaries
    asset_names = [b[1] for b in BANDS]
    cur_idx = asset_names.index(current_asset)

    # What would the raw signal say?
    raw_target = get_asset(spread_val)
    raw_idx    = asset_names.index(raw_target)

    if raw_idx == cur_idx:
        return current_asset  # No change needed

    if raw_idx < cur_idx:
        # Trying to move to more aggressive asset (e.g. QLD -> TQQQ)
        # Need spread to exceed the more aggressive band threshold + hysteresis
        target_thresh = BANDS[raw_idx][0]
        if spread_val >= target_thresh + hysteresis:
            return raw_target
        return current_asset
    else:
        # Trying to move to more defensive asset (e.g. QLD -> QQQ)
        # Need spread to drop below current band's lower boundary - hysteresis
        cur_thresh = BANDS[cur_idx][0]
        if spread_val < cur_thresh - hysteresis:
            return raw_target
        return current_asset


def backtest_rotation_friction(spread: pd.Series, asset_returns: pd.DataFrame,
                               cost_bps: float = COST_BPS,
                               ema_span: int = EMA_SPAN,
                               hysteresis: float = HYSTERESIS,
                               min_hold: int = MIN_HOLD_DAYS) -> tuple:
    """
    Rotation backtest with three friction mechanisms:
    1) EMA smoothing on spread (decay memory)
    2) Hysteresis: spread must exceed band boundary by extra margin to switch
    3) Minimum hold period: block switches for N days after a transition
    """
    # 1) Decay memory: EMA-smooth the raw spread
    spread_smooth = spread.ewm(span=ema_span, adjust=False).mean()

    cost_rate = cost_bps / 10_000.0
    dates = spread_smooth.index
    nav = np.ones(len(dates)) * 1_000_000.0
    allocs = []

    cur_asset   = get_asset(float(spread_smooth.iloc[0]))
    hold_counter = min_hold  # Start as if we just switched (enforce initial hold)
    allocs.append(cur_asset)

    for i in range(1, len(dates)):
        sig_val = float(spread_smooth.iloc[i-1])

        # 2) Hysteresis + 3) Min hold
        candidate = get_asset_with_hysteresis(sig_val, cur_asset, hysteresis)
        if candidate != cur_asset and hold_counter >= min_hold:
            # Switch allowed
            today_asset = candidate
            hold_counter = 0
        else:
            today_asset = cur_asset

        hold_counter += 1
        allocs.append(today_asset)

        daily_ret = float(asset_returns.loc[dates[i], today_asset])
        tc = cost_rate * nav[i-1] if today_asset != cur_asset else 0.0
        nav[i] = nav[i-1] * (1 + daily_ret) - tc
        cur_asset = today_asset

    return (pd.Series(nav, index=dates, name="nav"),
            pd.Series(allocs, index=dates, name="alloc"),
            spread_smooth)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    full_df   = build_full_dataset()
    asset_ret = download_rotation_assets(full_df)
    chart_df  = full_df.loc[CHART_START:].copy()
    chart_ret = asset_ret.loc[CHART_START:]

    # ── Walk-forward spread ──────────────────────────────────────────────────
    all_bull = pd.Series(dtype=float)
    all_bear = pd.Series(dtype=float)
    for i, (pred_from, train_through) in enumerate(RETRAIN_SCHEDULE):
        pred_end = RETRAIN_SCHEDULE[i+1][0] if i+1 < len(RETRAIN_SCHEDULE) else None
        seg = (chart_df.loc[pred_from:pred_end].copy()
               if pred_end else chart_df.loc[pred_from:].copy())
        if len(seg) == 0:
            continue
        print(f"  [{i+1:02d}/{len(RETRAIN_SCHEDULE)}] train<={train_through}  "
              f"predict {pred_from} -> {pred_end or 'latest'} ({len(seg)} rows)")
        bull_clf, bear_clf, feat_cols = train_for_period(full_df, train_through, FEAT_LIST)
        bp, brp, _ = compute_spread(seg, bull_clf, bear_clf, feat_cols)
        all_bull = pd.concat([all_bull, bp])
        all_bear = pd.concat([all_bear, brp])

    all_bull = all_bull[~all_bull.index.duplicated(keep="last")]
    all_bear = all_bear[~all_bear.index.duplicated(keep="last")]
    spread   = all_bull - all_bear

    # ── Rotation backtests ───────────────────────────────────────────────────
    nav_rot_raw, alloc_rot_raw = backtest_rotation(spread, chart_ret)
    nav_rot, alloc_rot, spread_smooth = backtest_rotation_friction(
        spread, chart_ret,
        ema_span=EMA_SPAN, hysteresis=HYSTERESIS, min_hold=MIN_HOLD_DAYS,
    )

    # Trade count comparison
    switches_raw  = (alloc_rot_raw != alloc_rot_raw.shift(1)).sum()
    switches_fric = (alloc_rot != alloc_rot.shift(1)).sum()
    print(f"\n  Switches: raw={switches_raw}  friction={switches_fric}  "
          f"(reduction: {1 - switches_fric/max(1,switches_raw):.0%})")

    # ── Comparison strategies ────────────────────────────────────────────────
    # QLD B&H
    nav_bah = backtest(chart_df, pd.Series(1.0, index=chart_df.index))
    # SMA200 on QLD
    sma_w   = w_sma200(chart_df)
    nav_sma = backtest(chart_df, sma_w)
    # ML binary (QLD/cash, buy=0.20, sell=-0.30)
    from chart_full import weights_asym
    w_bin = weights_asym(spread, 0.20, -0.30)
    nav_bin = backtest(chart_df, w_bin)
    # QQQ B&H
    qqq_ret = chart_ret["QQQ"]
    nav_qqq = pd.Series(1_000_000.0, index=chart_df.index)
    for i in range(1, len(nav_qqq)):
        nav_qqq.iloc[i] = nav_qqq.iloc[i-1] * (1 + float(qqq_ret.iloc[i]))

    # Rebase all to 100
    navs = {
        "ML-Rot(friction)": nav_rot / nav_rot.iloc[0] * 100,
        "ML-Rot(raw)":      nav_rot_raw / nav_rot_raw.iloc[0] * 100,
        "ML-Binary":        nav_bin / nav_bin.iloc[0] * 100,
        "SMA200 (QLD)":     nav_sma / nav_sma.iloc[0] * 100,
        "QLD B&H":          nav_bah / nav_bah.iloc[0] * 100,
        "QQQ B&H":          nav_qqq / nav_qqq.iloc[0] * 100,
    }

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics = {}
    for name, nav in navs.items():
        raw = nav / 100 * 1e6
        m = compute_metrics(raw)
        metrics[name] = m

    # ── Allocation stats ─────────────────────────────────────────────────────
    alloc_counts = alloc_rot.value_counts()
    total_days   = len(alloc_rot)

    # ── Plot ─────────────────────────────────────────────────────────────────
    BG   = "#0e1117"
    GRID = "#1e1e2e"
    TXT  = "#cccccc"

    fig, axes = plt.subplots(3, 1, figsize=(20, 14),
                             gridspec_kw={"height_ratios": [3.5, 0.8, 1.2]},
                             sharex=True)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333344")

    # ── Panel 1: NAV ──────────────────────────────────────────────────────────
    ax1 = axes[0]

    nav_styles = [
        ("QLD B&H",          "#444466", 1.0, 0.5, "-"),
        ("QQQ B&H",          "#666688", 1.0, 0.5, "--"),
        ("SMA200 (QLD)",     "#4a9eda", 1.4, 0.8, "-"),
        ("ML-Binary",        "#88bb44", 1.3, 0.7, "--"),
        ("ML-Rot(raw)",      "#cc7744", 1.3, 0.5, ":"),
        ("ML-Rot(friction)", "#f0c040", 2.0, 1.0, "-"),
    ]
    for name, color, lw, alpha, ls in nav_styles:
        m = metrics[name]
        lbl = (f"{name}  Ann={m['ann']:.1%}  MDD={m['mdd']:.1%}  "
               f"Cal={m['calmar']:.2f}")
        ax1.plot(navs[name].index, navs[name].values,
                 color=color, lw=lw, alpha=alpha, ls=ls, label=lbl)

    for yr in range(2016, 2027):
        ax1.axvline(pd.Timestamp(f"{yr}-01-01"), color="#333355", lw=0.5, ls=":", alpha=0.6)

    ax1.set_ylabel("NAV (rebased to 100)", color=TXT, fontsize=10)
    ax1.set_title(
        f"ML Rotation w/ Friction (EMA={EMA_SPAN}, hyst={HYSTERESIS}, hold={MIN_HOLD_DAYS}d)  "
        f"2015-2026   TQQQ/QLD/QQQ/SCHD/SGOV",
        color="#eeeeee", fontsize=13, pad=10
    )
    ax1.legend(loc="upper left", fontsize=7.5,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.9)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax1.grid(axis="y", color=GRID, lw=0.5)

    # ── Panel 2: Allocation strip ─────────────────────────────────────────────
    ax2 = axes[1]

    # Map allocation to numeric for coloring
    asset_names  = [b[1] for b in BANDS]
    asset_colors = [b[2] for b in BANDS]
    asset_map    = {name: i for i, name in enumerate(asset_names)}
    alloc_num    = alloc_rot.map(asset_map).values.astype(float)

    # Draw as colored segments
    dates = alloc_rot.index
    for i in range(1, len(dates)):
        a = alloc_rot.iloc[i]
        ax2.axvspan(dates[i-1], dates[i], alpha=0.8,
                    color=ASSET_COLORS[a], lw=0)

    # Legend patches
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=b[2], label=f"{b[3]}  ({alloc_counts.get(b[1], 0)/total_days:.0%})")
               for b in BANDS]
    ax2.legend(handles=patches, loc="upper right", fontsize=7,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455",
               ncol=5, framealpha=0.9)
    ax2.set_ylabel("Allocation", color=TXT, fontsize=9)
    ax2.set_yticks([])
    for yr in range(2016, 2027):
        ax2.axvline(pd.Timestamp(f"{yr}-01-01"), color="#111122", lw=0.5, ls=":", alpha=0.5)

    # ── Panel 3: Spread with band zones ───────────────────────────────────────
    ax3 = axes[2]
    sv  = spread.values
    idx = spread.index

    # Band zone shading
    band_ranges = [
        (0.25,  0.85,  "#ff6622", 0.10),  # TQQQ
        (0.10,  0.25,  "#f0c040", 0.10),  # QLD
        (0.00,  0.10,  "#4a9eda", 0.10),  # QQQ
        (-0.15, 0.00,  "#88bb44", 0.10),  # SCHD
        (-0.85, -0.15, "#aa66cc", 0.10),  # SGOV
    ]
    for lo, hi, color, alpha in band_ranges:
        ax3.axhspan(lo, hi, alpha=alpha, color=color, lw=0)

    # Threshold lines
    for thresh, _, color, label in BANDS:
        if thresh > -5:
            ax3.axhline(thresh, color=color, lw=0.8, ls="--", alpha=0.6)

    ax3.plot(idx, sv, color="#666688", lw=0.6, alpha=0.5, label="Raw spread")
    ax3.plot(spread_smooth.index, spread_smooth.values,
             color="#f0c040", lw=1.0, alpha=0.9, label=f"EMA({EMA_SPAN}) smoothed")
    ax3.axhline(0, color="#555577", lw=0.7)
    ax3.set_ylabel("Spread (bull-bear)", color=TXT, fontsize=9)
    ax3.set_ylim(-0.75, 0.75)
    ax3.legend(loc="upper right", fontsize=7,
               facecolor="#12121f", labelcolor=TXT, edgecolor="#444455", framealpha=0.9)
    ax3.grid(axis="y", color=GRID, lw=0.3)
    for yr in range(2016, 2027):
        ax3.axvline(pd.Timestamp(f"{yr}-01-01"), color="#333355", lw=0.5, ls=":", alpha=0.6)

    # X-axis
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=9)

    plt.tight_layout(h_pad=0.3)
    out = "chart_rotation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved: {out}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  STRATEGY COMPARISON  2015-2026")
    print(f"  Friction: EMA span={EMA_SPAN}, hysteresis={HYSTERESIS}, min hold={MIN_HOLD_DAYS}d")
    print(f"  Switches: raw={switches_raw}  friction={switches_fric}")
    print(f"{'='*80}")
    print(f"  {'Strategy':<22} {'Ann':>7}  {'MDD':>8}  {'Calmar':>7}  {'Sharpe':>7}  {'Total':>8}")
    print(f"  {'-'*70}")
    for name in ["ML-Rot(friction)", "ML-Rot(raw)", "ML-Binary", "SMA200 (QLD)", "QLD B&H", "QQQ B&H"]:
        m = metrics[name]
        tot = navs[name].iloc[-1] - 100
        print(f"  {name:<22} {m['ann']:>7.1%}  {m['mdd']:>8.1%}  "
              f"{m['calmar']:>7.3f}  {m['sharpe']:>7.3f}  {tot:>+7.0f}%")
    print(f"{'='*80}")

    # Allocation breakdown
    print(f"\n  Allocation breakdown:")
    for b in BANDS:
        name = b[1]
        cnt  = alloc_counts.get(name, 0)
        print(f"    {b[3]:<20}  {cnt:>5} days  ({cnt/total_days:>5.1%})")

    # Asset-level attribution
    print(f"\n  Per-asset contribution to rotation NAV:")
    for b in BANDS:
        name = b[1]
        mask = alloc_rot == name
        if mask.sum() == 0:
            continue
        days_in = mask.sum()
        ret_sum = chart_ret.loc[mask.shift(1, fill_value=False).values, name].sum()
        print(f"    {b[3]:<20}  {days_in:>5} days  cumulative contribution: {ret_sum:>+.1%}")

    # Current position
    last_alloc  = alloc_rot.iloc[-1]
    last_spread = float(spread.iloc[-1])
    print(f"\n  Current: {ASSET_LABELS[last_alloc]}  (spread={last_spread:+.3f})")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
