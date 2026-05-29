#!/usr/bin/env python3
"""
TopArb Parameter Calibration — Walk-Forward Grid Search

Downloads 2 years of historical data and finds optimal parameters
by maximizing Calmar ratio (ann_return / max_drawdown) on out-of-sample data.

Uses walk-forward validation to avoid overfitting:
  - Each train/test split is evaluated independently
  - Final score = mean(calmar) across all folds

Usage:
    TOPARB_FORCE_CPU=1 python calibrate.py
    TOPARB_FORCE_CPU=1 python calibrate.py --tickers 20 --years 2
"""
from __future__ import annotations

import argparse
import os
import sys
import types
import warnings
import logging

warnings.filterwarnings("ignore")

# ── CPU shim (must be before project imports) ─────────────────────────────────
if os.getenv("TOPARB_FORCE_CPU", "0").lower() in ("1", "true", "yes"):
    import numpy as _np
    _cupy = types.ModuleType("cupy")
    for _k, _v in _np.__dict__.items():
        setattr(_cupy, _k, _v)
    _cupy.asnumpy = lambda arr: _np.asarray(arr)
    _cupy.ndarray = _np.ndarray
    sys.modules["cupy"] = _cupy

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser(description="TopArb parameter calibration")
    p.add_argument("--tickers", type=int, default=20, help="Number of tickers (default: 20)")
    p.add_argument("--years", type=int, default=2, help="Years of history (default: 2)")
    p.add_argument("--folds", type=int, default=3, help="Walk-forward folds (default: 3)")
    return p.parse_args()


def download_data(tickers: list[str], years: int) -> np.ndarray:
    """Download and return (n_assets × n_days) log-return matrix."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: pip install yfinance")
        sys.exit(1)

    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=years * 365 + 60)).strftime("%Y-%m-%d")

    print(f"  Downloading {len(tickers)} tickers ({start} → {end})...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    closes: dict[str, pd.Series] = {}
    for t in tickers:
        col = raw["Close"][t] if len(tickers) > 1 else raw["Close"]
        s = col.dropna()
        if len(s) > 100:
            closes[t] = s

    if len(closes) < 5:
        print(f"ERROR: Only {len(closes)} tickers with data. Need >= 5.")
        sys.exit(1)

    df = pd.DataFrame(closes).ffill().bfill().dropna()
    rets = np.log(df / df.shift(1)).dropna().values.T   # (N × T)
    rets = np.clip(rets, -0.20, 0.20)
    rets = np.nan_to_num(rets, nan=0.0)
    print(f"  Matrix: {rets.shape[0]} assets × {rets.shape[1]} days")
    return rets


def run_single_backtest(
    return_matrix: np.ndarray,
    risk_aversion: float,
    gamma_tda: float,
    refit_every: int,
    train_window: int = 60,
) -> dict | None:
    """Run one backtest fold. Returns metrics dict or None on error."""
    try:
        from config.settings import ModelConfig, RiskConfig, BacktestConfig
        from src.backtest.backtester import TopArbBacktester

        model_cfg = ModelConfig()
        model_cfg.risk_aversion = risk_aversion
        model_cfg.gamma_tda = gamma_tda
        model_cfg.n_pca_components = min(10, return_matrix.shape[0] - 1)
        model_cfg.downsample_to = min(64, return_matrix.shape[0])

        risk_cfg = RiskConfig()
        risk_cfg.max_leverage = 1.5   # Conservative during calibration

        bt = TopArbBacktester(
            model_config=model_cfg,
            risk_config=risk_cfg,
            backtest_config=BacktestConfig(),
            refit_every=refit_every,
        )
        report = bt.run(return_matrix, train_window=train_window, verbose=False)

        n_days = max(len(report.daily_returns), 1)
        ann_ret = (report.nav_series[-1] / report.nav_series[0]) ** (252 / n_days) - 1
        calmar = ann_ret / max(abs(report.max_drawdown), 0.001)

        return {
            "ann_ret": ann_ret,
            "max_dd": report.max_drawdown,
            "sharpe": report.sharpe_ratio,
            "calmar": calmar,
            "avg_lev": report.avg_leverage,
            "tda_anomaly_rate": report.tda_anomaly_rate,
        }
    except Exception as e:
        return None


def main() -> None:
    args = parse_args()

    from config.settings import settings
    tickers = settings.universe.default_tickers[: args.tickers]

    print("\n" + "=" * 60)
    print("  TopArb Parameter Calibration (Walk-Forward)")
    print("=" * 60)

    ret_matrix = download_data(tickers, args.years)
    n_assets, T = ret_matrix.shape

    # Walk-forward folds
    fold_size = T // (args.folds + 1)
    train_window = min(120, fold_size - 10)

    # Each fold needs: train_window (in-sample) + at least 20 days OOS
    # fold_size = T // (folds + 1) ensures folds don't overlap
    # Recalculate with at most 2 folds if data is tight
    while fold_size < train_window + 20 and args.folds > 1:
        args.folds -= 1
        fold_size = T // (args.folds + 1)
        print(f"  ⚠  Reduced folds to {args.folds} (data constraint)")

    if fold_size < train_window + 20:
        # Last resort: single fold with everything
        args.folds = 1
        fold_size = T // 2
        print(f"  ⚠  Using single fold (only {T} days of data)")

    print(f"\n  Fold size: {fold_size} days | Train window: {train_window} days")

    # Grid
    GRID = {
        "risk_aversion": [1.5, 2.0, 3.0, 4.0],
        "gamma_tda":     [2.0, 5.0, 10.0, 20.0],
        "refit_every":   [5, 10, 15],
    }

    total = len(GRID["risk_aversion"]) * len(GRID["gamma_tda"]) * len(GRID["refit_every"])
    print(f"  Grid: {total} combinations × {args.folds} folds = {total * args.folds} runs\n")

    results: list[dict] = []
    count = 0

    for ra in GRID["risk_aversion"]:
        for gamma in GRID["gamma_tda"]:
            for refit in GRID["refit_every"]:
                count += 1
                fold_metrics: list[dict] = []

                for fold in range(args.folds):
                    t_start = fold * fold_size
                    t_end = t_start + fold_size + fold_size   # train + test
                    t_end = min(t_end, T)
                    window = ret_matrix[:, t_start:t_end]

                    m = run_single_backtest(window, ra, gamma, refit, train_window)
                    if m is not None:
                        fold_metrics.append(m)

                if not fold_metrics:
                    print(f"  [{count:3d}/{total}] RA={ra} γ={gamma:5.1f} refit={refit:2d} — ⚠ all folds failed")
                    continue

                mean_calmar = np.mean([m["calmar"] for m in fold_metrics])
                mean_ret    = np.mean([m["ann_ret"] for m in fold_metrics])
                mean_dd     = np.mean([m["max_dd"]  for m in fold_metrics])
                mean_sharpe = np.mean([m["sharpe"]  for m in fold_metrics])
                mean_lev    = np.mean([m["avg_lev"] for m in fold_metrics])

                icon = "✅" if mean_ret > 0 and mean_calmar > 0.5 else ("⚠️" if mean_ret > 0 else "❌")

                print(
                    f"  [{count:3d}/{total}] {icon} RA={ra} γ={gamma:5.1f} refit={refit:2d} | "
                    f"Ret={mean_ret*100:+6.2f}% DD={mean_dd*100:5.2f}% "
                    f"Sharpe={mean_sharpe:5.2f} Calmar={mean_calmar:5.2f} Lev={mean_lev:.2f}x"
                )

                results.append({
                    "risk_aversion": ra,
                    "gamma_tda": gamma,
                    "refit_every": refit,
                    "mean_calmar": mean_calmar,
                    "mean_ret": mean_ret,
                    "mean_dd": mean_dd,
                    "mean_sharpe": mean_sharpe,
                    "mean_lev": mean_lev,
                    "n_folds": len(fold_metrics),
                })

    print("\n" + "=" * 60)

    if not results:
        print("❌ All combinations failed. Check data quality.")
        return

    # Sort by Calmar
    results.sort(key=lambda x: x["mean_calmar"], reverse=True)
    best = results[0]

    print("🏆  OPTIMAL PARAMETERS (by Walk-Forward Calmar Ratio)")
    print("=" * 60)
    print(f"  TOPARB_RISK_AVERSION={best['risk_aversion']}")
    print(f"  TOPARB_GAMMA_TDA={best['gamma_tda']}")
    print(f"  Refit every {best['refit_every']} steps")
    print("-" * 60)
    print(f"  Ann. Return (OOS): {best['mean_ret']*100:+.2f}%")
    print(f"  Max Drawdown:      {best['mean_dd']*100:.2f}%")
    print(f"  Sharpe:            {best['mean_sharpe']:.3f}")
    print(f"  Calmar:            {best['mean_calmar']:.3f}")
    print(f"  Avg Leverage:      {best['mean_lev']:.2f}x")
    print(f"  Folds evaluated:   {best['n_folds']}/{args.folds}")
    print("=" * 60)

    print("\n📋  Top 5 parameter sets:")
    print(f"  {'Rank':<5} {'RA':<6} {'γ_TDA':<8} {'Refit':<7} {'Ret%':<8} {'DD%':<8} {'Sharpe':<8} {'Calmar'}")
    print("  " + "-" * 58)
    for i, r in enumerate(results[:5], 1):
        print(
            f"  {i:<5} {r['risk_aversion']:<6} {r['gamma_tda']:<8} {r['refit_every']:<7} "
            f"{r['mean_ret']*100:+6.2f}%  {r['mean_dd']*100:5.2f}%  "
            f"{r['mean_sharpe']:5.3f}   {r['mean_calmar']:5.3f}"
        )

    print(f"""
Next steps:
  1. Add to your .env:
       TOPARB_RISK_AVERSION={best['risk_aversion']}
       TOPARB_GAMMA_TDA={best['gamma_tda']}
  2. Run full backtest:
       make backtest
  3. Paper trade for >= 30 days:
       make paper
""")


if __name__ == "__main__":
    main()
