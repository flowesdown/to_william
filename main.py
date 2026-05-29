#!/usr/bin/env python3
"""
TopArb — Main Entry Point

Usage:
    # Paper trading (safe, default)
    python main.py

    # Run backtest on historical data
    python main.py --mode backtest --start 2020-01-01 --end 2023-12-31

    # Validate config and exit
    python main.py --mode validate

Environment variables (all optional — see config/settings.py for defaults):
    TOPARB_FORCE_CPU=1              Force CPU mode (no GPU required)
    TOPARB_PAPER_TRADING=true       Paper trading mode (default: true)
    TOPARB_FEED_MODE=yfinance       Data feed: yfinance | polygon
    TOPARB_POLYGON_KEY=<key>        Polygon.io API key (if using polygon feed)
    TOPARB_MAX_LEVERAGE=2.0         Maximum portfolio leverage
    TOPARB_DAILY_LOSS=0.05          Daily loss limit (fraction of NAV)
    TOPARB_DRAWDOWN_HALT=0.15       Drawdown halt threshold

SAFETY CHECKLIST before running live:
    [ ] Reviewed all risk limits in config/settings.py
    [ ] Ran backtests on your actual ticker universe
    [ ] Paper traded for >= 30 days
    [ ] Calibrated gamma_tda and kappa_anomaly
    [ ] InfluxDB/Grafana monitoring is up
    [ ] Reviewed all test output (pytest tests/ -v)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

# --- GPU to CPU SHIM ---
if os.getenv("TOPARB_FORCE_CPU", "0").lower() in ("1", "true", "yes"):
    import numpy as np
    import types
    import sys
    cupy_shim = types.ModuleType("cupy")
    for k, v in np.__dict__.items():
        setattr(cupy_shim, k, v)
    cupy_shim.asnumpy = lambda arr: np.asarray(arr)
    cupy_shim.ndarray = np.ndarray
    sys.modules["cupy"] = cupy_shim
# -----------------------

logger = logging.getLogger("toparb.main")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TopArb: GPU-Accelerated Topological StatArb & HJB Controller"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "backtest", "validate"],
        default="live",
        help="Run mode (default: live paper trading)",
    )
    parser.add_argument("--start", default="2020-01-01", help="Backtest start date")
    parser.add_argument("--end",   default="2023-12-31", help="Backtest end date")
    parser.add_argument(
        "--train-window", type=int, default=60,
        help="Training window in days for backtest (default: 60)",
    )
    parser.add_argument(
        "--no-paper",
        action="store_true",
        help="⚠ DISABLE paper trading safeguard. Requires TOPARB_PAPER_TRADING=false.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def _validate_config() -> bool:
    """Run sanity checks on current configuration. Returns True if valid."""
    from config.settings import settings
    ok = True
    checks = []

    cfg = settings.risk
    if cfg.max_leverage > cfg.max_leverage_hard:
        checks.append(f"ERROR: max_leverage ({cfg.max_leverage}) > hard cap ({cfg.max_leverage_hard})")
        ok = False

    if cfg.daily_loss_limit_frac <= 0 or cfg.daily_loss_limit_frac > 0.5:
        checks.append(f"WARN: daily_loss_limit_frac={cfg.daily_loss_limit_frac} seems unusual")

    if cfg.drawdown_halt_frac <= 0 or cfg.drawdown_halt_frac > 0.5:
        checks.append(f"WARN: drawdown_halt_frac={cfg.drawdown_halt_frac} seems unusual")

    if settings.model.gamma_tda < 0:
        checks.append("ERROR: gamma_tda must be >= 0")
        ok = False

    if settings.execution.paper_trading:
        checks.append("INFO: Paper trading mode active (safe)")
    else:
        checks.append("⚠ WARNING: LIVE trading mode. Real money at risk!")

    for check in checks:
        print(f"  {check}")

    return ok


def _run_backtest(args: argparse.Namespace) -> None:
    """Download historical data and run full backtest."""
    from config.settings import settings
    from src.backtest.backtester import TopArbBacktester
    import numpy as np

    print(f"\nTopArb Backtest: {args.start} → {args.end}")
    print(f"Universe: {len(settings.universe.default_tickers)} tickers")
    print(f"Train window: {args.train_window} days\n")

    # Try to download historical data
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    tickers = settings.universe.default_tickers[:settings.universe.size]
    print(f"Downloading {len(tickers)} tickers from {args.start} to {args.end}...")

    try:
        raw = yf.download(
            tickers,
            start=args.start,
            end=args.end,
            progress=args.verbose,
            group_by="ticker",
        )
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    # Build return matrix
    close_series = {}
    for ticker in tickers:
        try:
            col = raw[ticker]["Close"] if len(tickers) > 1 else raw["Close"]
            col = col.dropna()
            if len(col) > args.train_window + 10:
                import numpy as np
                close_series[ticker] = np.log(col.values[1:] / col.values[:-1])
        except Exception:
            pass

    if len(close_series) < 5:
        print(f"ERROR: Only {len(close_series)} tickers with valid data. Need >= 5.")
        sys.exit(1)

    import numpy as np
    min_len = min(len(v) for v in close_series.values())
    return_matrix = np.array([v[-min_len:] for v in close_series.values()])
    return_matrix = np.clip(return_matrix, -0.30, 0.30)
    return_matrix = np.nan_to_num(return_matrix, nan=0.0)

    print(f"Return matrix: {return_matrix.shape[0]} assets × {return_matrix.shape[1]} days\n")

    # Run backtest
    backtester = TopArbBacktester(
        model_config=settings.model,
        risk_config=settings.risk,
        backtest_config=settings.backtest,
        refit_every=5,
    )
    report = backtester.run(
        return_matrix,
        tickers=list(close_series.keys()),
        train_window=args.train_window,
        verbose=args.verbose,
    )
    report.print_summary()


async def _run_live(args: argparse.Namespace) -> None:
    """Run the live (paper) trading orchestrator."""
    from config.settings import settings
    from src.orchestrator import TopArbOrchestrator

    paper_override = not args.no_paper

    if args.no_paper:
        confirm = input(
            "\n⚠ LIVE TRADING MODE REQUESTED.\n"
            "This will place REAL orders with real money.\n"
            "Type 'I UNDERSTAND' to continue: "
        ).strip()
        if confirm != "I UNDERSTAND":
            print("Cancelled.")
            sys.exit(0)

    orchestrator = TopArbOrchestrator(
        cfg=settings,
        paper_trading_override=paper_override,
    )
    await orchestrator.run()


def main() -> None:
    args = _parse_args()

    # Set up basic logging for startup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )

    print("\n" + "=" * 60)
    print("  TopArb: GPU-Accelerated Topological StatArb")
    print("=" * 60)

    if args.mode == "validate":
        print("\nValidating configuration...\n")
        ok = _validate_config()
        if ok:
            print("\n✓ Configuration is valid.\n")
            sys.exit(0)
        else:
            print("\n✗ Configuration has errors. Fix before running.\n")
            sys.exit(1)

    elif args.mode == "backtest":
        _validate_config()
        _run_backtest(args)

    elif args.mode == "live":
        print("\nValidating configuration...")
        _validate_config()
        print("\nStarting orchestrator...\n")
        asyncio.run(_run_live(args))


if __name__ == "__main__":
    main()