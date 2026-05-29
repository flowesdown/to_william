"""
TopArb Configuration — All parameters in one place.

Override any value via environment variables:
    TOPARB_UNIVERSE_SIZE=100
    TOPARB_MAX_LEVERAGE=2.0
    TOPARB_PAPER_TRADING=true
    etc.

For live trading, copy .env.example → .env and fill in credentials.
NEVER commit credentials to version control.
"""
from __future__ import annotations

import os
from typing import Literal

# ---------------------------------------------------------------------------
# Simple config without pydantic dependency (pydantic-settings optional)
# ---------------------------------------------------------------------------

def _env(key: str, default):
    val = os.getenv(key)
    if val is None:
        return default
    # auto-cast booleans
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes")
    # auto-cast numerics
    if isinstance(default, float):
        return float(val)
    if isinstance(default, int):
        return int(val)
    return val


class UniverseConfig:
    """Asset universe parameters."""
    size: int = _env("TOPARB_UNIVERSE_SIZE", 23)
    # S&P 100 liquid subset — safe default for live
    default_tickers: list[str] = [
        "NVDA", "TSLA", "MSTR", "MARA", "RIOT", "AMD", "COIN", "PLTR",
        "SMCI", "GME", "AMC", "SOXL", "LABU", "TQQQ", "UPRO", "BITO",
        "HOOD", "AFRM", "PYPL", "AI", "PATH", "SNOW", "IONQ"
    ]
    pca_window_days: int = _env("TOPARB_PCA_WINDOW", 90)
    tda_window_days: int = _env("TOPARB_TDA_WINDOW", 20)


class ModelConfig:
    """Core model hyperparameters — calibrate via backtest before changing."""
    n_pca_components: int = _env("TOPARB_N_PCA", 10)
    gamma_tda: float = _env("TOPARB_GAMMA_TDA", 3.0)
    kappa_anomaly: float = _env("TOPARB_KAPPA_ANOMALY", 1.5)
    risk_aversion: float = _env("TOPARB_RISK_AVERSION", 2.0)
    persistence_threshold: float = _env("TOPARB_PERSIST_THRESH", 0.05)
    anomaly_zscore_threshold: float = _env("TOPARB_ANOMALY_Z", 2.5)
    fracture_ema_alpha: float = _env("TOPARB_EMA_ALPHA", 0.1)
    n_eigen_portfolios: int = _env("TOPARB_N_EIGEN", 1)
    downsample_to: int = _env("TOPARB_DOWNSAMPLE", 64)


class RiskConfig:
    """Hard risk limits. These are enforced by RiskManager and cannot be
    overridden at runtime. Change only with extreme care."""

    # --- Leverage ---
    max_leverage: float = _env("TOPARB_MAX_LEVERAGE", 2.0)
    max_leverage_hard: float = 4.0            # Absolute ceiling, never exceeded

    # --- Position sizing ---
    max_single_position_frac: float = 0.10   # 10% of portfolio per ticker
    max_sector_concentration: float = 0.35   # 35% per sector

    # --- Loss limits ---
    daily_loss_limit_frac: float = _env("TOPARB_DAILY_LOSS", 0.05)   # 5% NAV/day
    drawdown_halt_frac: float = _env("TOPARB_DRAWDOWN_HALT", 0.15)    # 15% from peak → halt

    # --- VaR ---
    var_confidence: float = 0.99
    var_budget_frac: float = _env("TOPARB_VAR_BUDGET", 0.02)          # 2% daily VaR budget

    # --- Order sizing ---
    max_order_usd: float = _env("TOPARB_MAX_ORDER_USD", 500_000.0)
    min_order_usd: float = 100.0

    # --- Rebalancing ---
    rebalance_threshold: float = _env("TOPARB_REBAL_THRESH", 0.02)    # 2% drift → rebalance
    max_turnover_per_step: float = 0.30                                 # 30% max turnover/step

    # --- Timing ---
    min_seconds_between_orders: float = 5.0
    market_open_buffer_minutes: int = 15   # Don't trade first 15 min
    market_close_buffer_minutes: int = 15  # Don't trade last 15 min


class ExecutionConfig:
    """Execution and broker settings."""
    paper_trading: bool = _env("TOPARB_PAPER_TRADING", True)
    transaction_cost_bps: float = _env("TOPARB_TC_BPS", 2.0)
    slippage_bps: float = _env("TOPARB_SLIPPAGE_BPS", 1.0)

    # IBKR settings (only used when paper_trading=False)
    ibkr_host: str = _env("TOPARB_IBKR_HOST", "127.0.0.1")
    ibkr_port: int = _env("TOPARB_IBKR_PORT", 7497)    # 7497=paper, 7496=live
    ibkr_client_id: int = _env("TOPARB_IBKR_CLIENT_ID", 1)

    # Polygon.io (live feed)
    polygon_api_key: str = _env("TOPARB_POLYGON_KEY", "")

    # Feed selection
    feed_mode: str = _env("TOPARB_FEED_MODE", "yfinance")  # yfinance | polygon | ibkr


class MonitoringConfig:
    """Telemetry and monitoring settings."""
    log_level: str = _env("TOPARB_LOG_LEVEL", "INFO")
    log_file: str = _env("TOPARB_LOG_FILE", "logs/toparb.log")

    # InfluxDB (optional — set TOPARB_INFLUX_URL to enable)
    influx_url: str = _env("TOPARB_INFLUX_URL", "")
    influx_token: str = _env("TOPARB_INFLUX_TOKEN", "")
    influx_org: str = _env("TOPARB_INFLUX_ORG", "toparb")
    influx_bucket: str = _env("TOPARB_INFLUX_BUCKET", "toparb_metrics")

    # Metrics emit interval
    metrics_interval_s: float = _env("TOPARB_METRICS_INTERVAL", 5.0)
    audit_log_file: str = _env("TOPARB_AUDIT_LOG", "logs/audit.jsonl")


class BacktestConfig:
    """Backtesting parameters."""
    start_date: str = _env("TOPARB_BT_START", "2018-01-01")
    end_date: str = _env("TOPARB_BT_END", "2024-12-31")
    initial_capital: float = _env("TOPARB_CAPITAL", 1_000_000.0)
    benchmark_ticker: str = _env("TOPARB_BENCHMARK", "SPY")
    walk_forward_n_splits: int = _env("TOPARB_WF_SPLITS", 4)


class Settings:
    """Aggregated settings object."""
    universe = UniverseConfig()
    model = ModelConfig()
    risk = RiskConfig()
    execution = ExecutionConfig()
    monitoring = MonitoringConfig()
    backtest = BacktestConfig()


settings = Settings()