"""
Telemetry — Structured logging + optional InfluxDB metrics.

InfluxDB is OPTIONAL. If not configured, all metrics are written to
structured JSON log only. Set TOPARB_INFLUX_URL to enable InfluxDB.

Grafana dashboard template: docker/grafana/dashboards/toparb.json

Metrics emitted:
  toparb.allocation.*   — HJB f*, leverage, regime
  toparb.tda.*          — Fracture score, Betti numbers, anomaly
  toparb.risk.*         — VaR, drawdown, daily PnL
  toparb.execution.*    — Orders, fills, slippage, commission
  toparb.portfolio.*    — NAV, returns, Sharpe
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AllocationMetric:
    f_star: float
    leverage: float
    regime: str
    tda_signal: float
    tda_anomaly: bool
    tda_zscore: float
    beta0: float
    beta1: float
    expected_log_growth: float
    effective_sharpe: float
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class RiskMetric:
    var_99: float
    cvar_99: float
    drawdown: float
    daily_pnl: float
    leverage: float
    max_position: float
    daily_turnover: float
    is_halted: bool
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ExecutionMetric:
    orders_submitted: int
    orders_filled: int
    total_turnover: float
    commission_usd: float
    avg_slippage_bps: float
    risk_rejected: bool
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class PortfolioMetric:
    nav: float
    daily_return: float
    total_return: float
    sharpe_rolling: float
    n_positions: int
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


def setup_logging(log_file: str, audit_file: str, log_level: str = "INFO") -> None:
    """Configure structured logging for production."""
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(audit_file) or ".", exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler (100MB × 10 files)
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=100 * 1024 * 1024, backupCount=10
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Audit log (JSON lines, separate file, never truncated)
    audit_handler = logging.handlers.RotatingFileHandler(
        audit_file, maxBytes=500 * 1024 * 1024, backupCount=100
    )
    audit_handler.setFormatter(logging.Formatter("%(message)s"))
    audit = logging.getLogger("toparb.audit")
    audit.addHandler(audit_handler)
    audit.propagate = False

    logger.info(f"Logging configured: {log_file}, audit: {audit_file}")


class Telemetry:
    """
    Metrics emitter. Writes to:
    1. Structured JSON log (always)
    2. InfluxDB (if configured)
    """

    def __init__(
        self,
        influx_url: str = "",
        influx_token: str = "",
        influx_org: str = "toparb",
        influx_bucket: str = "toparb_metrics",
    ) -> None:
        self._influx_url = influx_url
        self._influx_client = None
        self._write_api = None
        self._metrics_log = logging.getLogger("toparb.metrics")

        if influx_url:
            self._init_influx(influx_url, influx_token, influx_org, influx_bucket)

    def _init_influx(
        self, url: str, token: str, org: str, bucket: str
    ) -> None:
        try:
            from influxdb_client import InfluxDBClient  # type: ignore[import]
            from influxdb_client.client.write_api import SYNCHRONOUS  # type: ignore[import]
            self._influx_client = InfluxDBClient(
                url=url, token=token, org=org
            )
            self._write_api = self._influx_client.write_api(write_options=SYNCHRONOUS)
            self._influx_bucket = bucket
            self._influx_org = org
            logger.info(f"InfluxDB connected: {url}")
        except ImportError:
            logger.warning(
                "influxdb-client not installed. InfluxDB metrics disabled. "
                "Run: pip install influxdb-client"
            )
        except Exception as e:
            logger.warning(f"InfluxDB connection failed: {e}. Continuing without InfluxDB.")

    def emit_allocation(self, metric: AllocationMetric) -> None:
        self._log_metric("allocation", asdict(metric))
        if self._write_api:
            self._write_influx("toparb_allocation", {
                "f_star": metric.f_star,
                "leverage": metric.leverage,
                "tda_signal": metric.tda_signal,
                "tda_zscore": metric.tda_zscore,
                "expected_log_growth": metric.expected_log_growth,
                "effective_sharpe": metric.effective_sharpe,
                "beta0": metric.beta0,
                "beta1": metric.beta1,
            }, tags={
                "regime": metric.regime,
                "tda_anomaly": str(metric.tda_anomaly),
            }, timestamp=metric.timestamp)

    def emit_risk(self, metric: RiskMetric) -> None:
        self._log_metric("risk", asdict(metric))
        if self._write_api:
            self._write_influx("toparb_risk", {
                "var_99": metric.var_99,
                "cvar_99": metric.cvar_99,
                "drawdown": metric.drawdown,
                "daily_pnl": metric.daily_pnl,
                "leverage": metric.leverage,
                "max_position": metric.max_position,
                "daily_turnover": metric.daily_turnover,
            }, tags={"halted": str(metric.is_halted)}, timestamp=metric.timestamp)

    def emit_execution(self, metric: ExecutionMetric) -> None:
        self._log_metric("execution", asdict(metric))
        if self._write_api:
            self._write_influx("toparb_execution", {
                "orders_submitted": metric.orders_submitted,
                "orders_filled": metric.orders_filled,
                "total_turnover": metric.total_turnover,
                "commission_usd": metric.commission_usd,
                "avg_slippage_bps": metric.avg_slippage_bps,
            }, tags={"risk_rejected": str(metric.risk_rejected)}, timestamp=metric.timestamp)

    def emit_portfolio(self, metric: PortfolioMetric) -> None:
        self._log_metric("portfolio", asdict(metric))
        if self._write_api:
            self._write_influx("toparb_portfolio", {
                "nav": metric.nav,
                "daily_return": metric.daily_return,
                "total_return": metric.total_return,
                "sharpe_rolling": metric.sharpe_rolling,
                "n_positions": metric.n_positions,
            }, timestamp=metric.timestamp)

    def emit_heartbeat(self, step: int, state: str) -> None:
        self._log_metric("heartbeat", {"step": step, "state": state})
        if self._write_api:
            self._write_influx("toparb_heartbeat", {"step": step}, tags={"state": state})

    def _log_metric(self, metric_type: str, data: dict[str, Any]) -> None:
        record = {"type": metric_type, "ts": time.time(), **data}
        self._metrics_log.info(json.dumps(record))

    def _write_influx(
        self,
        measurement: str,
        fields: dict[str, Any],
        tags: dict[str, str] | None = None,
        timestamp: float | None = None,
    ) -> None:
        if not self._write_api:
            return
        try:
            from influxdb_client import Point  # type: ignore[import]
            from influxdb_client.domain.write_precision import WritePrecision  # type: ignore[import]
            p = Point(measurement)
            for k, v in (tags or {}).items():
                p = p.tag(k, v)
            for k, v in fields.items():
                p = p.field(k, float(v) if isinstance(v, (int, float)) else v)
            if timestamp:
                p = p.time(int(timestamp * 1e9), WritePrecision.NANOSECONDS)
            self._write_api.write(
                bucket=self._influx_bucket,
                org=self._influx_org,
                record=p,
            )
        except Exception as e:
            logger.debug(f"InfluxDB write failed (non-critical): {e}")

    def close(self) -> None:
        if self._influx_client:
            try:
                self._influx_client.close()
            except Exception:
                pass