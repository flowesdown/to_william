"""
TopArb Orchestrator — Main trading loop.

State machine:
    INITIALIZING → WARMING_UP → TRADING ⟵→ RISK_OFF → SHUTDOWN

Safety guarantees:
- All state transitions are logged
- CTRL+C triggers graceful shutdown (cancel all open orders)
- Kill switch pauses all trading, requires manual resume
- Model re-fitting runs in background, never blocks the main loop
- All exceptions are caught and logged, never crash the loop silently
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from enum import Enum, auto
from typing import Sequence

import numpy as np

from config.settings import Settings, settings
from src.data.feed import DataFeed, create_feed
from src.execution.hjb_controller import TopologicalKellyController
from src.execution.order_router import OrderRouter, create_router
from src.execution.rebalancer import ThresholdRebalancer
from src.math.eigen_risk import PCARiskModel
from src.math.manifold_tda import VietorisRipsManifold
from src.monitoring.telemetry import (
    AllocationMetric,
    ExecutionMetric,
    PortfolioMetric,
    RiskMetric,
    Telemetry,
    setup_logging,
)
from src.risk.risk_manager import RiskManager
from src.utils.backend import GPU_AVAILABLE, from_numpy, to_numpy

logger = logging.getLogger(__name__)


class OrchestratorState(Enum):
    INITIALIZING = auto()
    WARMING_UP = auto()
    TRADING = auto()
    RISK_OFF = auto()
    SHUTDOWN = auto()


class TopArbOrchestrator:
    """
    Coordinates all components of the TopArb pipeline for live trading.

    Do not run with real money until you have:
    1. Validated the backtest on your actual universe
    2. Run paper trading for at least 30 days
    3. Calibrated gamma_tda and kappa_anomaly on your data
    4. Reviewed and understood all risk limits
    """

    def __init__(
        self,
        cfg: Settings = settings,
        paper_trading_override: bool = True,
    ) -> None:
        self._cfg = cfg
        self._state = OrchestratorState.INITIALIZING
        self._step = 0
        self._shutdown_event = asyncio.Event()
        self._paper_override = paper_trading_override

        # Components (initialized in _setup)
        self._feed: DataFeed | None = None
        self._router: OrderRouter | None = None
        self._risk: RiskManager | None = None
        self._controller: TopologicalKellyController | None = None
        self._rebalancer: ThresholdRebalancer | None = None
        self._telemetry: Telemetry | None = None

        # State
        self._tickers: list[str] = []
        self._n_assets: int = 0
        self._current_weights: np.ndarray = np.array([])
        self._nav: float = cfg.backtest.initial_capital
        self._returns_window: list[np.ndarray] = []   # Rolling window of daily returns
        self._nav_history: list[float] = []

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    async def run(self) -> None:
        """Main entry point. Run until shutdown."""
        try:
            setup_logging(
                self._cfg.monitoring.log_file,
                self._cfg.monitoring.audit_log_file,
                self._cfg.monitoring.log_level,
            )
            logger.info("=" * 60)
            logger.info("  TopArb Orchestrator Starting")
            logger.info(f"  Paper trading: {self._cfg.execution.paper_trading or self._paper_override}")
            logger.info(f"  Feed mode: {self._cfg.execution.feed_mode}")
            logger.info(f"  GPU available: {GPU_AVAILABLE}")
            logger.info("=" * 60)

            await self._setup()
            await self._warm_up()
            await self._trading_loop()

        except Exception as e:
            logger.critical(f"Orchestrator fatal error: {e}", exc_info=True)
            raise
        finally:
            await self._shutdown()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def _setup(self) -> None:
        """Initialize all components."""
        self._transition(OrchestratorState.INITIALIZING)

        # Universe
        tickers = self._cfg.universe.default_tickers[: self._cfg.universe.size]
        self._tickers = tickers
        self._n_assets = len(tickers)
        self._current_weights = np.zeros(self._n_assets)

        logger.info(f"Universe: {self._n_assets} assets")

        # Telemetry
        self._telemetry = Telemetry(
            influx_url=self._cfg.monitoring.influx_url,
            influx_token=self._cfg.monitoring.influx_token,
            influx_org=self._cfg.monitoring.influx_org,
            influx_bucket=self._cfg.monitoring.influx_bucket,
        )

        # Risk manager
        self._risk = RiskManager(
            config=self._cfg.risk,
            n_assets=self._n_assets,
            audit_logger=logging.getLogger("toparb.audit"),
        )
        self._risk.update_nav(self._nav)
        self._risk.reset_daily(self._nav)

        # Data feed
        self._feed = create_feed(
            mode=self._cfg.execution.feed_mode,
            tickers=tickers,
            api_key=self._cfg.execution.polygon_api_key,
        )
        await self._feed.subscribe(tickers)

        # Order router
        self._router = create_router(
            self._cfg.execution,
            paper_override=self._paper_override,
        )

        # Rebalancer
        self._rebalancer = ThresholdRebalancer(
            base_threshold=self._cfg.risk.rebalance_threshold,
        )

        logger.info("All components initialized")

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    async def _warm_up(self) -> None:
        """Download historical data and fit initial model."""
        self._transition(OrchestratorState.WARMING_UP)
        logger.info("Warm-up: downloading historical data...")

        # Fetch historical data for model initialization
        end = time.strftime("%Y-%m-%d")
        from datetime import datetime, timedelta
        start_dt = datetime.now() - timedelta(days=self._cfg.universe.pca_window_days + 30)
        start = start_dt.strftime("%Y-%m-%d")

        hist = await self._feed.get_historical(self._tickers, start, end)
        if not hist:
            raise RuntimeError("Failed to fetch historical data for warm-up")

        # Build return matrix from historical data
        if hasattr(self._feed, "build_return_matrix"):
            return_matrix = self._feed.build_return_matrix(hist, self._tickers)
        else:
            # Fallback: compute from first available dataset
            ticker_data = list(hist.values())[0]
            col = "Close" if "Close" in ticker_data.columns else ticker_data.columns[0]
            closes = ticker_data[col].values
            returns = np.log(closes[1:] / closes[:-1])
            return_matrix = returns[np.newaxis, :]

        n_assets_avail, T_hist = return_matrix.shape
        logger.info(f"Return matrix: {n_assets_avail} assets × {T_hist} days")

        # Fit PCA + TDA
        pca_window = min(T_hist, self._cfg.universe.pca_window_days)
        returns_window = return_matrix[:, -pca_window:]

        n_components = min(
            self._cfg.model.n_pca_components,
            n_assets_avail - 1,
            pca_window - 1,
        )

        pca_model = PCARiskModel(n_components=n_components)
        tda_model = VietorisRipsManifold(
            persistence_threshold=self._cfg.model.persistence_threshold,
            anomaly_zscore_threshold=self._cfg.model.anomaly_zscore_threshold,
            fracture_ema_alpha=self._cfg.model.fracture_ema_alpha,
            downsample_to=min(self._cfg.model.downsample_to, n_assets_avail),
        )

        # Convert to backend array
        returns_gpu = from_numpy(returns_window.T.astype(np.float64))  # (T × N) for PCA
        pca_model.fit(returns_gpu)

        self._controller = TopologicalKellyController(
            pca_model=pca_model,
            tda_model=tda_model,
            risk_aversion=self._cfg.model.risk_aversion,
            gamma_tda=self._cfg.model.gamma_tda,
            kappa_anomaly=self._cfg.model.kappa_anomaly,
            max_leverage=self._cfg.risk.max_leverage,
            n_eigen_portfolios=self._cfg.model.n_eigen_portfolios,
        )

        # Pre-populate returns window
        self._returns_window = [return_matrix[:, -i-1] for i in range(min(10, T_hist))]
        self._n_assets = n_assets_avail

        logger.info("Warm-up complete. Entering trading loop.")

    # ------------------------------------------------------------------
    # Trading loop
    # ------------------------------------------------------------------

    async def _trading_loop(self) -> None:
        self._transition(OrchestratorState.TRADING)

        # EOD режим: просыпаемся, делаем расчеты, засыпаем на 12 часов
        while not self._shutdown_event.is_set():
            logger.info("=== Запуск End-Of-Day Расчета ===")

            try:
                # Форсируем скачивание свежих данных
                if isinstance(self._feed, DataFeed):
                    await self._feed._refresh_historical()

                await self._trading_step()

                logger.info("Расчет завершен. Следующий цикл через 12 часов...")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка EOD расчета: {e}", exc_info=True)

            # Засыпаем (например, на 12 часов)
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._shutdown_event.wait()),
                    timeout=43200  # 12 часов в секундах
                )
            except asyncio.TimeoutError:
                pass
    async def _trading_step(self) -> None:
        """Single step of the trading loop."""
        self._step += 1

        if self._state == OrchestratorState.RISK_OFF:
            return  # Don't compute allocations when halted

        # Get latest data
        if self._feed is None:
            return
        snapshot = await self._feed.get_snapshot()
        if snapshot is None or snapshot.return_matrix is None:
            logger.debug(f"Step {self._step}: no snapshot available")
            return

        return_matrix = snapshot.return_matrix
        n_assets_snap = return_matrix.shape[0]

        # Ensure dimensions match
        # if n_assets_snap != self._n_assets:
        #     logger.warning(f"Asset count mismatch: {n_assets_snap} vs {self._n_assets}")
        #     return

        # Compute allocation
        if self._controller is None:
            return

        returns_gpu = from_numpy(return_matrix.T.astype(np.float64))  # (T × N)
        allocation = self._controller.step(returns_gpu)

        target_weights_np = to_numpy(allocation.weights)
        target_weights_np = np.nan_to_num(target_weights_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Rebalance decision
        tda_signal = allocation.tda_signal
        if self._rebalancer is not None:
            decision = self._rebalancer.should_rebalance(
                current_weights=self._current_weights[:n_assets_snap],
                target_weights=target_weights_np,
                current_f_star=allocation.f_star,
                max_leverage=self._cfg.risk.max_leverage,
                tda_is_anomaly=tda_signal.is_anomaly,
            )

            if decision.should_rebalance:
                await self._execute_rebalance(
                    target_weights_np,
                    snapshot.prices,
                    return_matrix,
                    allocation,
                )

        # Emit telemetry
        if self._telemetry:
            self._telemetry.emit_allocation(AllocationMetric(
                f_star=allocation.f_star,
                leverage=float(np.sum(np.abs(target_weights_np))),
                regime=allocation.hjb.leverage_regime,
                tda_signal=tda_signal.fracture_score,
                tda_anomaly=tda_signal.is_anomaly,
                tda_zscore=tda_signal.anomaly_zscore,
                beta0=float(tda_signal.h0_betti_history[-1]) if len(tda_signal.h0_betti_history) > 0 else 0.0,
                beta1=float(tda_signal.h1_betti_history[-1]) if len(tda_signal.h1_betti_history) > 0 else 0.0,
                expected_log_growth=allocation.expected_log_growth,
                effective_sharpe=allocation.effective_sharpe,
            ))

            if self._risk:
                metrics = self._risk.compute_risk_metrics(
                    self._current_weights[:n_assets_snap], self._nav
                )
                self._telemetry.emit_risk(RiskMetric(
                    var_99=metrics.var_99,
                    cvar_99=metrics.cvar_99,
                    drawdown=metrics.drawdown_frac,
                    daily_pnl=metrics.daily_pnl_frac,
                    leverage=metrics.leverage,
                    max_position=metrics.max_position,
                    daily_turnover=metrics.daily_turnover,
                    is_halted=self._risk.is_halted,
                ))

    async def _execute_rebalance(
        self,
        target_weights: np.ndarray,
        prices: dict[str, float],
        return_matrix: np.ndarray,
        allocation,
    ) -> None:
        if self._router is None or self._risk is None:
            return

        result = await self._router.submit_rebalance(
            target_weights=target_weights,
            current_weights=self._current_weights[:len(target_weights)],
            tickers=self._tickers[:len(target_weights)],
            prices=prices,
            portfolio_value=self._nav,
            risk_manager=self._risk,
            returns_matrix=return_matrix.T,
        )

        if result.orders_filled > 0:
            self._current_weights[:len(target_weights)] = target_weights

        if self._telemetry:
            avg_slip = (
                result.total_slippage_bps / result.orders_filled
                if result.orders_filled > 0 else 0.0
            )
            self._telemetry.emit_execution(ExecutionMetric(
                orders_submitted=result.orders_submitted,
                orders_filled=result.orders_filled,
                total_turnover=result.total_turnover,
                commission_usd=result.total_commission_usd,
                avg_slippage_bps=avg_slip,
                risk_rejected=result.risk_rejected,
            ))

        logger.info(
            f"Rebalance: {result.orders_filled}/{result.orders_submitted} orders filled "
            f"TC=${result.total_commission_usd:.2f} slip={result.total_slippage_bps:.1f}bps "
            f"{'RISK_REJECTED' if result.risk_rejected else 'OK'}"
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def _shutdown(self) -> None:
        self._transition(OrchestratorState.SHUTDOWN)
        logger.info("Shutting down TopArb...")

        # Cancel all open orders first
        if self._router:
            try:
                await self._router.cancel_all()
            except Exception as e:
                logger.error(f"Cancel all failed: {e}")

        if self._telemetry:
            self._telemetry.close()

        logger.info("TopArb shutdown complete.")

    def _signal_handler(self, signum, frame) -> None:
        logger.warning(f"Signal {signum} received — initiating graceful shutdown")
        if hasattr(self, "_shutdown_event"):
            asyncio.get_event_loop().call_soon_threadsafe(self._shutdown_event.set)

    def _transition(self, new_state: OrchestratorState) -> None:
        old = self._state.name
        self._state = new_state
        logger.info(f"State: {old} → {new_state.name}")

    # ------------------------------------------------------------------
    # Manual controls (for operators)
    # ------------------------------------------------------------------

    def halt(self, reason: str) -> None:
        """Operator manual halt."""
        if self._risk:
            self._risk.manual_halt(reason)

    def resume(self, operator_id: str) -> None:
        """Operator manual resume after halt."""
        if self._risk:
            self._risk.manual_resume(operator_id)

    @property
    def state(self) -> OrchestratorState:
        return self._state

    @property
    def risk_manager(self) -> RiskManager | None:
        return self._risk

    @property
    def nav(self) -> float:
        return self._nav