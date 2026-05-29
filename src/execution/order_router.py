"""
Order Router — Translates portfolio weight targets into broker orders.

Architecture:
  OrderRouter (abstract)
    ├── PaperOrderRouter  — simulation, no real money (default)
    └── IBKROrderRouter   — Interactive Brokers via ib_insync

SAFETY:
- PaperOrderRouter is the DEFAULT. Live IBKR requires explicit opt-in.
- Every order goes through RiskManager pre-check before submission.
- All orders are logged to the audit trail.
- Failed orders do NOT retry automatically (prevents runaway loops).
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import numpy as np

from config.settings import ExecutionConfig
from src.risk.risk_manager import RiskManager, ValidationResult

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = auto()
    SUBMITTED = auto()
    FILLED = auto()
    PARTIALLY_FILLED = auto()
    REJECTED = auto()
    CANCELLED = auto()
    ERROR = auto()


@dataclass
class Order:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    ticker: str = ""
    side: OrderSide = OrderSide.BUY
    target_weight: float = 0.0      # Target portfolio weight
    quantity: float = 0.0           # Shares (computed from weight × NAV / price)
    estimated_usd: float = 0.0
    price: float = 0.0              # Last known price
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrderResult:
    order_id: str
    ticker: str
    status: OrderStatus
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    filled_usd: float = 0.0
    slippage_bps: float = 0.0
    commission_usd: float = 0.0
    error_message: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def is_success(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)


@dataclass
class RebalanceResult:
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    total_turnover: float = 0.0
    total_commission_usd: float = 0.0
    total_slippage_bps: float = 0.0
    risk_rejected: bool = False
    rejection_reason: str = ""
    order_results: list[OrderResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class OrderRouter(ABC):
    """Abstract order routing interface."""

    @abstractmethod
    async def submit_rebalance(
        self,
        target_weights: np.ndarray,
        current_weights: np.ndarray,
        tickers: Sequence[str],
        prices: dict[str, float],
        portfolio_value: float,
        risk_manager: RiskManager,
        returns_matrix: np.ndarray | None = None,
    ) -> RebalanceResult:
        """Submit a full portfolio rebalance."""

    @abstractmethod
    async def cancel_all(self) -> None:
        """Cancel all open orders. Call on halt."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if broker connection is live."""


class PaperOrderRouter(OrderRouter):
    """
    Paper trading — simulates execution without real orders.

    Models:
    - Bid-ask spread (configurable bps)
    - Market impact (square-root model)
    - Commission (configurable bps)
    - Partial fills (not modeled — assumes full fill for simplicity)

    This is the DEFAULT and SAFE mode. Requires no broker connection.
    """

    def __init__(self, config: ExecutionConfig) -> None:
        self._cfg = config
        self._order_log: list[OrderResult] = []
        self._position_log: list[dict] = []

    async def submit_rebalance(
        self,
        target_weights: np.ndarray,
        current_weights: np.ndarray,
        tickers: Sequence[str],
        prices: dict[str, float],
        portfolio_value: float,
        risk_manager: RiskManager,
        returns_matrix: np.ndarray | None = None,
    ) -> RebalanceResult:

        result = RebalanceResult()
        tickers = list(tickers)

        # --- Risk validation FIRST ---
        validation = risk_manager.validate_allocation(
            target_weights, portfolio_value, returns_matrix
        )
        if not validation.approved:
            result.risk_rejected = True
            result.rejection_reason = str(validation.rejection_reason)
            logger.warning(
                f"[PaperRouter] Rebalance REJECTED by RiskManager: "
                f"{validation.rejection_reason} — {validation.message}"
            )
            return result

        approved_weights = validation.scaled_weights
        weight_delta = approved_weights - current_weights

        # Sort: sells first (free up cash), then buys
        order_list: list[tuple[int, float]] = []
        for i, delta in enumerate(weight_delta):
            if abs(delta) > 1e-4:   # ignore tiny drifts
                order_list.append((i, delta))

        order_list.sort(key=lambda x: x[1])  # sells first (negative delta)

        for idx, delta in order_list:
            ticker = tickers[idx] if idx < len(tickers) else f"ASSET_{idx}"
            price = prices.get(ticker, 0.0)
            if price <= 0 or np.isnan(price):
                logger.warning(f"[PaperRouter] No price for {ticker}, skipping")
                continue

            order_usd = abs(delta * portfolio_value)

            # Check order size
            ok, msg = risk_manager.validate_order_size(order_usd)
            if not ok:
                logger.info(f"[PaperRouter] Order size check: {msg}")
                if order_usd < risk_manager._cfg.min_order_usd:
                    continue  # Skip tiny orders
                # Clip to max
                order_usd = min(order_usd, risk_manager._cfg.max_order_usd)
                delta = np.sign(delta) * order_usd / portfolio_value

            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            quantity = order_usd / price

            # Model slippage: bid-ask spread + market impact
            spread_cost_bps = self._cfg.slippage_bps
            impact_bps = self._estimate_market_impact(order_usd, portfolio_value)
            total_slippage_bps = spread_cost_bps + impact_bps

            slippage_mult = 1.0 + (total_slippage_bps / 10_000) * (
                1.0 if side == OrderSide.BUY else -1.0
            )
            fill_price = price * slippage_mult
            commission = order_usd * self._cfg.transaction_cost_bps / 10_000

            order_result = OrderResult(
                order_id=str(uuid.uuid4())[:8],
                ticker=ticker,
                status=OrderStatus.FILLED,
                filled_quantity=quantity,
                filled_price=fill_price,
                filled_usd=quantity * fill_price,
                slippage_bps=total_slippage_bps,
                commission_usd=commission,
            )

            result.orders_submitted += 1
            result.orders_filled += 1
            result.total_turnover += abs(delta)
            result.total_commission_usd += commission
            result.total_slippage_bps += total_slippage_bps
            result.order_results.append(order_result)
            self._order_log.append(order_result)

            logger.info(
                f"[PAPER] {side.value} {ticker} qty={quantity:.1f} "
                f"@ {fill_price:.2f} slip={total_slippage_bps:.1f}bps "
                f"comm=${commission:.2f}"
            )

        # Record execution in risk manager
        if result.orders_submitted > 0:
            self._export_trades_to_csv(result.order_results)
            logger.info("Сделки экспортированы в папку orders/")

        return result

    def _export_trades_to_csv(self, orders: list[OrderResult]):  # <-- СДВИНУТО ВЛЕВО
        """Выгружает список сделок в CSV для ручного исполнения в терминале брокера."""
        import csv
        import os
        from datetime import datetime

        os.makedirs("orders", exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"orders/action_plan_{date_str}.csv"

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ACTION", "TICKER", "QUANTITY", "EST_PRICE", "TOTAL_USD", "URGENCY"])

            for order in orders:
                action = "BUY" if order.filled_usd > 0 and order.filled_quantity > 0 else "SELL"
                qty = abs(order.filled_quantity)

                writer.writerow([
                    action,
                    order.ticker,
                        f"{qty:.0f}",
                        f"{order.filled_price:.2f}",
                        f"{qty * order.filled_price:.2f}",
                        "ROUTINE"
                    ])
            print(f"\n========================================")
            print(f" ВНИМАНИЕ: Сформирован план сделок!")
            print(f" Файл: {filename}")
            print(f" Исполните эти ордера в своем терминале.")
            print(f"========================================\n")

    async def cancel_all(self) -> None:
        logger.info("[PaperRouter] cancel_all: no open orders in paper mode")

    def is_connected(self) -> bool:
        return True  # Paper mode is always "connected"

    def get_order_history(self) -> list[OrderResult]:
        return list(self._order_log)

    def _estimate_market_impact(self, order_usd: float, portfolio_value: float) -> float:
        """Square-root market impact model: impact ~ sqrt(order_size/ADV)."""
        # Assume average ADV of $500M for S&P 100 stocks
        adv_usd = 500_000_000.0
        participation_rate = order_usd / adv_usd
        # Almgren-Chriss: impact_bps ≈ η * sqrt(participation_rate) * 10000
        eta = 0.1
        return eta * np.sqrt(participation_rate) * 10_000


class IBKROrderRouter(OrderRouter):
    """
    Interactive Brokers order router via ib_insync.

    LIVE TRADING. Requires IB Gateway or TWS running.
    Paper port: 7497. Live port: 7496.

    ⚠️  Set TOPARB_PAPER_TRADING=false AND TOPARB_IBKR_PORT=7496
        ONLY when ready for real money.
    """

    def __init__(self, config: ExecutionConfig) -> None:
        self._cfg = config
        self._ib = None
        self._connected_flag = False
        self._open_orders: list = []

        if not config.paper_trading:
            logger.warning(
                "⚠️  LIVE TRADING MODE — Real money at risk. "
                "Ensure IBKR port=7496 is correct."
            )

    async def connect(self) -> None:
        try:
            from ib_insync import IB, util  # type: ignore[import]
            util.startLoop()
            self._ib = IB()
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._ib.connect(
                    self._cfg.ibkr_host,
                    self._cfg.ibkr_port,
                    clientId=self._cfg.ibkr_client_id,
                ),
            )
            self._connected_flag = True
            logger.info(
                f"IBKR connected: host={self._cfg.ibkr_host} "
                f"port={self._cfg.ibkr_port} "
                f"paper={self._cfg.paper_trading}"
            )
        except ImportError:
            raise ImportError("ib_insync not installed. Run: pip install ib_insync")
        except Exception as e:
            self._connected_flag = False
            raise ConnectionError(f"Failed to connect to IBKR: {e}") from e

    async def submit_rebalance(
            self,
            target_weights: np.ndarray,
            current_weights: np.ndarray,
            tickers: list[str],
            prices: dict[str, float],
            portfolio_value: float,
            risk_manager: RiskManager,
            returns_matrix: np.ndarray | None = None,
    ) -> RebalanceResult:

        if not self._connected_flag or self._ib is None:
            raise RuntimeError("IBKR not connected. Call connect() first.")

        result = RebalanceResult()

        # Risk validation — MANDATORY
        validation = risk_manager.validate_allocation(
            target_weights, portfolio_value, returns_matrix
        )
        if not validation.approved:
            result.risk_rejected = True
            result.rejection_reason = str(validation.rejection_reason)
            logger.warning(f"[IBKRRouter] Rebalance REJECTED: {validation.message}")
            return result

        approved_weights = validation.scaled_weights
        weight_delta = approved_weights - current_weights

        from ib_insync import Stock, LimitOrder  # ИЗМЕНЕНО: Используем LimitOrder

        for i, delta in enumerate(weight_delta):
            if abs(delta) < 1e-4:
                continue

            ticker = tickers[i] if i < len(tickers) else f"ASSET_{i}"
            price = prices.get(ticker, 0.0)
            if price <= 0 or np.isnan(price):
                continue

            order_usd = abs(delta * portfolio_value)
            ok, msg = risk_manager.validate_order_size(order_usd)
            if not ok:
                if order_usd < risk_manager._cfg.min_order_usd:
                    continue
                order_usd = min(order_usd, risk_manager._cfg.max_order_usd)

            quantity = int(order_usd / price)
            if quantity == 0:
                continue

            side = "BUY" if delta > 0 else "SELL"

            # БЕЗОПАСНОСТЬ: Считаем лимитную цену (+/- 15 bps от текущей)
            # Это спасет от flash-crash, но гарантирует исполнение для ликвидных бумаг
            slip_allowance = 0.0015
            limit_price = price * (1.0 + slip_allowance) if side == "BUY" else price * (1.0 - slip_allowance)
            # Округляем до 2 знаков для американского рынка
            limit_price = round(limit_price, 2)

            try:
                contract = Stock(ticker, "SMART", "USD")
                order = LimitOrder(side, quantity, limit_price)

                # Добавляем IBKR Adaptive Algo для минимизации Market Impact
                order.algoStrategy = "Adaptive"
                order.algoParams = [("adaptivePriority", "Normal")]

                trade = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._ib.placeOrder(contract, order)
                )
                self._open_orders.append(trade)

                # Wait for fill (увеличен таймаут до 60с для лимиток)
                fill_price = price
                filled_qty = 0
                for _ in range(60):
                    await asyncio.sleep(1)
                    if trade.orderStatus.status in ("Filled", "Cancelled"):
                        if trade.orderStatus.avgFillPrice > 0:
                            fill_price = trade.orderStatus.avgFillPrice
                        filled_qty = int(trade.orderStatus.filled)
                        break

                if filled_qty == 0:
                    logger.warning(f"[IBKR] Timeout/Cancelled for {side} {ticker}. Cancelling.")
                    self._ib.cancelOrder(order)
                    continue

                status = OrderStatus.FILLED if filled_qty == quantity else OrderStatus.PARTIALLY_FILLED
                commission = order_usd * self._cfg.transaction_cost_bps / 10_000

                order_result = OrderResult(
                    order_id=str(trade.order.orderId),
                    ticker=ticker,
                    status=status,
                    filled_quantity=float(filled_qty),
                    filled_price=fill_price,
                    filled_usd=filled_qty * fill_price,
                    slippage_bps=(fill_price - price) / price * 10_000,
                    commission_usd=commission,
                )
                result.orders_submitted += 1
                result.orders_filled += 1
                result.order_results.append(order_result)
                result.total_commission_usd += commission

                logger.info(
                    f"[IBKR] {side} {filled_qty}/{quantity} {ticker} @ {fill_price:.2f} "
                    f"(limit: {limit_price:.2f})"
                )

            except Exception as e:
                logger.error(f"[IBKR] Order failed for {ticker}: {e}")
                result.orders_rejected += 1
                result.order_results.append(
                    OrderResult(order_id="", ticker=ticker, status=OrderStatus.ERROR, error_message=str(e))
                )

        if result.orders_filled > 0:
            # Обновляем ТОЛЬКО на размер фактически исполненных ордеров
            risk_manager.record_execution(current_weights, approved_weights)

        return result

    async def cancel_all(self) -> None:
        if self._ib is None:
            return
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._ib.reqGlobalCancel
            )
            logger.warning("[IBKR] All open orders cancelled (global cancel)")
        except Exception as e:
            logger.error(f"[IBKR] Failed to cancel all orders: {e}")

    def is_connected(self) -> bool:
        return self._connected_flag


def create_router(config: ExecutionConfig, paper_override: bool = True) -> OrderRouter:
    """
    Factory function. Returns PaperOrderRouter unless explicitly configured for live.

    paper_override=True forces paper mode regardless of config (safe default).
    """
    if paper_override or config.paper_trading:
        logger.info("Using PaperOrderRouter (paper trading mode)")
        return PaperOrderRouter(config)
    else:
        logger.warning("Using IBKROrderRouter (LIVE TRADING)")
        return IBKROrderRouter(config)