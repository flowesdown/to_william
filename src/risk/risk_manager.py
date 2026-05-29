from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final

import numpy as np

from config.settings import RiskConfig

logger = logging.getLogger(__name__)

_ABSOLUTE_MAX_LEVERAGE: Final[float] = 4.0
_ABSOLUTE_MAX_POSITION: Final[float] = 0.35
_ABSOLUTE_MIN_CASH_FRAC: Final[float] = 0.05
_MAX_DAILY_TURNOVER: Final[float] = 0.50   # 50% of portfolio per day


class RiskState(Enum):
    ACTIVE = auto()
    HALTED = auto()


class RejectionReason(Enum):
    KILL_SWITCH = "kill_switch_active"
    LEVERAGE_HARD_CAP = "leverage_exceeds_hard_cap"
    LEVERAGE_CONFIG_CAP = "leverage_exceeds_config_cap"
    POSITION_TOO_LARGE = "single_position_too_large"
    VAR_BUDGET_BREACH = "var_budget_exceeded"
    DAILY_LOSS_LIMIT = "daily_loss_limit_reached"
    DRAWDOWN_HALT = "drawdown_circuit_breaker"
    ORDER_TOO_LARGE = "order_size_exceeds_limit"
    ORDER_TOO_SMALL = "order_size_below_minimum"
    TURNOVER_LIMIT = "daily_turnover_limit_exceeded"
    INSUFFICIENT_DATA = "insufficient_data_for_validation"
    MARKET_HOURS = "outside_allowed_market_hours"
    RATE_LIMIT = "order_rate_limit_exceeded"


@dataclass(frozen=True)
class ValidationResult:
    approved: bool
    rejection_reason: RejectionReason | None
    scaled_weights: np.ndarray | None
    original_leverage: float
    approved_leverage: float
    var_99: float
    message: str

    @classmethod
    def approve(cls, weights: np.ndarray, leverage: float, var_99: float) -> "ValidationResult":
        return cls(
            approved=True,
            rejection_reason=None,
            scaled_weights=weights,
            original_leverage=leverage,
            approved_leverage=leverage,
            var_99=var_99,
            message="APPROVED",
        )

    @classmethod
    def reject(
        cls,
        reason: RejectionReason,
        original_leverage: float = 0.0,
        var_99: float = 0.0,
        message: str = "",
    ) -> "ValidationResult":
        return cls(
            approved=False,
            rejection_reason=reason,
            scaled_weights=None,
            original_leverage=original_leverage,
            approved_leverage=0.0,
            var_99=var_99,
            message=message or reason.value,
        )


@dataclass
class PortfolioSnapshot:
    nav: float
    peak_nav: float
    daily_start_nav: float
    weights: np.ndarray
    returns_history: list[float]
    last_order_ts: float = 0.0
    daily_turnover: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class RiskMetrics:
    leverage: float
    var_99: float
    cvar_99: float
    max_position: float
    daily_pnl_frac: float
    drawdown_frac: float
    daily_turnover: float
    timestamp: float = field(default_factory=time.time)


class RiskManager:
    """
    Enforces all risk limits. Must be called before every allocation change.

    State machine:  ACTIVE → (limit breach) → HALTED → (manual_resume) → ACTIVE
    """

    def __init__(
        self,
        config: RiskConfig,
        n_assets: int,
        audit_logger: logging.Logger | None = None,
    ) -> None:
        self._cfg = config
        self._n_assets = n_assets
        self._state = RiskState.ACTIVE
        self._halt_reason: str = ""
        self._audit = audit_logger or logging.getLogger("toparb.audit")

        self._peak_nav: float = 1.0
        self._daily_start_nav: float = 1.0
        self._current_nav: float = 1.0
        self._daily_turnover: float = 0.0
        self._last_order_ts: float = 0.0
        self._returns_history: list[float] = []
        self._current_weights: np.ndarray = np.zeros(n_assets)

    def validate_allocation(
        self,
        target_weights: np.ndarray,
        portfolio_value: float,
        returns_matrix: np.ndarray | None = None,
    ) -> ValidationResult:
        # 1. Kill switch — unconditional first check
        if self.is_halted:
            return ValidationResult.reject(
                RejectionReason.KILL_SWITCH,
                message=f"System halted: {self._halt_reason}",
            )

        # 2. Basic sanity
        if target_weights is None or len(target_weights) == 0:
            return ValidationResult.reject(
                RejectionReason.INSUFFICIENT_DATA, message="Empty weights vector"
            )

        weights = np.asarray(target_weights, dtype=np.float64)
        if np.isnan(weights).any() or np.isinf(weights).any():
            return ValidationResult.reject(
                RejectionReason.INSUFFICIENT_DATA, message="NaN/Inf in weights vector"
            )

        # 3. Stale quotes check
        if returns_matrix is not None:
            latest_rets = returns_matrix[:, -1]
            if np.all(np.abs(latest_rets) < 1e-8) and time.time() - self._last_order_ts > 300:
                return ValidationResult.reject(
                    RejectionReason.INSUFFICIENT_DATA,
                    message="Stale quotes detected: all returns zero for >5 minutes",
                )

        # 4. Hard leverage cap (absolute ceiling, never negotiable)
        leverage = float(np.sum(np.abs(weights)))
        if leverage > _ABSOLUTE_MAX_LEVERAGE:
            result = ValidationResult.reject(
                RejectionReason.LEVERAGE_HARD_CAP,
                original_leverage=leverage,
                message=f"Leverage {leverage:.2f} > absolute cap {_ABSOLUTE_MAX_LEVERAGE}",
            )
            self._audit_log("REJECT", result)
            return result

        # 5. Config leverage cap (soft — scale down)
        if leverage > self._cfg.max_leverage:
            scale = self._cfg.max_leverage / leverage
            weights = weights * scale
            leverage = float(np.sum(np.abs(weights)))
            logger.info(f"Weights scaled to leverage cap: {leverage:.3f}")

        # 6. Single position limit
        max_pos = float(np.max(np.abs(weights)))
        if max_pos > _ABSOLUTE_MAX_POSITION:
            result = ValidationResult.reject(
                RejectionReason.POSITION_TOO_LARGE,
                original_leverage=leverage,
                message=f"Max position {max_pos:.3f} > absolute cap {_ABSOLUTE_MAX_POSITION}",
            )
            self._audit_log("REJECT", result)
            return result

        config_pos_limit = min(self._cfg.max_single_position_frac, _ABSOLUTE_MAX_POSITION)
        if np.any(np.abs(weights) > config_pos_limit):
            weights = np.where(
                np.abs(weights) > config_pos_limit,
                np.sign(weights) * config_pos_limit,
                weights,
            )
            leverage = float(np.sum(np.abs(weights)))

        # 7. VaR check
        var_99 = self._estimate_var(weights, portfolio_value, returns_matrix)
        var_budget = self._cfg.var_budget_frac * portfolio_value
        if var_99 > var_budget:
            scale = var_budget / (var_99 + 1e-10)
            weights = weights * scale
            leverage = float(np.sum(np.abs(weights)))
            var_99 = var_99 * scale
            logger.info(f"Weights scaled to VaR budget: VaR={var_99:.2f}")

        # 8. Daily loss circuit breaker
        daily_pnl_frac = (self._current_nav - self._daily_start_nav) / max(self._daily_start_nav, 1e-10)
        if daily_pnl_frac < -self._cfg.daily_loss_limit_frac:
            self._halt(f"Daily loss {daily_pnl_frac:.2%} exceeds limit")
            result = ValidationResult.reject(
                RejectionReason.DAILY_LOSS_LIMIT,
                original_leverage=leverage,
                var_99=var_99,
                message=f"Daily loss {daily_pnl_frac:.2%} → HALTED",
            )
            self._audit_log("HALT+REJECT", result)
            return result

        # 9. Drawdown circuit breaker
        drawdown = (self._peak_nav - self._current_nav) / max(self._peak_nav, 1e-10)
        if drawdown > self._cfg.drawdown_halt_frac:
            self._halt(f"Drawdown {drawdown:.2%} exceeds halt threshold")
            result = ValidationResult.reject(
                RejectionReason.DRAWDOWN_HALT,
                original_leverage=leverage,
                var_99=var_99,
                message=f"Drawdown {drawdown:.2%} → HALTED",
            )
            self._audit_log("HALT+REJECT", result)
            return result

        # 10. Order rate limit
        elapsed = time.time() - self._last_order_ts
        if self._last_order_ts > 0 and elapsed < self._cfg.min_seconds_between_orders:
            result = ValidationResult.reject(
                RejectionReason.RATE_LIMIT,
                original_leverage=leverage,
                var_99=var_99,
                message=f"Rate limit: {elapsed:.1f}s since last order",
            )
            self._audit_log("REJECT", result)
            return result

        # 11. Turnover check
        turnover = self._estimate_turnover(weights)
        remaining_turnover = _MAX_DAILY_TURNOVER - self._daily_turnover
        if turnover > remaining_turnover:
            result = ValidationResult.reject(
                RejectionReason.TURNOVER_LIMIT,
                original_leverage=leverage,
                var_99=var_99,
                message=f"Turnover {turnover:.2%} would exceed daily limit {_MAX_DAILY_TURNOVER:.0%}",
            )
            self._audit_log("REJECT", result)
            return result

        result = ValidationResult.approve(weights, leverage, var_99)
        self._audit_log("APPROVE", result)
        return result

    def validate_order_size(self, order_usd: float) -> tuple[bool, str]:
        if order_usd < self._cfg.min_order_usd:
            return False, f"Order ${order_usd:.0f} < minimum ${self._cfg.min_order_usd:.0f}"
        if order_usd > self._cfg.max_order_usd:
            return False, f"Order ${order_usd:.0f} > maximum ${self._cfg.max_order_usd:.0f}"
        return True, "ok"

    def update_nav(self, new_nav: float) -> None:
        self._current_nav = new_nav
        if new_nav > self._peak_nav:
            self._peak_nav = new_nav

    def reset_daily(self, nav: float) -> None:
        self._daily_start_nav = nav
        self._daily_turnover = 0.0
        logger.info(f"Daily reset: NAV={nav:.2f}, peak={self._peak_nav:.2f}")

    def record_execution(self, old_weights: np.ndarray, new_weights: np.ndarray) -> None:
        turnover = float(np.sum(np.abs(new_weights - old_weights)))
        self._daily_turnover += turnover
        self._last_order_ts = time.time()
        self._current_weights = new_weights.copy()

    def compute_risk_metrics(
        self,
        weights: np.ndarray,
        portfolio_value: float,
        returns_matrix: np.ndarray | None = None,
    ) -> RiskMetrics:
        leverage = float(np.sum(np.abs(weights)))
        var_99 = self._estimate_var(weights, portfolio_value, returns_matrix)

        if returns_matrix is not None and returns_matrix.shape[0] > 10:
            port_rets = returns_matrix @ weights
            idx = max(1, int(0.01 * len(port_rets)))
            sorted_r = np.sort(port_rets)
            cvar = float(-np.mean(sorted_r[:idx]))
        else:
            cvar = var_99 * 1.3

        daily_pnl = (self._current_nav - self._daily_start_nav) / max(self._daily_start_nav, 1e-10)
        drawdown = (self._peak_nav - self._current_nav) / max(self._peak_nav, 1e-10)

        return RiskMetrics(
            leverage=leverage,
            var_99=var_99,
            cvar_99=cvar,
            max_position=float(np.max(np.abs(weights))),
            daily_pnl_frac=daily_pnl,
            drawdown_frac=drawdown,
            daily_turnover=self._daily_turnover,
        )

    def manual_halt(self, reason: str) -> None:
        self._halt(f"Manual halt: {reason}")

    def manual_resume(self, operator_id: str) -> None:
        if self._state == RiskState.HALTED:
            drawdown = (self._peak_nav - self._current_nav) / max(self._peak_nav, 1e-10)
            daily_loss = (self._current_nav - self._daily_start_nav) / max(self._daily_start_nav, 1e-10)
            if drawdown > self._cfg.drawdown_halt_frac:
                raise RuntimeError(f"Cannot resume: drawdown {drawdown:.2%} still exceeds halt threshold")
            if daily_loss < -self._cfg.daily_loss_limit_frac:
                raise RuntimeError(f"Cannot resume: daily loss {daily_loss:.2%} still exceeds limit")
            self._state = RiskState.ACTIVE
            self._halt_reason = ""
            logger.warning(f"System RESUMED by operator {operator_id}")
            self._audit.warning(f"RESUME operator={operator_id}")

    @property
    def is_halted(self) -> bool:
        return self._state == RiskState.HALTED

    @property
    def state(self) -> RiskState:
        return self._state

    @property
    def current_nav(self) -> float:
        return self._current_nav

    @property
    def drawdown(self) -> float:
        return (self._peak_nav - self._current_nav) / max(self._peak_nav, 1e-10)

    def _halt(self, reason: str) -> None:
        self._state = RiskState.HALTED
        self._halt_reason = reason
        logger.critical(f"RISK HALT: {reason}")
        self._audit.critical(f"HALT reason={reason}")

    def _estimate_var(
        self,
        weights: np.ndarray,
        portfolio_value: float,
        returns_matrix: np.ndarray | None,
    ) -> float:
        if returns_matrix is None or returns_matrix.shape[0] < 20:
            vol = 0.02
            return portfolio_value * float(np.sum(np.abs(weights))) * vol * 2.33

        port_rets = returns_matrix @ weights
        idx = max(1, int(0.01 * len(port_rets)))
        sorted_r = np.sort(port_rets)
        var_frac = float(-sorted_r[idx])
        return portfolio_value * var_frac

    def _estimate_turnover(self, new_weights: np.ndarray) -> float:
        n = min(len(new_weights), len(self._current_weights))
        return float(np.sum(np.abs(new_weights[:n] - self._current_weights[:n])))

    def _audit_log(self, action: str, result: ValidationResult) -> None:
        self._audit.info(
            f"{action} leverage={result.original_leverage:.3f} "
            f"approved_lev={result.approved_leverage:.3f} "
            f"VaR={result.var_99:.2f} "
            f"reason={result.rejection_reason} "
            f"msg={result.message}"
        )
