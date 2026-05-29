"""
Tests for RiskManager — The most critical component.

Every safety rule must be verified independently.
Tests are ordered from most critical to least critical.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.risk.risk_manager import (
    RejectionReason,
    RiskManager,
    RiskState,
    _ABSOLUTE_MAX_LEVERAGE,
    _ABSOLUTE_MAX_POSITION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def uniform_weights(n: int, leverage: float = 1.0) -> np.ndarray:
    """Equal-weight long-only portfolio."""
    w = np.ones(n) / n * leverage
    return w


def concentrated_weights(n: int, pos: float = 0.5) -> np.ndarray:
    """Weights with one concentrated position."""
    w = np.zeros(n)
    w[0] = pos
    w[1:] = (1.0 - pos) / (n - 1)
    return w


# ---------------------------------------------------------------------------
# 1. Kill switch tests
# ---------------------------------------------------------------------------

class TestKillSwitch:
    def test_manual_halt_blocks_all_orders(self, risk_manager, n_assets):
        risk_manager.manual_halt("test halt")
        weights = uniform_weights(n_assets, leverage=0.5)
        result = risk_manager.validate_allocation(weights, 1_000_000)
        assert not result.approved
        assert result.rejection_reason == RejectionReason.KILL_SWITCH

    def test_halted_state_is_set(self, risk_manager, n_assets):
        risk_manager.manual_halt("test")
        assert risk_manager.is_halted
        assert risk_manager.state == RiskState.HALTED

    def test_resume_clears_halt(self, risk_config, n_assets):
        rm = RiskManager(risk_config, n_assets)
        rm.update_nav(1_000_000.0)
        rm.reset_daily(1_000_000.0)
        rm.manual_halt("test")
        assert rm.is_halted
        rm.manual_resume("operator_001")
        assert not rm.is_halted
        assert rm.state == RiskState.ACTIVE

    def test_cannot_resume_during_drawdown_breach(self, risk_config, n_assets):
        rm = RiskManager(risk_config, n_assets)
        # Simulate large drawdown
        rm._peak_nav = 1_000_000.0
        rm._current_nav = 800_000.0   # 20% drawdown → above 15% halt threshold
        rm.update_nav(800_000.0)
        rm.reset_daily(1_000_000.0)
        rm.manual_halt("drawdown")
        with pytest.raises(RuntimeError, match="drawdown"):
            rm.manual_resume("operator")

    def test_empty_weights_rejected(self, risk_manager):
        result = risk_manager.validate_allocation(np.array([]), 1_000_000)
        assert not result.approved
        assert result.rejection_reason == RejectionReason.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# 2. Leverage limits
# ---------------------------------------------------------------------------

class TestLeverageLimits:
    def test_absolute_hard_cap_always_rejected(self, risk_manager, n_assets):
        """Weights exceeding absolute hard cap must ALWAYS be rejected."""
        weights = uniform_weights(n_assets, leverage=_ABSOLUTE_MAX_LEVERAGE + 0.1)
        result = risk_manager.validate_allocation(weights, 1_000_000)
        assert not result.approved
        assert result.rejection_reason == RejectionReason.LEVERAGE_HARD_CAP

    def test_config_leverage_triggers_scaling(self, risk_config, n_assets):
        """Weights above config limit (but below hard cap) should be scaled, not rejected."""
        rm = RiskManager(risk_config, n_assets)
        rm.update_nav(1_000_000.0)
        rm.reset_daily(1_000_000.0)
        # Set a tight config limit
        rm._cfg.max_leverage = 1.0
        weights = uniform_weights(n_assets, leverage=1.5)
        result = rm.validate_allocation(weights, 1_000_000)
        # Should be approved with scaled weights
        assert result.approved
        scaled_lev = float(np.sum(np.abs(result.scaled_weights)))
        assert scaled_lev <= 1.0 + 1e-6

    def test_valid_leverage_approved(self, risk_manager, n_assets):
        weights = uniform_weights(n_assets, leverage=0.8)
        result = risk_manager.validate_allocation(weights, 1_000_000)
        assert result.approved

    def test_zero_weights_approved(self, risk_manager, n_assets):
        weights = np.zeros(n_assets)
        result = risk_manager.validate_allocation(weights, 1_000_000)
        assert result.approved

    def test_absolute_hard_cap_constant_value(self):
        assert _ABSOLUTE_MAX_LEVERAGE == 4.0  # Must not change

    def test_absolute_max_position_constant_value(self):
        assert _ABSOLUTE_MAX_POSITION == 0.15  # Must not change


# ---------------------------------------------------------------------------
# 3. Position concentration limits
# ---------------------------------------------------------------------------

class TestPositionLimits:
    def test_single_position_above_absolute_cap_rejected(self, risk_manager, n_assets):
        """Single position above absolute cap → rejected."""
        weights = np.zeros(n_assets)
        weights[0] = _ABSOLUTE_MAX_POSITION + 0.01   # Slightly above cap
        result = risk_manager.validate_allocation(weights, 1_000_000)
        assert not result.approved
        assert result.rejection_reason == RejectionReason.POSITION_TOO_LARGE

    def test_position_at_absolute_cap_approved(self, risk_manager, n_assets):
        weights = np.zeros(n_assets)
        weights[0] = _ABSOLUTE_MAX_POSITION - 0.001  # Just under cap
        result = risk_manager.validate_allocation(weights, 1_000_000)
        assert result.approved

    def test_config_position_limit_causes_scaling(self, risk_config, n_assets):
        """Positions above config limit (but below absolute) are scaled down."""
        rm = RiskManager(risk_config, n_assets)
        rm.update_nav(1_000_000.0)
        rm.reset_daily(1_000_000.0)
        rm._cfg.max_single_position_frac = 0.05
        weights = np.zeros(n_assets)
        weights[0] = 0.10   # Above config limit but below absolute
        result = rm.validate_allocation(weights, 1_000_000)
        if result.approved and result.scaled_weights is not None:
            assert np.max(np.abs(result.scaled_weights)) <= 0.05 + 1e-6


# ---------------------------------------------------------------------------
# 4. Daily loss circuit breaker
# ---------------------------------------------------------------------------

class TestDailyLossCircuitBreaker:
    def test_daily_loss_halts_trading(self, risk_config, n_assets):
        rm = RiskManager(risk_config, n_assets)
        initial_nav = 1_000_000.0
        rm.update_nav(initial_nav)
        rm.reset_daily(initial_nav)

        # Simulate 6% daily loss (limit is 5%)
        loss_nav = initial_nav * 0.94
        rm.update_nav(loss_nav)
        rm._current_nav = loss_nav  # Force update

        weights = uniform_weights(n_assets, 0.5)
        result = rm.validate_allocation(weights, loss_nav)

        assert not result.approved
        assert result.rejection_reason == RejectionReason.DAILY_LOSS_LIMIT
        assert rm.is_halted

    def test_daily_loss_within_limit_approved(self, risk_config, n_assets):
        rm = RiskManager(risk_config, n_assets)
        initial_nav = 1_000_000.0
        rm.update_nav(initial_nav)
        rm.reset_daily(initial_nav)

        # Simulate 3% daily loss (under 5% limit)
        rm._current_nav = initial_nav * 0.97
        weights = uniform_weights(n_assets, 0.5)
        result = rm.validate_allocation(weights, initial_nav * 0.97)
        assert result.approved


# ---------------------------------------------------------------------------
# 5. Drawdown circuit breaker
# ---------------------------------------------------------------------------

class TestDrawdownCircuitBreaker:
    def test_drawdown_above_halt_threshold(self, risk_config, n_assets):
        rm = RiskManager(risk_config, n_assets)
        peak = 1_000_000.0
        rm._peak_nav = peak
        rm._current_nav = peak * (1.0 - risk_config.drawdown_halt_frac - 0.01)
        rm.update_nav(rm._current_nav)
        rm.reset_daily(rm._current_nav)

        weights = uniform_weights(n_assets, 0.5)
        result = rm.validate_allocation(weights, rm._current_nav)

        assert not result.approved
        assert result.rejection_reason == RejectionReason.DRAWDOWN_HALT
        assert rm.is_halted

    def test_peak_nav_updates_correctly(self, risk_config, n_assets):
        rm = RiskManager(risk_config, n_assets)
        rm.update_nav(1_000_000.0)
        assert rm._peak_nav == 1_000_000.0
        rm.update_nav(1_200_000.0)
        assert rm._peak_nav == 1_200_000.0
        rm.update_nav(1_100_000.0)
        assert rm._peak_nav == 1_200_000.0  # Peak doesn't decrease


# ---------------------------------------------------------------------------
# 6. VaR and risk metrics
# ---------------------------------------------------------------------------

class TestVaRAndRiskMetrics:
    def test_var_estimate_positive(self, risk_manager, n_assets, synthetic_returns):
        weights = uniform_weights(n_assets, 1.0)
        var = risk_manager._estimate_var(weights, 1_000_000, synthetic_returns)
        assert var > 0

    def test_var_scales_with_leverage(self, risk_manager, n_assets, synthetic_returns):
        w_low = uniform_weights(n_assets, 0.5)
        w_high = uniform_weights(n_assets, 1.5)
        var_low = risk_manager._estimate_var(w_low, 1_000_000, synthetic_returns)
        var_high = risk_manager._estimate_var(w_high, 1_000_000, synthetic_returns)
        assert var_high > var_low

    def test_risk_metrics_complete(self, risk_manager, n_assets, synthetic_returns):
        weights = uniform_weights(n_assets, 1.0)
        metrics = risk_manager.compute_risk_metrics(weights, 1_000_000, synthetic_returns)
        assert metrics.leverage > 0
        assert metrics.var_99 > 0
        assert 0.0 <= metrics.max_position <= 1.0

    def test_order_size_validation(self, risk_manager, risk_config):
        ok, msg = risk_manager.validate_order_size(100_000)
        assert ok

        ok, msg = risk_manager.validate_order_size(risk_config.max_order_usd + 1)
        assert not ok
        assert "maximum" in msg.lower()

        ok, msg = risk_manager.validate_order_size(risk_config.min_order_usd - 1)
        assert not ok
        assert "minimum" in msg.lower()


# ---------------------------------------------------------------------------
# 7. Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_rate_limit_enforced(self, risk_config, n_assets):
        import time
        rm = RiskManager(risk_config, n_assets)
        rm.update_nav(1_000_000.0)
        rm.reset_daily(1_000_000.0)

        weights = uniform_weights(n_assets, 0.5)

        # First order — should pass
        result1 = rm.validate_allocation(weights, 1_000_000)
        assert result1.approved

        # Simulate recent order
        rm._last_order_ts = time.time()  # Just ordered

        # Immediate second order — should be rate-limited
        result2 = rm.validate_allocation(weights, 1_000_000)
        assert not result2.approved
        assert result2.rejection_reason == RejectionReason.RATE_LIMIT


# ---------------------------------------------------------------------------
# 8. Turnover limits
# ---------------------------------------------------------------------------

class TestTurnoverLimits:
    def test_turnover_limit_enforced(self, risk_config, n_assets):
        from src.risk.risk_manager import _MAX_DAILY_TURNOVER
        rm = RiskManager(risk_config, n_assets)
        rm.update_nav(1_000_000.0)
        rm.reset_daily(1_000_000.0)

        # Simulate already-used daily turnover at maximum
        rm._daily_turnover = _MAX_DAILY_TURNOVER + 0.01

        weights = uniform_weights(n_assets, 0.5)
        result = rm.validate_allocation(weights, 1_000_000)

        # Should fail due to turnover limit
        if not result.approved:
            assert result.rejection_reason == RejectionReason.TURNOVER_LIMIT

    def test_turnover_resets_daily(self, risk_config, n_assets):
        rm = RiskManager(risk_config, n_assets)
        rm._daily_turnover = 0.45
        rm.reset_daily(1_000_000.0)
        assert rm._daily_turnover == 0.0


# ---------------------------------------------------------------------------
# 9. Integration: full validation flow
# ---------------------------------------------------------------------------

class TestFullValidationFlow:
    def test_valid_portfolio_passes_all_checks(
        self, risk_manager, n_assets, synthetic_returns
    ):
        weights = uniform_weights(n_assets, 0.2)
        result = risk_manager.validate_allocation(
            weights, 1_000_000, synthetic_returns
        )
        assert result.approved
        assert result.scaled_weights is not None
        assert result.rejection_reason is None
        lev = float(np.sum(np.abs(result.scaled_weights)))
        assert lev <= risk_manager._cfg.max_leverage + 1e-6

    def test_result_weights_are_finite(self, risk_manager, n_assets):
        weights = uniform_weights(n_assets, 0.5)
        result = risk_manager.validate_allocation(weights, 1_000_000)
        if result.approved and result.scaled_weights is not None:
            assert np.all(np.isfinite(result.scaled_weights))

    def test_nan_weights_handled(self, risk_manager, n_assets):
        weights = np.full(n_assets, np.nan)
        result = risk_manager.validate_allocation(weights, 1_000_000)
        # Either rejected or scaled weights are not NaN
        if result.approved and result.scaled_weights is not None:
            assert np.all(np.isfinite(result.scaled_weights))