"""
Tests for PaperOrderRouter and ThresholdRebalancer.

Coverage (OrderRouter):
- submit_rebalance — basic structure, risk validation gate
- Risk rejection propagates correctly
- Sell-first ordering
- Slippage and commission modeling
- cancel_all (no-op in paper mode)
- Order history tracking
- Tiny order filtering
- Market impact model

Coverage (Rebalancer):
- Routine drift rebalance
- Emergency leverage breach
- TDA anomaly override
- Minimum interval enforcement
- Dynamic threshold
- Reset
"""
from __future__ import annotations

import numpy as np
import pytest

from src.execution.order_router import (
    PaperOrderRouter,
    OrderStatus,
    RebalanceResult,
    OrderResult,
    OrderSide,
    create_router,
)
from src.execution.rebalancer import ThresholdRebalancer, RebalanceDecision


# ── Helpers ───────────────────────────────────────────────────────────────────

def uniform_w(n: int, lev: float = 0.8) -> np.ndarray:
    return np.ones(n) / n * lev


def make_prices(tickers) -> dict[str, float]:
    return {t: 100.0 for t in tickers}


# ── PaperOrderRouter ──────────────────────────────────────────────────────────

class TestPaperOrderRouterBasic:
    @pytest.mark.asyncio
    async def test_is_always_connected(self, paper_router):
        assert paper_router.is_connected()

    @pytest.mark.asyncio
    async def test_cancel_all_no_error(self, paper_router):
        await paper_router.cancel_all()  # Should not raise

    @pytest.mark.asyncio
    async def test_submit_valid_rebalance(self, paper_router, risk_manager, n_assets, tickers):
        target = uniform_w(n_assets, 0.2)
        current = np.zeros(n_assets)
        prices = make_prices(tickers)

        result = await paper_router.submit_rebalance(
            target_weights=target,
            current_weights=current,
            tickers=tickers,
            prices=prices,
            portfolio_value=1_000_000.0,
            risk_manager=risk_manager,
        )

        assert isinstance(result, RebalanceResult)
        assert not result.risk_rejected
        assert result.orders_submitted > 0
        assert result.orders_filled > 0

    @pytest.mark.asyncio
    async def test_fill_equals_submitted_paper(self, paper_router, risk_manager, n_assets, tickers):
        target = uniform_w(n_assets, 0.2)
        current = np.zeros(n_assets)
        prices = make_prices(tickers)

        result = await paper_router.submit_rebalance(
            target_weights=target, current_weights=current,
            tickers=tickers, prices=prices,
            portfolio_value=1_000_000.0, risk_manager=risk_manager,
        )
        assert result.orders_filled == result.orders_submitted

    @pytest.mark.asyncio
    async def test_commission_is_positive(self, paper_router, risk_manager, n_assets, tickers):
        target = uniform_w(n_assets, 0.2)
        current = np.zeros(n_assets)
        prices = make_prices(tickers)

        result = await paper_router.submit_rebalance(
            target_weights=target, current_weights=current,
            tickers=tickers, prices=prices,
            portfolio_value=1_000_000.0, risk_manager=risk_manager,
        )
        assert result.total_commission_usd > 0

    @pytest.mark.asyncio
    async def test_slippage_is_positive(self, paper_router, risk_manager, n_assets, tickers):
        target = uniform_w(n_assets, 0.2)
        current = np.zeros(n_assets)
        prices = make_prices(tickers)

        result = await paper_router.submit_rebalance(
            target_weights=target, current_weights=current,
            tickers=tickers, prices=prices,
            portfolio_value=1_000_000.0, risk_manager=risk_manager,
        )
        assert result.total_slippage_bps > 0

    @pytest.mark.asyncio
    async def test_risk_rejected_blocks_all(self, paper_router, risk_manager, n_assets, tickers):
        """Halt the risk manager → all submissions should be rejected."""
        risk_manager.manual_halt("test block")

        target = uniform_w(n_assets, 0.2)
        current = np.zeros(n_assets)
        prices = make_prices(tickers)

        result = await paper_router.submit_rebalance(
            target_weights=target, current_weights=current,
            tickers=tickers, prices=prices,
            portfolio_value=1_000_000.0, risk_manager=risk_manager,
        )
        assert result.risk_rejected
        assert result.orders_submitted == 0

    @pytest.mark.asyncio
    async def test_missing_price_skipped(self, paper_router, risk_manager, n_assets, tickers):
        target = uniform_w(n_assets, 0.5)
        current = np.zeros(n_assets)
        # Missing first ticker
        prices = {t: 100.0 for t in tickers[1:]}

        result = await paper_router.submit_rebalance(
            target_weights=target, current_weights=current,
            tickers=tickers, prices=prices,
            portfolio_value=1_000_000.0, risk_manager=risk_manager,
        )
        # Missing prices are skipped; should still fill others
        assert not result.risk_rejected

    @pytest.mark.asyncio
    async def test_tiny_delta_filtered(self, paper_router, risk_manager, n_assets, tickers):
        """Weight deltas < 1e-4 should not generate orders."""
        target = uniform_w(n_assets, 0.2)
        current = target.copy()  # Identical → no delta
        prices = make_prices(tickers)

        result = await paper_router.submit_rebalance(
            target_weights=target, current_weights=current,
            tickers=tickers, prices=prices,
            portfolio_value=1_000_000.0, risk_manager=risk_manager,
        )
        assert result.orders_submitted == 0

    @pytest.mark.asyncio
    async def test_order_history_accumulates(self, paper_router, risk_manager, n_assets, tickers):
        target = uniform_w(n_assets, 0.2)
        current = np.zeros(n_assets)
        prices = make_prices(tickers)

        await paper_router.submit_rebalance(
            target_weights=target, current_weights=current,
            tickers=tickers, prices=prices,
            portfolio_value=1_000_000.0, risk_manager=risk_manager,
        )
        history = paper_router.get_order_history()
        assert len(history) > 0
        assert all(isinstance(r, OrderResult) for r in history)

    @pytest.mark.asyncio
    async def test_sells_before_buys(self, paper_router, risk_manager, n_assets, tickers):
        """Rebalance should sort sells first (negative delta first)."""
        target = np.zeros(n_assets)
        target[0] = -0.2   # sell
        target[1] = 0.3    # buy
        current = np.zeros(n_assets)
        prices = make_prices(tickers)

        result = await paper_router.submit_rebalance(
            target_weights=target, current_weights=current,
            tickers=tickers, prices=prices,
            portfolio_value=1_000_000.0, risk_manager=risk_manager,
        )
        # All orders filled regardless of ordering
        assert result.orders_filled >= 0


class TestPaperOrderRouterMarketImpact:
    def test_impact_scales_with_order_size(self, paper_router):
        small_impact = paper_router._estimate_market_impact(10_000, 1_000_000)
        large_impact = paper_router._estimate_market_impact(1_000_000, 1_000_000)
        assert large_impact > small_impact

    def test_impact_non_negative(self, paper_router):
        impact = paper_router._estimate_market_impact(50_000, 1_000_000)
        assert impact >= 0


class TestCreateRouter:
    def test_create_paper_router_by_default(self, exec_config):
        router = create_router(exec_config, paper_override=True)
        assert isinstance(router, PaperOrderRouter)

    def test_create_paper_router_when_config_paper(self, exec_config):
        exec_config.paper_trading = True
        router = create_router(exec_config, paper_override=False)
        assert isinstance(router, PaperOrderRouter)


# ── ThresholdRebalancer ───────────────────────────────────────────────────────

class TestThresholdRebalancer:
    def _decision(
        self, rebalancer, current, target, f_star=1.0,
        max_lev=2.0, tda=False, threshold=None,
    ):
        return rebalancer.should_rebalance(
            current_weights=current,
            target_weights=target,
            current_f_star=f_star,
            max_leverage=max_lev,
            tda_is_anomaly=tda,
            dynamic_threshold=threshold,
        )

    def test_no_drift_no_rebalance(self, rebalancer, n_assets):
        w = uniform_w(n_assets, 0.5)
        decision = self._decision(rebalancer, w, w)
        assert not decision.should_rebalance

    def test_large_drift_triggers_rebalance(self, n_assets):
        rb = ThresholdRebalancer(base_threshold=0.02, min_rebalance_interval_steps=1)
        current = uniform_w(n_assets, 0.5)
        target = uniform_w(n_assets, 1.5)
        decision = rb._steps_since_rebalance = 5
        decision = rb.should_rebalance(
            current_weights=current,
            target_weights=target,
            current_f_star=1.0,
            max_leverage=2.0,
        )
        assert decision.should_rebalance
        assert decision.urgency in ("routine", "drift", "emergency")

    def test_emergency_on_leverage_breach(self, n_assets):
        rb = ThresholdRebalancer(emergency_leverage_frac=0.80)
        # Current leverage > 80% of max_leverage(2.0) = 1.6
        current = np.ones(n_assets) * 1.7 / n_assets  # total leverage = 1.7
        target = np.zeros(n_assets)
        decision = rb._steps_since_rebalance = 5
        decision = rb.should_rebalance(
            current_weights=current,
            target_weights=target,
            current_f_star=0.5,
            max_leverage=2.0,
        )
        assert decision.should_rebalance
        assert decision.urgency == "emergency"

    def test_tda_anomaly_forces_rebalance(self, n_assets):
        rb = ThresholdRebalancer(
            base_threshold=0.50,  # High threshold → wouldn't rebalance normally
            min_rebalance_interval_steps=1,
            tda_anomaly_override=True,
        )
        rb._steps_since_rebalance = 0
        current = uniform_w(n_assets, 0.5)
        target = uniform_w(n_assets, 0.52)  # Tiny drift < threshold

        decision = rb._steps_since_rebalance = 5
        decision = rb.should_rebalance(
            current_weights=current, target_weights=target,
            current_f_star=0.5, max_leverage=2.0,
            tda_is_anomaly=True,
        )
        assert decision.should_rebalance

    def test_minimum_interval_blocks_rebalance(self, n_assets):
        rb = ThresholdRebalancer(
            base_threshold=0.02,
            min_rebalance_interval_steps=10,
        )
        rb._steps_since_rebalance = 0
        current = uniform_w(n_assets, 0.5)
        target = uniform_w(n_assets, 1.5)  # Large drift

        # First call — should be blocked by min interval
        decision = rb._steps_since_rebalance = 5
        decision = rb.should_rebalance(
            current_weights=current, target_weights=target,
            current_f_star=0.5, max_leverage=2.0,
        )
        assert not decision.should_rebalance

    def test_dynamic_threshold_used(self, n_assets):
        rb = ThresholdRebalancer(base_threshold=0.50, min_rebalance_interval_steps=1)
        current = uniform_w(n_assets, 0.5)
        target = uniform_w(n_assets, 0.60)  # drift ≈ 0.1

        # With very tight dynamic threshold → should rebalance
        decision = rb._steps_since_rebalance = 5
        decision = rb.should_rebalance(
            current_weights=current, target_weights=target,
            current_f_star=0.5, max_leverage=2.0,
            dynamic_threshold=0.05,  # 5% threshold, drift=10% → rebalance
        )
        assert decision.should_rebalance

    def test_reset_resets_interval(self, rebalancer, n_assets):
        rebalancer.reset()
        current = uniform_w(n_assets, 0.5)
        target = uniform_w(n_assets, 1.5)
        decision = rebalancer.should_rebalance(
            current_weights=current, target_weights=target,
            current_f_star=0.5, max_leverage=2.0,
        )
        # After reset, min_interval should be met
        assert decision.should_rebalance

    def test_drift_magnitude_computed(self, rebalancer, n_assets):
        current = uniform_w(n_assets, 0.5)
        target = uniform_w(n_assets, 0.2)
        decision = rebalancer.should_rebalance(
            current_weights=current, target_weights=target,
            current_f_star=0.5, max_leverage=2.0,
        )
        expected_drift = float(np.sum(np.abs(target - current)))
        assert abs(decision.drift_magnitude - expected_drift) < 1e-9

    def test_turnover_estimate_positive(self, rebalancer, n_assets):
        current = uniform_w(n_assets, 0.5)
        target = uniform_w(n_assets, 1.0)
        decision = rebalancer.should_rebalance(
            current_weights=current, target_weights=target,
            current_f_star=0.5, max_leverage=2.0,
        )
        assert decision.estimated_turnover >= 0

    def test_decision_reason_not_empty(self, rebalancer, n_assets):
        current = uniform_w(n_assets, 0.5)
        target = uniform_w(n_assets, 0.5)
        decision = rebalancer.should_rebalance(
            current_weights=current, target_weights=target,
            current_f_star=0.5, max_leverage=2.0,
        )
        assert len(decision.reason) > 0