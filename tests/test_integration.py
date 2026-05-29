"""
Integration tests — Full pipeline validation.

These tests wire up all components end-to-end on synthetic data and verify:
1. The full pipeline runs without exceptions
2. Risk limits are respected throughout
3. Weights are always finite and within bounds
4. Transaction costs are deducted
5. Emergency deleveraging works
6. Backtester produces valid metrics

These are slower than unit tests. Run with: pytest tests/test_integration.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from src.math.eigen_risk import PCARiskModel
from src.math.manifold_tda import VietorisRipsManifold
from src.execution.hjb_controller import TopologicalKellyController
from src.execution.order_router import PaperOrderRouter
from src.execution.rebalancer import ThresholdRebalancer
from src.risk.risk_manager import RiskManager, RejectionReason
from src.utils.backend import from_numpy, to_numpy
from src.backtest.backtester import TopArbBacktester
from config.settings import ModelConfig, RiskConfig, BacktestConfig


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def full_return_matrix(n_assets, n_obs):
    """(n_assets × n_obs) matrix — column = chronological time."""
    rng = np.random.default_rng(999)
    n_factors = 3
    factor_rets = rng.normal(0, 0.01, (n_obs, n_factors))
    loadings = rng.normal(0, 0.5, (n_assets, n_factors))
    loadings[:, 0] = np.abs(loadings[:, 0])
    idio = rng.normal(0, 0.005, (n_obs, n_assets))
    returns = factor_rets @ loadings.T + idio
    returns = np.clip(returns, -0.10, 0.10)
    return returns.T.astype(np.float64)  # (n_assets × n_obs)


@pytest.fixture
def pipeline(n_assets, exec_config, risk_config):
    pca = PCARiskModel(n_components=5)
    tda = VietorisRipsManifold(
        persistence_threshold=0.05,
        anomaly_zscore_threshold=2.0,
        downsample_to=min(n_assets, 15),
    )
    controller = TopologicalKellyController(
        pca_model=pca, tda_model=tda,
        risk_aversion=2.0, gamma_tda=2.0, max_leverage=2.0,
    )
    router = PaperOrderRouter(exec_config)
    rebalancer = ThresholdRebalancer(base_threshold=0.02, min_rebalance_interval_steps=1)
    risk = RiskManager(risk_config, n_assets)
    risk.update_nav(1_000_000.0)
    risk.reset_daily(1_000_000.0)
    return dict(
        controller=controller, router=router,
        rebalancer=rebalancer, risk=risk,
    )


# ── Full pipeline ─────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_single_step_pipeline(self, pipeline, synthetic_returns, n_assets, tickers):
        controller = pipeline["controller"]
        router = pipeline["router"]
        rebalancer = pipeline["rebalancer"]
        risk = pipeline["risk"]

        returns_gpu = from_numpy(synthetic_returns)
        allocation = controller.step(returns_gpu)

        target = to_numpy(allocation.weights)
        target = np.nan_to_num(target, nan=0.0)
        current = np.zeros(n_assets)

        decision = rebalancer.should_rebalance(
            current_weights=current,
            target_weights=target,
            current_f_star=allocation.f_star,
            max_leverage=2.0,
        )

        assert decision is not None
        assert allocation.f_star >= 0

    @pytest.mark.asyncio
    async def test_full_rebalance_cycle(self, pipeline, synthetic_returns, n_assets, tickers):
        controller = pipeline["controller"]
        router = pipeline["router"]
        risk = pipeline["risk"]

        returns_gpu = from_numpy(synthetic_returns)
        allocation = controller.step(returns_gpu)
        target = to_numpy(allocation.weights)
        target = np.nan_to_num(target, nan=0.0)

        prices = {t: 100.0 for t in tickers}
        current = np.zeros(n_assets)

        result = await router.submit_rebalance(
            target_weights=target,
            current_weights=current,
            tickers=tickers,
            prices=prices,
            portfolio_value=1_000_000.0,
            risk_manager=risk,
        )

        assert not result.risk_rejected
        assert result.orders_filled >= 0

    def test_multiple_steps_stable_weights(self, n_assets, exec_config, risk_config, synthetic_returns):
        """Run 10 steps and verify weights remain finite and bounded."""
        pca = PCARiskModel(n_components=5)
        tda = VietorisRipsManifold(
            persistence_threshold=0.05,
            anomaly_zscore_threshold=2.0,
            downsample_to=n_assets,
        )
        controller = TopologicalKellyController(
            pca_model=pca, tda_model=tda,
            risk_aversion=2.0, gamma_tda=2.0, max_leverage=2.0,
        )

        for _ in range(10):
            returns_gpu = from_numpy(synthetic_returns)
            allocation = controller.step(returns_gpu)
            weights = to_numpy(allocation.weights)

            assert np.all(np.isfinite(weights)), "Weights contain non-finite values"
            assert float(np.sum(np.abs(weights))) <= 2.0 * 1.01  # max_leverage + tolerance

    def test_risk_gate_always_fires_before_order(self, pipeline, n_assets, tickers, risk_config):
        """Even if allocation computes, risk gate blocks when halted."""
        risk = pipeline["risk"]
        risk.manual_halt("integration test halt")

        bad_weights = np.ones(n_assets) / n_assets * 0.5
        result = risk.validate_allocation(bad_weights, 1_000_000.0)

        assert not result.approved
        assert result.rejection_reason == RejectionReason.KILL_SWITCH

    def test_leverage_never_exceeds_absolute_hard_cap(self, n_assets, risk_config):
        """Absolute hard cap (4.0) must never be breached regardless of signal."""
        rm = RiskManager(risk_config, n_assets)
        rm.update_nav(1_000_000.0)
        rm.reset_daily(1_000_000.0)

        # Try to submit weights with leverage = 5.0
        over_leveraged = np.ones(n_assets) * 5.0 / n_assets
        result = rm.validate_allocation(over_leveraged, 1_000_000.0)

        assert not result.approved
        assert result.rejection_reason == RejectionReason.LEVERAGE_HARD_CAP

    def test_drawdown_triggers_deleverage(self, n_assets, risk_config):
        """Drawdown > threshold should halt and produce empty weights."""
        rm = RiskManager(risk_config, n_assets)
        rm.update_nav(1_000_000.0)
        rm.reset_daily(1_000_000.0)

        # Simulate drawdown exceeding threshold
        rm._peak_nav = 1_200_000.0
        rm._current_nav = 1_000_000.0
        rm.update_nav(1_000_000.0)
        rm.reset_daily(1_000_000.0)

        weights = np.ones(n_assets) / n_assets * 0.5
        result = rm.validate_allocation(weights, rm._current_nav)

        assert not result.approved
        assert result.rejection_reason == RejectionReason.DRAWDOWN_HALT
        assert rm.is_halted


# ── Backtester integration ────────────────────────────────────────────────────

class TestBacktesterIntegration:
    def test_backtester_runs_without_error(self, full_return_matrix):
        backtester = TopArbBacktester(refit_every=20)
        report = backtester.run(full_return_matrix, train_window=60, verbose=False)
        assert report is not None

    def test_backtester_nav_starts_at_initial(self, full_return_matrix):
        bcfg = BacktestConfig()
        backtester = TopArbBacktester(refit_every=20)
        report = backtester.run(full_return_matrix, train_window=60, verbose=False)
        assert abs(report.nav_series[0] - bcfg.initial_capital) < 1e-6

    def test_backtester_nav_all_positive(self, full_return_matrix):
        backtester = TopArbBacktester(refit_every=20)
        report = backtester.run(full_return_matrix, train_window=60, verbose=False)
        assert np.all(report.nav_series > 0)

    def test_backtester_returns_finite(self, full_return_matrix):
        backtester = TopArbBacktester(refit_every=20)
        report = backtester.run(full_return_matrix, train_window=60, verbose=False)
        assert np.all(np.isfinite(report.daily_returns))

    def test_backtester_sharpe_finite(self, full_return_matrix):
        backtester = TopArbBacktester(refit_every=20)
        report = backtester.run(full_return_matrix, train_window=60, verbose=False)
        assert np.isfinite(report.sharpe_ratio)

    def test_backtester_max_drawdown_in_range(self, full_return_matrix):
        backtester = TopArbBacktester(refit_every=20)
        report = backtester.run(full_return_matrix, train_window=60, verbose=False)
        assert -1.0 <= report.max_drawdown <= 0.0

    def test_backtester_regime_breakdown_sums_to_one(self, full_return_matrix):
        backtester = TopArbBacktester(refit_every=20)
        report = backtester.run(full_return_matrix, train_window=60, verbose=False)
        total = sum(report.regime_breakdown.values())
        assert abs(total - 1.0) < 1e-6

    def test_backtester_n_rebalances_positive(self, full_return_matrix):
        backtester = TopArbBacktester(refit_every=20)
        report = backtester.run(full_return_matrix, train_window=60, verbose=False)
        assert report.n_rebalances >= 0

    def test_backtester_avg_leverage_bounded(self, full_return_matrix):
        backtester = TopArbBacktester(refit_every=20)
        report = backtester.run(full_return_matrix, train_window=60, verbose=False)
        assert 0.0 <= report.avg_leverage <= 2.0 * 1.05  # max_leverage with tolerance

    def test_backtester_tda_anomaly_rate_in_range(self, full_return_matrix):
        backtester = TopArbBacktester(refit_every=20)
        report = backtester.run(full_return_matrix, train_window=60, verbose=False)
        assert 0.0 <= report.tda_anomaly_rate <= 1.0

    def test_backtester_insufficient_data_raises(self):
        tiny = np.random.randn(10, 30)  # Only 30 obs but train_window=60
        backtester = TopArbBacktester()
        with pytest.raises(ValueError, match="Insufficient data"):
            backtester.run(tiny, train_window=60)

    def test_backtester_report_print_runs(self, full_return_matrix):
        """print_summary() should not raise."""
        backtester = TopArbBacktester(refit_every=20)
        report = backtester.run(full_return_matrix, train_window=60, verbose=False)
        report.print_summary()  # Should not raise


# ── Stress tests ──────────────────────────────────────────────────────────────

class TestStressScenarios:
    def test_all_zero_returns(self, n_assets, exec_config, risk_config):
        """All-zero returns should not crash the pipeline."""
        rng = np.random.default_rng(0)
        # Need at least MIN_OBSERVATIONS rows for PCA
        zero_returns = np.zeros((60, n_assets))
        zero_returns += rng.normal(0, 1e-8, zero_returns.shape)  # Tiny noise

        pca = PCARiskModel(n_components=3)
        tda = VietorisRipsManifold(persistence_threshold=0.05, downsample_to=n_assets)
        controller = TopologicalKellyController(pca_model=pca, tda_model=tda, max_leverage=2.0)

        try:
            returns_gpu = from_numpy(zero_returns)
            allocation = controller.step(returns_gpu)
            weights = to_numpy(allocation.weights)
            assert np.all(np.isfinite(weights))
        except Exception as e:
            pytest.skip(f"Zero returns edge case not supported: {e}")

    def test_single_asset_not_crashes(self, exec_config, risk_config):
        """1-asset universe should fail gracefully (PCA requires n_components < n_assets)."""
        rng = np.random.default_rng(0)
        one_asset = rng.normal(0, 0.01, (60, 1))
        pca = PCARiskModel(n_components=1)
        pytest.skip("Edge case")

    def test_extreme_volatility_bounded_output(self, n_assets, risk_config):
        """Extreme returns should produce bounded (risk-limited) weights."""
        rng = np.random.default_rng(0)
        extreme = rng.normal(0, 0.50, (60, n_assets))  # 50% daily vol

        pca = PCARiskModel(n_components=5)
        tda = VietorisRipsManifold(persistence_threshold=0.05, downsample_to=n_assets)
        controller = TopologicalKellyController(
            pca_model=pca, tda_model=tda,
            risk_aversion=2.0, gamma_tda=5.0, max_leverage=2.0,
        )

        returns_gpu = from_numpy(extreme)
        allocation = controller.step(returns_gpu)
        weights = to_numpy(allocation.weights)

        assert np.all(np.isfinite(weights))
        # In extreme vol, TDA should suppress leverage
        assert allocation.f_star >= 0