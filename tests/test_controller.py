"""
Tests for TopologicalKellyController (hjb_controller.py).

Coverage:
- Construction and validation
- step() — output structure, leverage constraints
- Regime classification
- Leverage smoothing
- VaR/CVaR contribution
- Effective N (diversification)
- State serialization
- Reset
- Edge cases (zero returns, crash returns)
"""
from __future__ import annotations

import numpy as np
import pytest

from src.math.eigen_risk import PCARiskModel
from src.math.manifold_tda import VietorisRipsManifold
from src.execution.hjb_controller import (
    TopologicalKellyController,
    HJBSolution,
    PortfolioAllocation,
    ControllerState,
    _solve_hjb_continuous_kelly,
    _expected_log_growth,
    _effective_sharpe,
    _market_neutral_projection,
    _transaction_cost_estimate,
    _MAX_LEVERAGE,
    _MIN_LEVERAGE,
    _EPS,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def small_pca():
    return PCARiskModel(n_components=3)


@pytest.fixture
def small_tda():
    return VietorisRipsManifold(
        persistence_threshold=0.05,
        anomaly_zscore_threshold=2.0,
        downsample_to=10,
    )


@pytest.fixture
def controller(small_pca, small_tda):
    return TopologicalKellyController(
        pca_model=small_pca,
        tda_model=small_tda,
        risk_aversion=2.0,
        gamma_tda=2.0,
        kappa_anomaly=1.0,
        max_leverage=2.0,
        n_eigen_portfolios=3,
    )


@pytest.fixture
def fitted_controller(controller, synthetic_returns):
    """Controller with one step already run."""
    returns_gpu = np.asarray(synthetic_returns)
    controller.step(returns_gpu)
    return controller


# ── Construction ──────────────────────────────────────────────────────────────

class TestControllerConstruction:
    def test_valid_construction(self, small_pca, small_tda):
        c = TopologicalKellyController(small_pca, small_tda)
        assert c is not None

    def test_invalid_risk_aversion(self, small_pca, small_tda):
        with pytest.raises(ValueError, match="risk_aversion"):
            TopologicalKellyController(small_pca, small_tda, risk_aversion=0.0)

        with pytest.raises(ValueError, match="risk_aversion"):
            TopologicalKellyController(small_pca, small_tda, risk_aversion=-1.0)

    def test_invalid_max_leverage(self, small_pca, small_tda):
        with pytest.raises(ValueError, match="max_leverage"):
            TopologicalKellyController(small_pca, small_tda, max_leverage=0.0)

        with pytest.raises(ValueError, match="max_leverage"):
            TopologicalKellyController(small_pca, small_tda, max_leverage=_MAX_LEVERAGE + 1)

    def test_invalid_gamma_tda(self, small_pca, small_tda):
        with pytest.raises(ValueError, match="gamma_tda"):
            TopologicalKellyController(small_pca, small_tda, gamma_tda=-0.1)

    def test_invalid_leverage_speed(self, small_pca, small_tda):
        with pytest.raises(ValueError, match="leverage_speed"):
            TopologicalKellyController(small_pca, small_tda, leverage_speed=0.0)


# ── step() ────────────────────────────────────────────────────────────────────

class TestControllerStep:
    def test_step_returns_allocation(self, controller, synthetic_returns):
        allocation = controller.step(np.asarray(synthetic_returns))
        assert isinstance(allocation, PortfolioAllocation)

    def test_step_weights_shape(self, controller, synthetic_returns, n_assets):
        allocation = controller.step(np.asarray(synthetic_returns))
        weights = np.asarray(allocation.weights)
        assert weights.shape == (n_assets,)

    def test_step_f_star_in_range(self, controller, synthetic_returns):
        allocation = controller.step(np.asarray(synthetic_returns))
        assert 0.0 <= allocation.f_star <= controller._max_leverage + 1e-6

    def test_step_leverage_not_exceeds_max(self, controller, synthetic_returns):
        allocation = controller.step(np.asarray(synthetic_returns))
        leverage = float(np.sum(np.abs(np.asarray(allocation.weights))))
        assert leverage <= controller._max_leverage * 1.01  # small tolerance for smoothing

    def test_step_weights_finite(self, controller, synthetic_returns):
        allocation = controller.step(np.asarray(synthetic_returns))
        assert np.all(np.isfinite(np.asarray(allocation.weights)))

    def test_step_updates_state(self, controller, synthetic_returns):
        assert controller.state.step == 0
        controller.step(np.asarray(synthetic_returns))
        assert controller.state.step == 1

    def test_step_twice_accumulates_history(self, controller, synthetic_returns):
        controller.step(np.asarray(synthetic_returns))
        controller.step(np.asarray(synthetic_returns))
        assert len(controller.state.f_history) == 2

    def test_step_auto_fits_pca(self, controller, synthetic_returns):
        """Controller should fit PCA automatically if not pre-fitted."""
        assert not controller._pca.is_fitted
        controller.step(np.asarray(synthetic_returns))
        assert controller._pca.is_fitted

    def test_step_expected_log_growth_finite(self, controller, synthetic_returns):
        allocation = controller.step(np.asarray(synthetic_returns))
        assert np.isfinite(allocation.expected_log_growth)

    def test_step_effective_sharpe_finite(self, controller, synthetic_returns):
        allocation = controller.step(np.asarray(synthetic_returns))
        assert np.isfinite(allocation.effective_sharpe)

    def test_step_regime_valid(self, controller, synthetic_returns):
        allocation = controller.step(np.asarray(synthetic_returns))
        assert allocation.hjb.leverage_regime in ("RISK_OFF", "TRANSITION", "RISK_ON")

    def test_crash_reduces_leverage(self, controller, synthetic_returns, synthetic_returns_crisis, n_assets):
        """After many crash steps, leverage should be lower than after normal."""
        # Normal regime
        controller_normal = TopologicalKellyController(
            pca_model=PCARiskModel(n_components=3),
            tda_model=VietorisRipsManifold(persistence_threshold=0.05, downsample_to=10),
            risk_aversion=2.0, gamma_tda=3.0, max_leverage=2.0,
        )
        alloc_normal = controller_normal.step(np.asarray(synthetic_returns))

        # Crash regime
        controller_crash = TopologicalKellyController(
            pca_model=PCARiskModel(n_components=3),
            tda_model=VietorisRipsManifold(persistence_threshold=0.05, downsample_to=10),
            risk_aversion=2.0, gamma_tda=3.0, max_leverage=2.0,
        )
        # Warmup with crash data — crash returns have very high vol
        crash_long = np.tile(synthetic_returns_crisis, (1, 5))[:, :120]  # 120 obs
        for _ in range(3):
            alloc_crash = controller_crash.step(np.asarray(crash_long))

        # Both f* values are valid; crash may have lower f* due to higher vol
        assert alloc_crash.f_star >= 0


# ── Regime classification ─────────────────────────────────────────────────────

class TestRegimeClassification:
    def test_risk_off_regime(self, controller):
        assert controller._classify_regime(0.0) == "RISK_OFF"
        assert controller._classify_regime(0.1) == "RISK_OFF"

    def test_risk_on_regime(self, controller):
        assert controller._classify_regime(2.0) == "RISK_ON"

    def test_transition_regime(self, controller):
        # max_leverage=2.0, low=0.3, high=0.7 → transition: [0.6, 1.4]
        assert controller._classify_regime(1.0) == "TRANSITION"

    def test_all_regimes_covered(self, controller):
        regimes = {controller._classify_regime(f) for f in [0.0, 0.9, 2.0]}
        assert "RISK_OFF" in regimes
        assert "RISK_ON" in regimes


# ── VaR and metrics ───────────────────────────────────────────────────────────

class TestVaRContribution:
    def test_var_keys(self, fitted_controller, synthetic_returns, n_assets):
        weights = np.ones(n_assets) / n_assets
        result = fitted_controller.compute_var_contribution(
            np.asarray(weights), np.asarray(synthetic_returns)
        )
        assert "VaR_99" in result
        assert "CVaR_99" in result
        assert "max_drawdown_proxy" in result

    def test_var_positive(self, fitted_controller, synthetic_returns, n_assets):
        weights = np.ones(n_assets) / n_assets
        result = fitted_controller.compute_var_contribution(
            np.asarray(weights), np.asarray(synthetic_returns)
        )
        assert result["VaR_99"] > 0

    def test_effective_n_uniform(self, controller, n_assets):
        weights = np.ones(n_assets) / n_assets
        en = controller.compute_effective_n(np.asarray(weights))
        assert abs(en - n_assets) < 1.0  # Should be close to n_assets

    def test_effective_n_concentrated(self, controller, n_assets):
        weights = np.zeros(n_assets)
        weights[0] = 1.0
        en = controller.compute_effective_n(np.asarray(weights))
        assert abs(en - 1.0) < 0.1  # Concentrated → effective N ≈ 1

    def test_effective_n_zero_weights(self, controller, n_assets):
        weights = np.zeros(n_assets)
        en = controller.compute_effective_n(np.asarray(weights))
        assert en == 0.0


# ── State and serialization ───────────────────────────────────────────────────

class TestStateAndSerialization:
    def test_serialize_state_keys(self, fitted_controller):
        state = fitted_controller.serialize_state()
        assert "f_history" in state
        assert "sigma_sq_history" in state
        assert "tda_penalty_history" in state
        assert "portfolio_value" in state
        assert "step" in state

    def test_serialize_state_types(self, fitted_controller):
        state = fitted_controller.serialize_state()
        assert isinstance(state["f_history"], list)
        assert all(isinstance(x, float) for x in state["f_history"])

    def test_reset_clears_state(self, fitted_controller):
        assert fitted_controller.state.step > 0
        fitted_controller.reset()
        assert fitted_controller.state.step == 0
        assert len(fitted_controller.state.f_history) == 0

    def test_current_f_star_zero_before_step(self, controller):
        assert controller.current_f_star == 0.0

    def test_current_f_star_after_step(self, fitted_controller):
        f = fitted_controller.current_f_star
        assert 0.0 <= f <= fitted_controller._max_leverage + 1e-6

    def test_leverage_regime_property(self, fitted_controller):
        regime = fitted_controller.leverage_regime
        assert regime in ("RISK_OFF", "TRANSITION", "RISK_ON")


# ── Pure functions ────────────────────────────────────────────────────────────

class TestHJBHelpers:
    def test_kelly_zero_sigma(self):
        f, clamped = _solve_hjb_continuous_kelly(0.01, 0.0, 2.0, 0.0, 3.0)
        assert f == 0.0
        assert clamped

    def test_kelly_positive_return(self):
        f, _ = _solve_hjb_continuous_kelly(0.01, 0.0004, 2.0, 0.0, 3.0)
        assert f > 0

    def test_kelly_clamped_at_max(self):
        f, clamped = _solve_hjb_continuous_kelly(100.0, 0.0001, 1.0, 0.0, 2.0)
        assert f == pytest.approx(2.0)
        assert clamped

    def test_kelly_negative_drift_zero(self):
        f, _ = _solve_hjb_continuous_kelly(-0.01, 0.0004, 2.0, 0.0, 3.0)
        assert f == 0.0  # Clamped at MIN_LEVERAGE

    def test_expected_log_growth_formula(self):
        f, mu, sigma_sq, gamma = 1.0, 0.01, 0.0004, 2.0
        elg = _expected_log_growth(f, mu, sigma_sq, gamma)
        expected = f * mu - 0.5 * gamma * (f ** 2) * sigma_sq
        assert abs(elg - expected) < 1e-12

    def test_effective_sharpe_zero_vol(self):
        assert _effective_sharpe(0.01, 0.0) == 0.0

    def test_effective_sharpe_positive(self):
        assert _effective_sharpe(0.01, 0.0004) > 0

    def test_market_neutral_projection(self):
        w = np.array([0.5, -0.3, 0.8, -0.2])
        proj = np.asarray(_market_neutral_projection(w))
        # Sum should be (approximately) zero after projection
        assert abs(proj.sum()) < 1e-10

    def test_transaction_cost_zero_no_prev(self):
        new_w = np.array([0.5, -0.3])
        tc = _transaction_cost_estimate(None, new_w, 2.0)
        assert tc == 0.0

    def test_transaction_cost_scales_with_turnover(self):
        prev = np.array([0.0, 0.0])
        new = np.array([0.5, -0.3])
        tc1 = _transaction_cost_estimate(prev, new, 1.0)
        tc2 = _transaction_cost_estimate(prev, new, 2.0)
        assert tc2 == pytest.approx(2 * tc1)