from __future__ import annotations

import os as _os, numpy as _np, types as _types, sys as _sys
if _os.getenv("TOPARB_FORCE_CPU", "0").lower() in ("1", "true", "yes"):
    if "cupy" not in _sys.modules:
        _cupy_shim = _types.ModuleType("cupy")
        for _k, _v in _np.__dict__.items():
            if not _k.startswith("__"):
                setattr(_cupy_shim, _k, _v)
        _cupy_shim.asnumpy = lambda arr: _np.asarray(arr)
        _cupy_shim.ndarray = _np.ndarray
        _cupy_shim.get_default_memory_pool = lambda: type("P", (), {
            "used_bytes": lambda s: 0, "total_bytes": lambda s: 0,
            "free_all_blocks": lambda s: None})()
        _sys.modules["cupy"] = _cupy_shim

from dataclasses import dataclass, field
from typing import Final, Literal

import cupy as cp
import numpy as np

from src.math.eigen_risk import EigenPortfolioWeights, PCARiskModel
from src.math.manifold_tda import TopologicalFractureSignal, VietorisRipsManifold

_EPS: Final[float] = 1e-10
_MAX_LEVERAGE: Final[float] = 10.0
_MIN_LEVERAGE: Final[float] = 0.0
_HJB_DT: Final[float] = 1.0 / 252.0


@dataclass(frozen=True, slots=True)
class HJBSolution:
    f_star: float
    mu_net: float
    sigma_sq_effective: float
    tda_penalty: float
    raw_kelly: float
    clamped: bool
    leverage_regime: str


@dataclass(frozen=True, slots=True)
class PortfolioAllocation:
    weights: cp.ndarray
    f_star: float
    hjb: HJBSolution
    tda_signal: TopologicalFractureSignal
    eigen_weights: EigenPortfolioWeights
    transaction_cost_estimate: float
    expected_log_growth: float
    effective_sharpe: float


@dataclass(slots=True)
class ControllerState:
    f_history: list[float] = field(default_factory=list)
    sigma_sq_history: list[float] = field(default_factory=list)
    tda_penalty_history: list[float] = field(default_factory=list)
    portfolio_value: float = 1.0
    current_weights: cp.ndarray | None = None
    step: int = 0

    def update_value(self, f_star: float, dt: float = _HJB_DT) -> None:
        self.f_history.append(f_star)
        self.step += 1

    def trim(self, max_len: int = 2048) -> None:
        if len(self.f_history) > max_len:
            self.f_history = self.f_history[-max_len:]
            self.sigma_sq_history = self.sigma_sq_history[-max_len:]
            self.tda_penalty_history = self.tda_penalty_history[-max_len:]


def _compute_portfolio_moments(
        weights: cp.ndarray,
        returns: cp.ndarray,
        cov_matrix: cp.ndarray,
) -> tuple[float, float]:
    """
    Compute expected return and variance of a portfolio.

    μ_p = wᵀ μ̄   where μ̄ = (1/T) Σ rₜ  (sample mean of returns)
    σ²_p = wᵀ Σ w  (portfolio variance from covariance matrix)
    """
    T = returns.shape[0]
    mu_assets = returns.mean(axis=0)          # (N,) sample mean vector
    mu_port = float(weights @ mu_assets)      # scalar expected return
    sigma_sq_port = float(weights @ cov_matrix @ weights)
    return mu_port, sigma_sq_port


def _topological_variance_penalty(
    sigma_sq_base: float,
    fracture_signal: TopologicalFractureSignal,
    gamma_tda: float,
    kappa_anomaly: float,
) -> float:
    base_penalty = gamma_tda * fracture_signal.normalized_fracture * sigma_sq_base
    anomaly_multiplier = (
        kappa_anomaly * abs(fracture_signal.anomaly_zscore)
        if fracture_signal.is_anomaly else 0.0
    )
    loop_penalty = gamma_tda * 0.5 * abs(fracture_signal.delta_beta1) * sigma_sq_base
    return float(base_penalty + anomaly_multiplier * sigma_sq_base + loop_penalty)


def _solve_hjb_continuous_kelly(
    mu: float,
    sigma_sq: float,
    risk_aversion: float,
    transaction_cost: float,
    max_f: float,
) -> tuple[float, bool]:
    """
    Continuous-time Kelly / HJB solution for log-utility investor.

    HJB equation for V(w, t) = E[log W_T]:
        0 = max_f { μ·f - ½·γ·σ²·f² - c·|f| }
    First-order condition:  f* = (μ - c) / (γ · σ²)
    """
    if sigma_sq < _EPS:
        return 0.0, True

    mu_adj = mu - transaction_cost
    f_raw = mu_adj / (risk_aversion * sigma_sq)

    clamped = not (_MIN_LEVERAGE <= f_raw <= max_f)
    f_clamped = float(np.clip(f_raw, _MIN_LEVERAGE, max_f))
    return f_clamped, clamped


def _smooth_leverage_transition(
    f_current: float,
    f_target: float,
    speed: float,
    dt: float,
) -> float:
    delta = f_target - f_current
    return f_current + speed * delta * dt


def _expected_log_growth(
    f: float,
    mu: float,
    sigma_sq: float,
    risk_aversion: float,
) -> float:
    return f * mu - 0.5 * risk_aversion * (f ** 2) * sigma_sq


def _effective_sharpe(mu: float, sigma_sq: float) -> float:
    if sigma_sq < _EPS:
        return 0.0
    return mu / np.sqrt(sigma_sq)


def _market_neutral_projection(weights: cp.ndarray) -> cp.ndarray:
    n = weights.shape[0]
    ones = cp.ones(n, dtype=weights.dtype)
    return weights - (cp.dot(ones, weights) / n) * ones


def _transaction_cost_estimate(
    prev_weights: cp.ndarray | None,
    new_weights: cp.ndarray,
    cost_bps: float,
) -> float:
    if prev_weights is None:
        return 0.0
    turnover = float(cp.sum(cp.abs(new_weights - prev_weights)))
    return cost_bps * 1e-4 * turnover


class TopologicalKellyController:
    def __init__(
        self,
        pca_model: PCARiskModel,
        tda_model: VietorisRipsManifold,
        risk_aversion: float = 2.0,
        gamma_tda: float = 3.0,
        kappa_anomaly: float = 1.5,
        max_leverage: float = 3.0,
        leverage_speed: float = 5.0,
        transaction_cost_bps: float = 2.0,
        regime_thresholds: tuple[float, float] = (0.3, 0.7),
        n_eigen_portfolios: int = 5,
    ) -> None:
        if risk_aversion <= 0:
            raise ValueError(f"risk_aversion must be > 0, got {risk_aversion}")
        if not (0.0 < max_leverage <= _MAX_LEVERAGE):
            raise ValueError(f"max_leverage must be in (0, {_MAX_LEVERAGE}], got {max_leverage}")
        if gamma_tda < 0:
            raise ValueError(f"gamma_tda must be >= 0, got {gamma_tda}")
        if not (0.0 < leverage_speed <= 100.0):
            raise ValueError(f"leverage_speed must be in (0, 100], got {leverage_speed}")

        self._pca = pca_model
        self._tda = tda_model
        self._risk_aversion = risk_aversion
        self._gamma_tda = gamma_tda
        self._kappa_anomaly = kappa_anomaly
        self._max_leverage = max_leverage
        self._leverage_speed = leverage_speed
        self._transaction_cost_bps = transaction_cost_bps
        self._low_threshold, self._high_threshold = regime_thresholds
        self._n_eigen = n_eigen_portfolios

        self._state: ControllerState = ControllerState()

    def _classify_regime(self, f_star: float) -> str:
        if f_star < self._max_leverage * self._low_threshold:
            return "RISK_OFF"
        elif f_star < self._max_leverage * self._high_threshold:
            return "TRANSITION"
        else:
            return "RISK_ON"

    def _compute_alpha_weights(self, eigen_result: EigenPortfolioWeights) -> cp.ndarray:
        """
        Mean-reversion alpha signal from PCA residuals.

        z_i = (1/τ) Σ_{t-τ}^t ε_{i,t}   (recent residual mean per asset)
        α_i = -z_i                         (negative → expect reversion)

        Projected onto zero-sum space (market-neutral) and L2-normalized.
        """
        residuals = eigen_result.residual_returns
        recent_res = residuals[-5:, :] if residuals.shape[0] >= 5 else residuals
        alpha = -cp.mean(recent_res, axis=0)
        alpha = _market_neutral_projection(alpha)
        norm = cp.linalg.norm(alpha)
        if float(norm) > _EPS:
            return alpha / norm
        return alpha

    def _blend_eigen_weights(
        self,
        eigen_result: EigenPortfolioWeights,
        f_star: float,
    ) -> cp.ndarray:
        """
        Final portfolio weights as variance-weighted blend of eigen-portfolios,
        scaled by the HJB leverage scalar f*.

        w = f* · Σᵢ (λᵢ / Σλ) · vᵢ / ‖vᵢ‖

        where λᵢ are explained variances (eigenvalues) and vᵢ are the
        eigen-portfolio weight vectors. Then projected market-neutral and
        L1-clipped to f*.
        """
        weights_2d = eigen_result.weights
        ev = self._pca.explained_variance

        if weights_2d.ndim == 2 and ev is not None:
            ev_safe = cp.where(ev > _EPS, ev, _EPS)
            ev_norm = ev_safe / ev_safe.sum()
            n_use = min(self._n_eigen, weights_2d.shape[0])
            blended = cp.zeros(weights_2d.shape[1], dtype=cp.float64)
            for i in range(n_use):
                blended += ev_norm[i] * weights_2d[i, :]
        else:
            blended = weights_2d[0, :] if weights_2d.ndim == 2 else weights_2d

        blended = _market_neutral_projection(blended)
        norm = cp.linalg.norm(blended)
        if float(norm) > _EPS:
            blended = blended / norm

        weights = blended * f_star
        l1 = float(cp.sum(cp.abs(weights)))
        if l1 > f_star + _EPS:
            weights = weights * (f_star / l1)
        return weights

    def _solve_hjb_with_tda(
        self,
        mu: float,
        sigma_sq_base: float,
        tda_signal: TopologicalFractureSignal,
    ) -> HJBSolution:
        tda_penalty = _topological_variance_penalty(
            sigma_sq_base, tda_signal, self._gamma_tda, self._kappa_anomaly
        )
        sigma_sq_effective = sigma_sq_base + tda_penalty
        tc_estimate = self._transaction_cost_bps * 1e-4

        f_star, clamped = _solve_hjb_continuous_kelly(
            mu=mu,
            sigma_sq=sigma_sq_effective,
            risk_aversion=self._risk_aversion,
            transaction_cost=tc_estimate,
            max_f=self._max_leverage,
        )

        raw_kelly, _ = _solve_hjb_continuous_kelly(
            mu=mu,
            sigma_sq=sigma_sq_base,
            risk_aversion=self._risk_aversion,
            transaction_cost=0.0,
            max_f=_MAX_LEVERAGE,
        )

        if self._state.f_history:
            f_prev = self._state.f_history[-1]
            f_star = _smooth_leverage_transition(f_prev, f_star, self._leverage_speed, _HJB_DT)
            f_star = float(np.clip(f_star, _MIN_LEVERAGE, self._max_leverage))

        return HJBSolution(
            f_star=f_star,
            mu_net=mu,
            sigma_sq_effective=sigma_sq_effective,
            tda_penalty=tda_penalty,
            raw_kelly=raw_kelly,
            clamped=clamped,
            leverage_regime=self._classify_regime(f_star),
        )

    def step(
        self,
        returns: cp.ndarray,
        asset_return_matrix: cp.ndarray | None = None,
    ) -> PortfolioAllocation:
        if not self._pca.is_fitted:
            self._pca.fit(returns)

        eigen_result = self._pca.compute_eigen_portfolios(returns)
        cov = self._pca.compute_covariance_gpu(returns)

        alpha_weights = self._compute_alpha_weights(eigen_result)
        mu, sigma_sq_base = _compute_portfolio_moments(alpha_weights, returns, cov)

        tda_input = asset_return_matrix if asset_return_matrix is not None else returns.T
        tda_signal = self._tda.compute_fracture_signal(tda_input)

        hjb = self._solve_hjb_with_tda(mu, sigma_sq_base, tda_signal)

        final_weights = self._blend_eigen_weights(eigen_result, hjb.f_star)

        tc_est = _transaction_cost_estimate(
            self._state.current_weights,
            final_weights,
            self._transaction_cost_bps,
        )

        elg = _expected_log_growth(hjb.f_star, mu, hjb.sigma_sq_effective, self._risk_aversion)
        eff_sharpe = _effective_sharpe(mu, hjb.sigma_sq_effective)

        self._state.update_value(hjb.f_star)
        self._state.sigma_sq_history.append(hjb.sigma_sq_effective)
        self._state.tda_penalty_history.append(hjb.tda_penalty)
        self._state.current_weights = final_weights
        self._state.trim()

        return PortfolioAllocation(
            weights=final_weights,
            f_star=hjb.f_star,
            hjb=hjb,
            tda_signal=tda_signal,
            eigen_weights=eigen_result,
            transaction_cost_estimate=tc_est,
            expected_log_growth=elg,
            effective_sharpe=eff_sharpe,
        )

    def compute_optimal_rebalance_threshold(self, returns: cp.ndarray) -> float:
        if not self._pca.is_fitted:
            raise RuntimeError("PCARiskModel must be fitted before computing rebalance threshold.")

        cov = self._pca.compute_covariance_gpu(returns)
        n = returns.shape[1]
        avg_var = float(cp.trace(cov)) / max(n, 1)
        tc = self._transaction_cost_bps * 1e-4
        threshold = tc / (self._risk_aversion * avg_var + _EPS)
        return float(np.clip(threshold, 1e-4, 0.1))

    def compute_var_contribution(
        self,
        weights: cp.ndarray,
        returns: cp.ndarray,
        confidence: float = 0.99,
    ) -> dict[str, float]:
        portfolio_returns = returns @ weights
        sorted_r = cp.sort(portfolio_returns)
        idx = int(np.floor((1.0 - confidence) * len(sorted_r)))
        var = float(-sorted_r[idx])
        cvar = float(-cp.mean(sorted_r[:idx]))

        return {
            "VaR_99": var,
            "CVaR_99": cvar,
            "max_drawdown_proxy": float(cp.min(portfolio_returns)),
        }

    def compute_effective_n(self, weights: cp.ndarray) -> float:
        w_abs = cp.abs(weights)
        w_sum = cp.sum(w_abs)
        if float(w_sum) < _EPS:
            return 0.0
        w_norm = w_abs / w_sum
        hhi = float(cp.sum(w_norm ** 2))
        return 1.0 / max(hhi, _EPS)

    def serialize_state(self) -> dict[str, list[float]]:
        return {
            "f_history": list(self._state.f_history),
            "sigma_sq_history": list(self._state.sigma_sq_history),
            "tda_penalty_history": list(self._state.tda_penalty_history),
            "portfolio_value": [self._state.portfolio_value],
            "step": [float(self._state.step)],
        }

    def reset(self) -> None:
        self._state = ControllerState()
        self._tda.reset_state()

    @property
    def state(self) -> ControllerState:
        return self._state

    @property
    def current_f_star(self) -> float:
        if not self._state.f_history:
            return 0.0
        return self._state.f_history[-1]

    @property
    def leverage_regime(self) -> str:
        return self._classify_regime(self.current_f_star)
