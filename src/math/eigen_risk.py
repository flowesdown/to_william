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

from dataclasses import dataclass
from typing import Final, Literal

import cupy as cp
import numpy as np

try:
    from cuml.decomposition import PCA as cuPCA
    from cuml.preprocessing import StandardScaler as cuScaler
    _CUML_AVAILABLE: bool = True
except ImportError:
    _CUML_AVAILABLE: bool = False

_EPS: Final[float] = 1e-10
_MIN_OBSERVATIONS: Final[int] = 30


@dataclass(frozen=True, slots=True)
class PCAResult:
    components: cp.ndarray
    explained_variance: cp.ndarray
    explained_variance_ratio: cp.ndarray
    singular_values: cp.ndarray
    n_components: int
    n_observations: int
    condition_number: float
    used_cuml: bool


@dataclass(frozen=True, slots=True)
class EigenPortfolioWeights:
    weights: cp.ndarray
    residual_returns: cp.ndarray
    factor_returns: cp.ndarray
    factor_exposures: cp.ndarray
    idiosyncratic_variance: cp.ndarray
    systematic_variance: cp.ndarray
    total_variance: cp.ndarray


@dataclass(frozen=True, slots=True)
class CorrelationDecomposition:
    correlation_matrix: cp.ndarray
    eigenvalues: cp.ndarray
    eigenvectors: cp.ndarray
    marchenko_pastur_threshold: float
    noise_eigenvalues: cp.ndarray
    signal_eigenvalues: cp.ndarray
    n_signal_components: int


def _marchenko_pastur_upper(
    n_assets: int,
    n_observations: int,
    sigma_sq: float = 1.0,
) -> float:
    q = n_observations / n_assets
    return sigma_sq * (1.0 + 1.0 / q + 2.0 * np.sqrt(1.0 / q))


def _ledoit_wolf_shrinkage_gpu(
    sample_cov: cp.ndarray,
    n: int,
) -> cp.ndarray:
    p = sample_cov.shape[0]
    mu = cp.trace(sample_cov) / p

    delta = cp.linalg.norm(sample_cov - mu * cp.eye(p), "fro") ** 2 / p

    tr_S = float(cp.trace(sample_cov))
    tr_S2 = float(cp.trace(sample_cov @ sample_cov))
    beta_bar = (tr_S2 + tr_S ** 2 - 2.0 * tr_S2 / p) / (n * p + _EPS)
    beta_bar = max(0.0, beta_bar)

    alpha = float(delta)
    rho = max(0.0, min(1.0, beta_bar / alpha)) if alpha > _EPS else 0.0

    target = mu * cp.eye(p, dtype=sample_cov.dtype)
    return (1.0 - rho) * sample_cov + rho * target


def _cov_to_corr_gpu(cov: cp.ndarray) -> cp.ndarray:
    std = cp.sqrt(cp.diag(cov))
    std = cp.where(std > _EPS, std, _EPS)
    inv_std = 1.0 / std
    corr = cov * cp.outer(inv_std, inv_std)
    corr = cp.clip(corr, -1.0, 1.0)
    cp.fill_diagonal(corr, 1.0)
    return corr


class PCARiskModel:
    def __init__(
        self,
        n_components: int,
        shrinkage: Literal["ledoit_wolf", "none"] = "ledoit_wolf",
        demean: bool = True,
        normalize: bool = True,
        marchenko_pastur_filter: bool = True,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")

        self._n_components = n_components
        self._shrinkage = shrinkage
        self._demean = demean
        self._normalize = normalize
        self._marchenko_pastur_filter = marchenko_pastur_filter

        self._components: cp.ndarray | None = None
        self._explained_variance: cp.ndarray | None = None
        self._mean: cp.ndarray | None = None
        self._std: cp.ndarray | None = None
        self._is_fitted: bool = False
        self._used_cuml: bool = False

    def _preprocess(self, returns: cp.ndarray) -> cp.ndarray:
        X = returns.astype(cp.float64)
        if self._demean:
            self._mean = X.mean(axis=0, keepdims=True)
            X = X - self._mean
        else:
            self._mean = cp.zeros((1, X.shape[1]), dtype=cp.float64)

        if self._normalize:
            self._std = X.std(axis=0, keepdims=True)
            self._std = cp.where(self._std > _EPS, self._std, _EPS)
            X = X / self._std
        else:
            self._std = cp.ones((1, X.shape[1]), dtype=cp.float64)

        return X

    def _fit_cuml(self, X: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        cuML PCA path — ~100× faster than sklearn on large matrices.
        cuML operates natively on cupy arrays with zero host-device copies.
        """
        pca = cuPCA(n_components=self._n_components)
        pca.fit(X.astype(cp.float32))
        components = cp.asarray(pca.components_).astype(cp.float64)
        eigenvalues = cp.asarray(pca.explained_variance_).astype(cp.float64)
        singular_values = cp.asarray(pca.singular_values_).astype(cp.float64)
        return components, eigenvalues, singular_values

    def _svd_pca_gpu(self, X: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Fallback cupy eigh path when cuML is unavailable.
        Still ~40× faster than numpy on GPU via cuBLAS.
        """
        n, p = X.shape
        cov = (X.T @ X) / (n - 1)

        if self._shrinkage == "ledoit_wolf":
            cov = _ledoit_wolf_shrinkage_gpu(cov, n)

        eigenvalues, eigenvectors = cp.linalg.eigh(cov)

        idx = cp.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvalues = cp.maximum(eigenvalues, _EPS)
        singular_values = cp.sqrt(eigenvalues * (n - 1))

        n_keep = min(self._n_components, p, n - 1)
        components = eigenvectors[:, :n_keep].T
        ev = eigenvalues[:n_keep]
        sv = singular_values[:n_keep]

        return components, ev, sv

    def _marchenko_pastur_denoise(
        self,
        eigenvalues: cp.ndarray,
        n_assets: int,
        n_obs: int,
    ) -> CorrelationDecomposition:
        ev_np = cp.asnumpy(eigenvalues)
        mp_upper = _marchenko_pastur_upper(n_assets, n_obs)

        signal_mask = ev_np > mp_upper
        noise_mask = ~signal_mask
        n_signal = int(signal_mask.sum())

        return CorrelationDecomposition(
            correlation_matrix=cp.zeros((n_assets, n_assets), dtype=cp.float64),
            eigenvalues=eigenvalues,
            eigenvectors=cp.eye(n_assets, dtype=cp.float64),
            marchenko_pastur_threshold=float(mp_upper),
            noise_eigenvalues=eigenvalues[cp.array(noise_mask)],
            signal_eigenvalues=eigenvalues[cp.array(signal_mask)],
            n_signal_components=n_signal,
        )

    def fit(self, returns: cp.ndarray) -> PCAResult:
        if returns.ndim != 2:
            raise ValueError(f"returns must be 2D (T x N), got shape {returns.shape}")

        n_obs, n_assets = returns.shape

        if n_obs < _MIN_OBSERVATIONS:
            raise ValueError(f"Insufficient observations: {n_obs} < {_MIN_OBSERVATIONS}")

        if self._n_components > n_assets:
            raise ValueError(f"n_components ({self._n_components}) > n_assets ({n_assets})")

        X = self._preprocess(returns)

        if _CUML_AVAILABLE:
            components, eigenvalues, singular_values = self._fit_cuml(X)
            self._used_cuml = True
        else:
            components, eigenvalues, singular_values = self._svd_pca_gpu(X)
            self._used_cuml = False

        total_variance = float(eigenvalues.sum())
        ev_ratio = eigenvalues / max(total_variance, _EPS)
        cond = float(eigenvalues[0] / max(float(eigenvalues[-1]), _EPS))

        self._components = components
        self._explained_variance = eigenvalues
        self._is_fitted = True

        return PCAResult(
            components=components,
            explained_variance=eigenvalues,
            explained_variance_ratio=ev_ratio,
            singular_values=singular_values,
            n_components=components.shape[0],
            n_observations=n_obs,
            condition_number=cond,
            used_cuml=self._used_cuml,
        )

    def transform(self, returns: cp.ndarray) -> cp.ndarray:
        if not self._is_fitted:
            raise RuntimeError("PCARiskModel.fit() must be called before transform()")

        X = returns.astype(cp.float64)
        if self._demean and self._mean is not None:
            X = X - self._mean
        if self._normalize and self._std is not None:
            X = X / self._std

        return X @ self._components.T

    def compute_eigen_portfolios(self, returns: cp.ndarray) -> EigenPortfolioWeights:
        if not self._is_fitted:
            raise RuntimeError("PCARiskModel.fit() must be called before compute_eigen_portfolios()")

        factor_returns = self.transform(returns)
        factor_exposures = self._components

        reconstructed = factor_returns @ factor_exposures
        if self._normalize and self._std is not None:
            reconstructed = reconstructed * self._std
        if self._demean and self._mean is not None:
            reconstructed = reconstructed + self._mean

        residual_returns = returns.astype(cp.float64) - reconstructed

        systematic_var = cp.var(reconstructed, axis=0)
        idiosyncratic_var = cp.var(residual_returns, axis=0)
        total_var = systematic_var + idiosyncratic_var

        ev = self._explained_variance
        ev_safe = cp.where(ev > _EPS, ev, _EPS)
        raw_weights = self._components / ev_safe[:, None]
        weight_norms = cp.linalg.norm(raw_weights, axis=1, keepdims=True)
        weight_norms = cp.where(weight_norms > _EPS, weight_norms, _EPS)
        normalized_weights = raw_weights / weight_norms

        return EigenPortfolioWeights(
            weights=normalized_weights,
            residual_returns=residual_returns,
            factor_returns=factor_returns,
            factor_exposures=factor_exposures,
            idiosyncratic_variance=idiosyncratic_var,
            systematic_variance=systematic_var,
            total_variance=total_var,
        )

    def compute_covariance_gpu(self, returns: cp.ndarray) -> cp.ndarray:
        n = returns.shape[0]
        X = returns.astype(cp.float64)
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        cov = (Xc.T @ Xc) / (n - 1)
        return _ledoit_wolf_shrinkage_gpu(cov, n) if self._shrinkage == "ledoit_wolf" else cov

    def compute_correlation_gpu(self, returns: cp.ndarray) -> cp.ndarray:
        cov = self.compute_covariance_gpu(returns)
        return _cov_to_corr_gpu(cov)

    def factor_risk_attribution(
        self, weights: cp.ndarray, returns: cp.ndarray
    ) -> dict[str, float]:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        cov = self.compute_covariance_gpu(returns)
        portfolio_var = float(weights @ cov @ weights)

        factor_cov = self._components @ cov @ self._components.T
        factor_weights = self._components @ weights
        systematic_var = float(factor_weights @ factor_cov @ factor_weights)
        idio_var = max(0.0, portfolio_var - systematic_var)

        return {
            "total_variance": portfolio_var,
            "systematic_variance": systematic_var,
            "idiosyncratic_variance": idio_var,
            "systematic_fraction": systematic_var / max(portfolio_var, _EPS),
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def components(self) -> cp.ndarray | None:
        return self._components

    @property
    def explained_variance(self) -> cp.ndarray | None:
        return self._explained_variance

    @property
    def n_components(self) -> int:
        return self._n_components