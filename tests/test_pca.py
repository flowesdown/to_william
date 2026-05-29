"""
Tests for PCARiskModel (eigen_risk.py).

Coverage:
- Construction and validation
- fit() on synthetic data
- transform() and inverse consistency
- compute_eigen_portfolios()
- compute_covariance_gpu() and compute_correlation_gpu()
- factor_risk_attribution()
- Marchenko-Pastur denoising
- Edge cases (insufficient data, n_components > n_assets)
"""
from __future__ import annotations

import numpy as np
import pytest

from src.math.eigen_risk import (
    PCARiskModel,
    PCAResult,
    EigenPortfolioWeights,
    CorrelationDecomposition,
    _marchenko_pastur_upper,
    _EPS,
    _MIN_OBSERVATIONS,
)


class TestPCARiskModelConstruction:
    def test_valid_construction(self):
        model = PCARiskModel(n_components=5)
        assert model.n_components == 5
        assert not model.is_fitted
        assert model.components is None
        assert model.explained_variance is None

    def test_invalid_n_components(self):
        with pytest.raises(ValueError, match="n_components"):
            PCARiskModel(n_components=0)

        with pytest.raises(ValueError, match="n_components"):
            PCARiskModel(n_components=-1)

    def test_shrinkage_modes(self):
        m1 = PCARiskModel(n_components=3, shrinkage="ledoit_wolf")
        m2 = PCARiskModel(n_components=3, shrinkage="none")
        assert m1.n_components == m2.n_components


class TestPCARiskModelFit:
    def test_fit_returns_pca_result(self, synthetic_returns, n_assets):
        model = PCARiskModel(n_components=5)
        returns_gpu = np.asarray(synthetic_returns)
        result = model.fit(returns_gpu)

        assert isinstance(result, PCAResult)
        assert result.n_components == 5
        assert result.n_observations == synthetic_returns.shape[0]
        assert model.is_fitted

    def test_fit_components_shape(self, synthetic_returns, n_assets):
        model = PCARiskModel(n_components=5)
        result = model.fit(synthetic_returns)
        assert result.components.shape == (5, n_assets)

    def test_explained_variance_ratio_sums_to_leq_one(self, synthetic_returns):
        model = PCARiskModel(n_components=5)
        result = model.fit(synthetic_returns)
        ratio_sum = float(np.sum(result.explained_variance_ratio))
        assert 0.0 < ratio_sum <= 1.05

    def test_explained_variance_positive(self, synthetic_returns):
        model = PCARiskModel(n_components=5)
        result = model.fit(synthetic_returns)
        assert np.all(np.asarray(result.explained_variance) > 0)

    def test_condition_number_positive(self, synthetic_returns):
        model = PCARiskModel(n_components=5)
        result = model.fit(synthetic_returns)
        assert result.condition_number > 0

    def test_fit_requires_2d_input(self):
        model = PCARiskModel(n_components=3)
        with pytest.raises(ValueError, match="2D"):
            model.fit(np.ones(100))

    def test_fit_requires_minimum_observations(self, n_assets):
        model = PCARiskModel(n_components=3)
        tiny = np.random.randn(_MIN_OBSERVATIONS - 1, n_assets)
        with pytest.raises(ValueError, match="Insufficient"):
            model.fit(tiny)

    def test_fit_n_components_exceeds_n_assets(self, synthetic_returns, n_assets):
        model = PCARiskModel(n_components=n_assets + 1)
        with pytest.raises(ValueError, match="n_components"):
            model.fit(synthetic_returns)

    def test_fit_with_no_shrinkage(self, synthetic_returns):
        model = PCARiskModel(n_components=5, shrinkage="none")
        result = model.fit(synthetic_returns)
        assert model.is_fitted
        assert result.n_components == 5

    def test_fit_with_no_demean(self, synthetic_returns):
        model = PCARiskModel(n_components=5, demean=False)
        result = model.fit(synthetic_returns)
        assert model.is_fitted

    def test_refit_updates_components(self, synthetic_returns, synthetic_returns_crisis, n_assets):
        model = PCARiskModel(n_components=3)
        result1 = model.fit(synthetic_returns)
        comp1 = np.array(result1.components)

        # Fit again on different data
        result2 = model.fit(synthetic_returns_crisis)
        comp2 = np.array(result2.components)

        # Components should change
        assert comp1.shape == comp2.shape  # Same shape
        # They won't be identical
        assert not np.allclose(comp1, comp2, atol=1e-3)


class TestPCARiskModelTransform:
    def test_transform_requires_fitted_model(self, synthetic_returns):
        model = PCARiskModel(n_components=5)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            model.transform(synthetic_returns)

    def test_transform_shape(self, fitted_pca, synthetic_returns):
        result = fitted_pca.transform(synthetic_returns)
        n_obs = synthetic_returns.shape[0]
        assert result.shape == (n_obs, fitted_pca.n_components)

    def test_transform_finite_values(self, fitted_pca, synthetic_returns):
        result = fitted_pca.transform(synthetic_returns)
        assert np.all(np.isfinite(np.asarray(result)))


class TestEigenPortfolioWeights:
    def test_compute_eigen_portfolios_requires_fitted(self, pca_model, synthetic_returns):
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            pca_model.compute_eigen_portfolios(synthetic_returns)

    def test_compute_eigen_portfolios_structure(self, fitted_pca, synthetic_returns, n_assets):
        result = fitted_pca.compute_eigen_portfolios(synthetic_returns)
        assert isinstance(result, EigenPortfolioWeights)
        assert np.asarray(result.weights).shape[1] == n_assets

    def test_eigen_portfolio_variances_positive(self, fitted_pca, synthetic_returns):
        result = fitted_pca.compute_eigen_portfolios(synthetic_returns)
        assert np.all(np.asarray(result.idiosyncratic_variance) >= 0)
        assert np.all(np.asarray(result.systematic_variance) >= 0)
        assert np.all(np.asarray(result.total_variance) > 0)

    def test_residual_returns_shape(self, fitted_pca, synthetic_returns):
        result = fitted_pca.compute_eigen_portfolios(synthetic_returns)
        assert result.residual_returns.shape == synthetic_returns.shape

    def test_factor_returns_shape(self, fitted_pca, synthetic_returns):
        result = fitted_pca.compute_eigen_portfolios(synthetic_returns)
        n_obs = synthetic_returns.shape[0]
        assert result.factor_returns.shape == (n_obs, fitted_pca.n_components)


class TestCovarianceAndCorrelation:
    def test_covariance_shape(self, fitted_pca, synthetic_returns, n_assets):
        cov = fitted_pca.compute_covariance_gpu(synthetic_returns)
        assert cov.shape == (n_assets, n_assets)

    def test_covariance_symmetric(self, fitted_pca, synthetic_returns):
        cov = np.asarray(fitted_pca.compute_covariance_gpu(synthetic_returns))
        assert np.allclose(cov, cov.T, atol=1e-10)

    def test_covariance_positive_definite(self, fitted_pca, synthetic_returns):
        cov = np.asarray(fitted_pca.compute_covariance_gpu(synthetic_returns))
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > -1e-8)

    def test_correlation_diagonal_ones(self, fitted_pca, synthetic_returns):
        corr = np.asarray(fitted_pca.compute_correlation_gpu(synthetic_returns))
        diag = np.diag(corr)
        assert np.allclose(diag, 1.0, atol=1e-6)

    def test_correlation_bounded(self, fitted_pca, synthetic_returns):
        corr = np.asarray(fitted_pca.compute_correlation_gpu(synthetic_returns))
        assert np.all(corr >= -1.0 - 1e-8)
        assert np.all(corr <= 1.0 + 1e-8)

    def test_correlation_symmetric(self, fitted_pca, synthetic_returns):
        corr = np.asarray(fitted_pca.compute_correlation_gpu(synthetic_returns))
        assert np.allclose(corr, corr.T, atol=1e-10)


class TestFactorRiskAttribution:
    def test_attribution_requires_fitted(self, pca_model, synthetic_returns, n_assets):
        weights = np.ones(n_assets) / n_assets
        with pytest.raises(RuntimeError, match="not fitted"):
            pca_model.factor_risk_attribution(weights, synthetic_returns)

    def test_attribution_keys(self, fitted_pca, synthetic_returns, n_assets):
        weights = np.ones(n_assets) / n_assets
        result = fitted_pca.factor_risk_attribution(weights, synthetic_returns)
        assert "total_variance" in result
        assert "systematic_variance" in result
        assert "idiosyncratic_variance" in result
        assert "systematic_fraction" in result

    def test_attribution_fraction_in_range(self, fitted_pca, synthetic_returns, n_assets):
        weights = np.ones(n_assets) / n_assets
        result = fitted_pca.factor_risk_attribution(weights, synthetic_returns)
        assert 0.0 <= result["systematic_fraction"] <= 1.05

    def test_attribution_variance_positive(self, fitted_pca, synthetic_returns, n_assets):
        weights = np.ones(n_assets) / n_assets
        result = fitted_pca.factor_risk_attribution(weights, synthetic_returns)
        assert result["total_variance"] > 0

    def test_attribution_idio_non_negative(self, fitted_pca, synthetic_returns, n_assets):
        weights = np.ones(n_assets) / n_assets
        result = fitted_pca.factor_risk_attribution(weights, synthetic_returns)
        assert result["idiosyncratic_variance"] >= 0


class TestMarchenkoPastur:
    def test_upper_bound_positive(self):
        upper = _marchenko_pastur_upper(50, 252)
        assert upper > 0

    def test_upper_bound_increases_with_fewer_obs(self):
        # Fewer observations relative to assets → higher noise threshold
        upper_250 = _marchenko_pastur_upper(50, 250)
        upper_100 = _marchenko_pastur_upper(50, 100)
        assert upper_100 > upper_250

    def test_upper_bound_formula(self):
        n, t = 50, 252
        q = t / n
        expected = 1.0 * (1.0 + 1.0 / q + 2.0 * np.sqrt(1.0 / q))
        actual = _marchenko_pastur_upper(n, t)
        assert abs(actual - expected) < 1e-10