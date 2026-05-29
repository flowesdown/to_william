"""
Pytest configuration and shared fixtures.

Key setup:
- Patches cupy with a numpy wrapper so all tests run without a GPU
- Provides synthetic return matrices for model tests
- All fixtures are deterministic (fixed random seed)
"""
from __future__ import annotations

import os
import sys
import types
from typing import Generator

import numpy as np
import pytest

# Force CPU mode BEFORE any project imports
os.environ["TOPARB_FORCE_CPU"] = "1"
os.environ["TOPARB_PAPER_TRADING"] = "1"
os.environ["TOPARB_INFLUX_URL"] = ""   # Disable InfluxDB in tests


# ---------------------------------------------------------------------------
# cupy → numpy shim  (applied before project imports)
# ---------------------------------------------------------------------------

def _make_cupy_shim():
    """Create a numpy-compatible cupy shim for test environments without GPU."""
    cupy = types.ModuleType("cupy")

    # Core array operations — delegate to numpy
    _passthrough = [
        "array", "zeros", "ones", "eye", "arange", "linspace",
        "asarray", "ascontiguousarray", "empty",
        "zeros_like", "ones_like", "empty_like",
        "sum", "prod", "mean", "std", "var", "min", "max",
        "abs", "sqrt", "log", "exp", "clip", "where",
        "dot", "outer", "diag", "trace",
        "sort", "argsort", "unique",
        "concatenate", "stack", "hstack", "vstack",
        "diff", "cumsum", "cumprod",
        "maximum", "minimum",
        "isfinite", "isnan", "isinf",
        "nan_to_num",
        "fill_diagonal",
        "newaxis",
        "pi", "inf", "nan",
        "float32", "float64", "int32", "int64",
    ]

    for name in _passthrough:
        val = getattr(np, name, None)
        if val is not None:
            setattr(cupy, name, val)

    # linalg
    cupy.linalg = np.linalg

    # Special: asnumpy == identity for numpy arrays
    cupy.asnumpy = lambda arr: np.asarray(arr)

    # ndarray = np.ndarray
    cupy.ndarray = np.ndarray

    # cupy.get_default_memory_pool (no-op)
    class _FakePool:
        def used_bytes(self): return 0
        def total_bytes(self): return 0
        def free_all_blocks(self): pass

    cupy.get_default_memory_pool = _FakePool

    return cupy


# Install shim before any project module is imported
_cupy_shim = _make_cupy_shim()
sys.modules["cupy"] = _cupy_shim

# Also shim cudf and cuml (not used in tests but imported in some modules)
_cudf = types.ModuleType("cudf")
_cudf.from_pandas = lambda df, **kw: df
sys.modules["cudf"] = _cudf

_cuml = types.ModuleType("cuml")
_cuml_decomp = types.ModuleType("cuml.decomposition")
_cuml_pre = types.ModuleType("cuml.preprocessing")
sys.modules["cuml"] = _cuml
sys.modules["cuml.decomposition"] = _cuml_decomp
sys.modules["cuml.preprocessing"] = _cuml_pre

# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
SEED = 42


# ---------------------------------------------------------------------------
# Synthetic market data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def n_assets() -> int:
    return 20


@pytest.fixture(scope="session")
def n_obs() -> int:
    return 120


@pytest.fixture(scope="session")
def synthetic_returns(n_assets, n_obs) -> np.ndarray:
    """
    Synthetic log-returns: (n_obs × n_assets).
    Uses a 3-factor model to mimic realistic correlation structure.
    """
    rng = np.random.default_rng(SEED)
    n_factors = 3

    # Factor returns
    factor_rets = rng.normal(0, 0.01, (n_obs, n_factors))

    # Factor loadings
    loadings = rng.normal(0, 0.5, (n_assets, n_factors))
    loadings[:, 0] = np.abs(loadings[:, 0])  # Market factor (positive)

    # Idiosyncratic returns
    idio = rng.normal(0, 0.005, (n_obs, n_assets))

    # Combined
    returns = factor_rets @ loadings.T + idio

    # Clip to realistic range
    returns = np.clip(returns, -0.15, 0.15)
    return returns.astype(np.float64)


@pytest.fixture(scope="session")
def synthetic_returns_crisis(n_assets) -> np.ndarray:
    """
    Returns during a synthetic crash — high correlation, large negative.
    """
    rng = np.random.default_rng(SEED + 1)
    n_crisis = 40
    # All assets crash together
    market_shock = rng.normal(-0.03, 0.01, (n_crisis, 1))
    idio = rng.normal(0, 0.003, (n_crisis, n_assets))
    returns = np.broadcast_to(market_shock, (n_crisis, n_assets)).copy() + idio
    return returns.astype(np.float64)


@pytest.fixture(scope="session")
def tickers(n_assets) -> list[str]:
    return [f"ASSET_{i:03d}" for i in range(n_assets)]


@pytest.fixture
def pca_model(n_assets):
    """Fitted PCARiskModel on synthetic data."""
    from src.math.eigen_risk import PCARiskModel
    model = PCARiskModel(n_components=5)
    return model


@pytest.fixture
def fitted_pca(pca_model, synthetic_returns):
    """Pre-fitted PCA model."""
    pca_model.fit(synthetic_returns)
    return pca_model


@pytest.fixture
def tda_model():
    """VietorisRipsManifold for testing."""
    from src.math.manifold_tda import VietorisRipsManifold
    return VietorisRipsManifold(
        persistence_threshold=0.05,
        anomaly_zscore_threshold=2.0,
        downsample_to=20,
    )


@pytest.fixture
def risk_config():
    from config.settings import RiskConfig
    return RiskConfig()


@pytest.fixture
def risk_manager(risk_config, n_assets):
    from src.risk.risk_manager import RiskManager
    rm = RiskManager(config=risk_config, n_assets=n_assets)
    rm.update_nav(1_000_000.0)
    rm.reset_daily(1_000_000.0)
    return rm


@pytest.fixture
def exec_config():
    from config.settings import ExecutionConfig
    return ExecutionConfig()


@pytest.fixture
def paper_router(exec_config):
    from src.execution.order_router import PaperOrderRouter
    return PaperOrderRouter(exec_config)


@pytest.fixture
def rebalancer():
    from src.execution.rebalancer import ThresholdRebalancer
    return ThresholdRebalancer(base_threshold=0.02, min_rebalance_interval_steps=1)