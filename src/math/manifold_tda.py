from __future__ import annotations
# ── GPU/CPU backend shim ─────────────────────────────────────────────────────
import os as _os, numpy as _np, types as _types, sys as _sys
if _os.getenv("TOPARB_FORCE_CPU", "0").lower() in ("1", "true", "yes"):
    if "cupy" not in _sys.modules:
        _cupy_shim = _types.ModuleType("cupy")
        for _k, _v in _np.__dict__.items():
            if not _k.startswith("__"):
                setattr(_cupy_shim, _k, _v)
        _cupy_shim.asnumpy = lambda arr: _np.asarray(arr)
        _cupy_shim.ndarray = _np.ndarray
        _cupy_shim.get_default_memory_pool = lambda: type("P",(),{
            "used_bytes": lambda s: 0, "total_bytes": lambda s: 0,
            "free_all_blocks": lambda s: None})()
        _sys.modules["cupy"] = _cupy_shim
# ─────────────────────────────────────────────────────────────────────────────


from dataclasses import dataclass, field
from typing import Final, Literal, Sequence

import cupy as cp
import numpy as np
from scipy.spatial.distance import squareform
from ripser import ripser

_EPS: Final[float] = 1e-10
_MAX_SIMPLEX_DIM: Final[int] = 2
_BETTI_FRACTURE_WINDOW: Final[int] = 5


@dataclass(frozen=True, slots=True)
class PersistenceDiagram:
    h0_bars: np.ndarray
    h1_bars: np.ndarray
    h0_betti: int
    h1_betti: float
    h0_total_persistence: float
    h1_total_persistence: float
    max_finite_death_h0: float
    max_death_h1: float
    epsilon_cutoff: float


@dataclass(frozen=True, slots=True)
class TopologicalFractureSignal:
    delta_beta0: float
    delta_beta1: float
    fracture_score: float
    normalized_fracture: float
    is_anomaly: bool
    anomaly_zscore: float
    h0_betti_history: np.ndarray
    h1_betti_history: np.ndarray


@dataclass(slots=True)
class ManifoldState:
    persistence_history: list[PersistenceDiagram] = field(default_factory=list)
    beta0_history: list[float] = field(default_factory=list)
    beta1_history: list[float] = field(default_factory=list)
    fracture_history: list[float] = field(default_factory=list)
    distance_matrix_history: list[cp.ndarray] = field(default_factory=list)

    def trim(self, max_len: int = 512) -> None:
        if len(self.persistence_history) > max_len:
            self.persistence_history = self.persistence_history[-max_len:]
            self.beta0_history = self.beta0_history[-max_len:]
            self.beta1_history = self.beta1_history[-max_len:]
            self.fracture_history = self.fracture_history[-max_len:]
            self.distance_matrix_history = self.distance_matrix_history[-max_len:]


def _gpu_pairwise_correlation_distance(X: cp.ndarray) -> cp.ndarray:
    X_norm = X - X.mean(axis=1, keepdims=True)
    norms = cp.linalg.norm(X_norm, axis=1, keepdims=True)
    norms = cp.where(norms > _EPS, norms, _EPS)
    X_unit = X_norm / norms
    corr = X_unit @ X_unit.T
    corr = cp.clip(corr, -1.0, 1.0)
    dist = cp.sqrt(cp.maximum(2.0 * (1.0 - corr), 0.0))
    cp.fill_diagonal(dist, 0.0)
    return dist


def _gpu_pairwise_euclidean_distance(X: cp.ndarray) -> cp.ndarray:
    sq_norms = cp.sum(X ** 2, axis=1, keepdims=True)
    gram = X @ X.T
    dist_sq = sq_norms + sq_norms.T - 2.0 * gram
    dist_sq = cp.maximum(dist_sq, 0.0)
    dist = cp.sqrt(dist_sq)
    cp.fill_diagonal(dist, 0.0)
    return dist


def _gpu_pairwise_cosine_distance(X: cp.ndarray) -> cp.ndarray:
    norms = cp.linalg.norm(X, axis=1, keepdims=True)
    norms = cp.where(norms > _EPS, norms, _EPS)
    X_unit = X / norms
    sim = X_unit @ X_unit.T
    sim = cp.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def _count_persistent_h0(h0_bars: np.ndarray, threshold: float) -> int:
    if h0_bars.shape[0] == 0:
        return 0
    finite_mask = np.isfinite(h0_bars[:, 1])
    infinite_mask = ~finite_mask
    persistent_finite = np.sum(
        finite_mask & ((h0_bars[:, 1] - h0_bars[:, 0]) >= threshold)
    )
    return int(np.sum(infinite_mask) + persistent_finite)


def _count_persistent_h1(h1_bars: np.ndarray, threshold: float) -> float:
    if h1_bars.shape[0] == 0:
        return 0.0
    finite_mask = np.isfinite(h1_bars[:, 1])
    if not np.any(finite_mask):
        return 0.0
    lifetimes = h1_bars[finite_mask, 1] - h1_bars[finite_mask, 0]
    significant = lifetimes[lifetimes >= threshold]
    return float(len(significant)) + float(np.sum(significant)) / max(float(np.sum(lifetimes)), _EPS)


def _total_persistence(bars: np.ndarray) -> float:
    if bars.shape[0] == 0:
        return 0.0
    finite_mask = np.isfinite(bars[:, 1])
    if not np.any(finite_mask):
        return 0.0
    return float(np.sum(bars[finite_mask, 1] - bars[finite_mask, 0]))


class VietorisRipsManifold:
    def __init__(
        self,
        max_epsilon: float = 2.0,
        persistence_threshold: float = 0.05,
        metric: Literal["correlation", "euclidean", "cosine"] = "correlation",
        max_dimension: int = 1,
        anomaly_zscore_threshold: float = 2.5,
        fracture_ema_alpha: float = 0.1,
        downsample_to: int | None = 128,
    ) -> None:
        if max_epsilon <= 0:
            raise ValueError(f"max_epsilon must be > 0, got {max_epsilon}")
        if not (0.0 < fracture_ema_alpha <= 1.0):
            raise ValueError(f"fracture_ema_alpha must be in (0, 1], got {fracture_ema_alpha}")

        self._max_epsilon = max_epsilon
        self._persistence_threshold = persistence_threshold
        self._metric = metric
        self._max_dimension = max_dimension
        self._anomaly_zscore_threshold = anomaly_zscore_threshold
        self._fracture_ema_alpha = fracture_ema_alpha
        self._downsample_to = downsample_to

        self._state: ManifoldState = ManifoldState()
        self._ema_fracture: float = 0.0
        self._ema_fracture_sq: float = 0.0
        self._ema_initialized: bool = False

    def _compute_distance_matrix_gpu(self, X: cp.ndarray) -> cp.ndarray:
        if self._metric == "correlation":
            return _gpu_pairwise_correlation_distance(X)
        elif self._metric == "euclidean":
            return _gpu_pairwise_euclidean_distance(X)
        elif self._metric == "cosine":
            return _gpu_pairwise_cosine_distance(X)
        else:
            raise ValueError(f"Unknown metric: {self._metric}")

    def _downsample_assets(self, X: cp.ndarray) -> cp.ndarray:
        if self._downsample_to is None or X.shape[0] <= self._downsample_to:
            return X
        variance = cp.var(X, axis=1)
        top_k_idx = cp.argsort(variance)[::-1][: self._downsample_to]
        return X[top_k_idx, :]

    def _gpu_dist_to_cpu_squareform(self, dist_gpu: cp.ndarray) -> np.ndarray:
        dist_cpu = cp.asnumpy(dist_gpu).astype(np.float64)
        np.fill_diagonal(dist_cpu, 0.0)
        dist_cpu = (dist_cpu + dist_cpu.T) / 2.0
        dist_cpu = np.clip(dist_cpu, 0.0, None)
        return dist_cpu

    def _run_ripser(self, dist_cpu: np.ndarray) -> dict:
        return ripser(
            dist_cpu,
            maxdim=self._max_dimension,
            distance_matrix=True,
            thresh=self._max_epsilon,
        )

    def _parse_ripser_output(
        self, ripser_result: dict, epsilon_cutoff: float
    ) -> PersistenceDiagram:
        diagrams = ripser_result["dgms"]

        h0_bars = diagrams[0] if len(diagrams) > 0 else np.empty((0, 2))
        h1_bars = diagrams[1] if len(diagrams) > 1 else np.empty((0, 2))

        h0_bars = np.where(np.isinf(h0_bars), self._max_epsilon * 10.0, h0_bars)
        h0_bars_clean = h0_bars[h0_bars[:, 1] > h0_bars[:, 0] + _EPS] if h0_bars.shape[0] > 0 else h0_bars

        h0_betti = _count_persistent_h0(h0_bars_clean, self._persistence_threshold)
        h1_betti = _count_persistent_h1(h1_bars, self._persistence_threshold)

        h0_total_pers = _total_persistence(h0_bars_clean)
        h1_total_pers = _total_persistence(h1_bars)

        max_finite_h0 = float(
            np.max(h0_bars_clean[np.isfinite(h0_bars_clean[:, 1]), 1])
        ) if h0_bars_clean.shape[0] > 0 and np.any(np.isfinite(h0_bars_clean[:, 1])) else 0.0

        max_h1 = float(
            np.max(h1_bars[np.isfinite(h1_bars[:, 1]), 1])
        ) if h1_bars.shape[0] > 0 and np.any(np.isfinite(h1_bars[:, 1])) else 0.0

        return PersistenceDiagram(
            h0_bars=h0_bars_clean,
            h1_bars=h1_bars,
            h0_betti=h0_betti,
            h1_betti=h1_betti,
            h0_total_persistence=h0_total_pers,
            h1_total_persistence=h1_total_pers,
            max_finite_death_h0=max_finite_h0,
            max_death_h1=max_h1,
            epsilon_cutoff=epsilon_cutoff,
        )

    def _update_ema(self, fracture_score: float) -> tuple[float, float]:
        alpha = self._fracture_ema_alpha
        if not self._ema_initialized:
            self._ema_fracture = fracture_score
            self._ema_fracture_sq = fracture_score ** 2
            self._ema_initialized = True
        else:
            self._ema_fracture = alpha * fracture_score + (1.0 - alpha) * self._ema_fracture
            self._ema_fracture_sq = alpha * fracture_score ** 2 + (1.0 - alpha) * self._ema_fracture_sq

        ema_var = max(self._ema_fracture_sq - self._ema_fracture ** 2, _EPS)
        ema_std = np.sqrt(ema_var)
        return self._ema_fracture, ema_std

    def _compute_fracture_score(
        self,
        current: PersistenceDiagram,
        previous: PersistenceDiagram | None,
    ) -> tuple[float, float, float]:
        if previous is None:
            return 0.0, 0.0, 0.0

        delta_b0 = float(current.h0_betti - previous.h0_betti)
        delta_b1 = float(current.h1_betti - previous.h1_betti)

        h0_weight = 1.5
        h1_weight = 2.0
        fracture = h0_weight * abs(delta_b0) + h1_weight * abs(delta_b1)
        fracture += 0.5 * (
            abs(current.h0_total_persistence - previous.h0_total_persistence) +
            abs(current.h1_total_persistence - previous.h1_total_persistence)
        )
        return delta_b0, delta_b1, fracture

    def compute_persistence(
        self,
        asset_returns: cp.ndarray,
        epsilon_cutoff: float | None = None,
    ) -> PersistenceDiagram:
        if asset_returns.ndim != 2:
            raise ValueError(f"asset_returns must be 2D (N_assets x T_window), got {asset_returns.shape}")

        X = self._downsample_assets(asset_returns)
        dist_gpu = self._compute_distance_matrix_gpu(X)
        dist_cpu = self._gpu_dist_to_cpu_squareform(dist_gpu)

        eps = epsilon_cutoff if epsilon_cutoff is not None else self._max_epsilon
        ripser_result = self._run_ripser(dist_cpu)
        diagram = self._parse_ripser_output(ripser_result, eps)

        self._state.distance_matrix_history.append(dist_gpu)
        self._state.persistence_history.append(diagram)
        self._state.beta0_history.append(float(diagram.h0_betti))
        self._state.beta1_history.append(float(diagram.h1_betti))
        self._state.trim()

        return diagram

    def compute_fracture_signal(
        self,
        asset_returns: cp.ndarray,
        epsilon_cutoff: float | None = None,
    ) -> TopologicalFractureSignal:
        current = self.compute_persistence(asset_returns, epsilon_cutoff)

        previous: PersistenceDiagram | None = (
            self._state.persistence_history[-2]
            if len(self._state.persistence_history) >= 2 else None
        )

        delta_b0, delta_b1, fracture = self._compute_fracture_score(current, previous)
        ema_mu, ema_std = self._update_ema(fracture)

        z_score = (fracture - ema_mu) / max(ema_std, _EPS)
        is_anomaly = abs(z_score) > self._anomaly_zscore_threshold

        max_historical = max(self._state.fracture_history) if self._state.fracture_history else 1.0
        normalized = fracture / max(max_historical, _EPS)

        self._state.fracture_history.append(fracture)

        h0_arr = np.array(self._state.beta0_history[-_BETTI_FRACTURE_WINDOW:])
        h1_arr = np.array(self._state.beta1_history[-_BETTI_FRACTURE_WINDOW:])

        signal = TopologicalFractureSignal(
            delta_beta0=delta_b0,
            delta_beta1=delta_b1,
            fracture_score=fracture,
            normalized_fracture=float(np.clip(normalized, 0.0, 1.0)),
            is_anomaly=bool(is_anomaly),
            anomaly_zscore=float(z_score),
            h0_betti_history=h0_arr,
            h1_betti_history=h1_arr,
        )

        return signal

    def compute_wasserstein_distance(
        self,
        diag_a: PersistenceDiagram,
        diag_b: PersistenceDiagram,
        dimension: int = 1,
    ) -> float:
        try:
            from persim import wasserstein

            bars_a = diag_a.h1_bars if dimension == 1 else diag_a.h0_bars
            bars_b = diag_b.h1_bars if dimension == 1 else diag_b.h0_bars

            finite_a = bars_a[np.isfinite(bars_a).all(axis=1)]
            finite_b = bars_b[np.isfinite(bars_b).all(axis=1)]

            if finite_a.shape[0] == 0 or finite_b.shape[0] == 0:
                return 0.0

            return float(wasserstein(finite_a, finite_b))
        except ImportError:
            return float(abs(diag_a.h1_total_persistence - diag_b.h1_total_persistence))

    def rolling_tda_signal(
        self,
        full_return_matrix: cp.ndarray,
        window: int = 60,
        stride: int = 5,
    ) -> np.ndarray:
        n_assets, T = full_return_matrix.shape
        fracture_series: list[float] = []

        for t_end in range(window, T + 1, stride):
            t_start = t_end - window
            window_slice = full_return_matrix[:, t_start:t_end]
            signal = self.compute_fracture_signal(window_slice)
            fracture_series.append(signal.normalized_fracture)

        return np.array(fracture_series)

    def get_distance_matrix(self, asset_returns: cp.ndarray) -> cp.ndarray:
        X = self._downsample_assets(asset_returns)
        return self._compute_distance_matrix_gpu(X)

    def reset_state(self) -> None:
        self._state = ManifoldState()
        self._ema_fracture = 0.0
        self._ema_fracture_sq = 0.0
        self._ema_initialized = False

    @property
    def state(self) -> ManifoldState:
        return self._state

    @property
    def beta0_history(self) -> list[float]:
        return self._state.beta0_history

    @property
    def beta1_history(self) -> list[float]:
        return self._state.beta1_history

    @property
    def fracture_history(self) -> list[float]:
        return self._state.fracture_history
