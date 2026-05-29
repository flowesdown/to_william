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

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Final, Sequence

import cudf
import cupy as cp
import numpy as np
import pandas as pd

_OHLCV_COLS: Final[list[str]] = [
    "timestamp", "open", "high", "low", "close", "volume",
    "bid_price", "bid_size", "ask_price", "ask_size",
]

_NUMERIC_COLS: Final[list[str]] = [c for c in _OHLCV_COLS if c != "timestamp"]

_CUDF_DTYPES: Final[dict[str, str]] = {
    "timestamp": "int64",
    "open": "float32",
    "high": "float32",
    "low": "float32",
    "close": "float32",
    "volume": "float64",
    "bid_price": "float32",
    "bid_size": "float32",
    "ask_price": "float32",
    "ask_size": "float32",
}


@dataclass(frozen=True, slots=True)
class StreamConfig:
    universe_size: int
    batch_size: int
    max_vram_batches: int
    dtype_close: str = "float32"
    dtype_volume: str = "float64"
    ring_buffer_depth: int = 128
    staleness_threshold_ms: float = 250.0


@dataclass(slots=True)
class VRAMBatch:
    ticker_ids: cp.ndarray
    close_matrix: cp.ndarray
    mid_matrix: cp.ndarray
    volume_matrix: cp.ndarray
    ask_matrix: cp.ndarray
    bid_matrix: cp.ndarray
    timestamps: cp.ndarray
    ingestion_ns: int = field(default_factory=time.time_ns)

    @property
    def shape(self) -> tuple[int, int]:
        return self.close_matrix.shape

    @property
    def age_ms(self) -> float:
        return (time.time_ns() - self.ingestion_ns) / 1e6

    def is_stale(self, threshold_ms: float) -> bool:
        return self.age_ms > threshold_ms


class VRAMRingBuffer:
    def __init__(self, depth: int, universe_size: int, batch_size: int) -> None:
        self._depth = depth
        self._universe_size = universe_size
        self._batch_size = batch_size
        self._buffer: list[VRAMBatch | None] = [None] * depth
        self._head: int = 0
        self._tail: int = 0
        self._count: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    async def push(self, batch: VRAMBatch) -> None:
        async with self._lock:
            self._buffer[self._tail] = batch
            self._tail = (self._tail + 1) % self._depth
            if self._count < self._depth:
                self._count += 1
            else:
                self._head = (self._head + 1) % self._depth

    async def pop(self) -> VRAMBatch | None:
        async with self._lock:
            if self._count == 0:
                return None
            batch = self._buffer[self._head]
            self._buffer[self._head] = None
            self._head = (self._head + 1) % self._depth
            self._count -= 1
            return batch

    async def peek_latest(self) -> VRAMBatch | None:
        async with self._lock:
            if self._count == 0:
                return None
            latest_idx = (self._tail - 1) % self._depth
            return self._buffer[latest_idx]

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_full(self) -> bool:
        return self._count == self._depth


class L2VRAMStreamer:
    def __init__(self, config: StreamConfig, ticker_index: Sequence[str]) -> None:
        if len(ticker_index) != config.universe_size:
            raise ValueError(
                f"ticker_index length {len(ticker_index)} != universe_size {config.universe_size}"
            )

        self._config = config
        self._ticker_index: dict[str, int] = {t: i for i, t in enumerate(ticker_index)}
        self._ring: VRAMRingBuffer = VRAMRingBuffer(
            config.ring_buffer_depth,
            config.universe_size,
            config.batch_size,
        )

        self._close_staging: cp.ndarray = cp.zeros(
            (config.batch_size, config.universe_size), dtype=cp.float32
        )
        self._mid_staging: cp.ndarray = cp.zeros(
            (config.batch_size, config.universe_size), dtype=cp.float32
        )
        self._bid_staging: cp.ndarray = cp.zeros(
            (config.batch_size, config.universe_size), dtype=cp.float32
        )
        self._ask_staging: cp.ndarray = cp.zeros(
            (config.batch_size, config.universe_size), dtype=cp.float32
        )
        self._volume_staging: cp.ndarray = cp.zeros(
            (config.batch_size, config.universe_size), dtype=cp.float64
        )
        self._ts_staging: cp.ndarray = cp.zeros(config.batch_size, dtype=cp.int64)

        self._batch_cursor: int = 0
        self._total_batches_ingested: int = 0
        self._total_ticks_dropped: int = 0

    def _validate_raw_df(self, df: pd.DataFrame) -> None:
        missing = set(_OHLCV_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Raw feed missing columns: {missing}")

    def _pandas_to_cudf_gpu(self, df: pd.DataFrame) -> cudf.DataFrame:
        gdf = cudf.from_pandas(df[_OHLCV_COLS])
        for col, dtype in _CUDF_DTYPES.items():
            if gdf[col].dtype != dtype:
                gdf[col] = gdf[col].astype(dtype)
        return gdf

    def _extract_close_mid_volume(
        self,
        gdf: cudf.DataFrame,
        ticker: str,
        row: int,
    ) -> None:
        if ticker not in self._ticker_index:
            self._total_ticks_dropped += 1
            return

        col_idx: int = self._ticker_index[ticker]
        close_val: cp.ndarray = cp.asarray(gdf["close"].values)
        bid_val: cp.ndarray = cp.asarray(gdf["bid_price"].values)
        ask_val: cp.ndarray = cp.asarray(gdf["ask_price"].values)
        vol_val: cp.ndarray = cp.asarray(gdf["volume"].values)
        ts_val: cp.ndarray = cp.asarray(gdf["timestamp"].values)

        self._close_staging[row, col_idx] = close_val[-1]
        self._bid_staging[row, col_idx] = bid_val[-1]
        self._ask_staging[row, col_idx] = ask_val[-1]
        self._mid_staging[row, col_idx] = (bid_val[-1] + ask_val[-1]) * cp.float32(0.5)
        self._volume_staging[row, col_idx] = vol_val[-1]
        self._ts_staging[row] = ts_val[-1]

    def _flush_staging_to_batch(self) -> VRAMBatch:
        batch = VRAMBatch(
            ticker_ids=cp.arange(self._config.universe_size, dtype=cp.int32),
            close_matrix=self._close_staging.copy(),
            mid_matrix=self._mid_staging.copy(),
            bid_matrix=self._bid_staging.copy(),
            ask_matrix=self._ask_staging.copy(),
            volume_matrix=self._volume_staging.copy(),
            timestamps=self._ts_staging.copy(),
        )
        self._close_staging[:] = cp.float32(0.0)
        self._mid_staging[:] = cp.float32(0.0)
        self._bid_staging[:] = cp.float32(0.0)
        self._ask_staging[:] = cp.float32(0.0)
        self._volume_staging[:] = cp.float64(0.0)
        self._ts_staging[:] = cp.int64(0)
        self._batch_cursor = 0
        return batch

    async def ingest_tick(self, ticker: str, raw_df: pd.DataFrame) -> None:
        self._validate_raw_df(raw_df)
        gdf = self._pandas_to_cudf_gpu(raw_df)
        self._extract_close_mid_volume(gdf, ticker, self._batch_cursor)
        self._batch_cursor += 1

        if self._batch_cursor >= self._config.batch_size:
            batch = self._flush_staging_to_batch()
            await self._ring.push(batch)
            self._total_batches_ingested += 1

    async def ingest_batch_raw(
        self,
        raw_df: pd.DataFrame,
        ticker_col: str = "ticker",
    ) -> None:
        self._validate_raw_df(raw_df)
        grouped = raw_df.groupby(ticker_col)
        tasks = [
            self.ingest_tick(ticker, grp.drop(columns=[ticker_col]))
            for ticker, grp in grouped
        ]
        await asyncio.gather(*tasks)

    async def stream_batches(
        self,
        poll_interval_s: float = 0.001,
    ) -> AsyncIterator[VRAMBatch]:
        while True:
            batch = await self._ring.pop()
            if batch is not None:
                if not batch.is_stale(self._config.staleness_threshold_ms):
                    yield batch
                else:
                    self._total_ticks_dropped += batch.shape[0]
            else:
                await asyncio.sleep(poll_interval_s)

    async def get_latest_close_matrix(self) -> cp.ndarray | None:
        batch = await self._ring.peek_latest()
        if batch is None:
            return None
        return batch.close_matrix

    def build_log_return_matrix(self, batch: VRAMBatch) -> cp.ndarray:
        close = batch.close_matrix
        eps = cp.float32(1e-9)
        safe_close = cp.where(close > eps, close, eps)
        returns = cp.diff(cp.log(safe_close), axis=0)
        return returns

    def build_vwap_mid_matrix(self, batch: VRAMBatch) -> cp.ndarray:
        mid = batch.mid_matrix
        vol = batch.volume_matrix.astype(cp.float32)
        vol_sum = vol.sum(axis=0, keepdims=True)
        vol_sum = cp.where(vol_sum > cp.float32(0.0), vol_sum, cp.float32(1.0))
        vwap = (mid * vol).sum(axis=0, keepdims=True) / vol_sum
        return vwap

    def compute_spread_matrix(self, batch: VRAMBatch) -> cp.ndarray:
        return batch.ask_matrix - batch.bid_matrix

    @property
    def diagnostics(self) -> dict[str, int | float]:
        return {
            "total_batches_ingested": self._total_batches_ingested,
            "total_ticks_dropped": self._total_ticks_dropped,
            "ring_buffer_count": self._ring.count,
            "ring_buffer_full": self._ring.is_full,
            "batch_cursor": self._batch_cursor,
        }

    @property
    def universe_size(self) -> int:
        return self._config.universe_size

    @property
    def batch_size(self) -> int:
        return self._config.batch_size