from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class TickData:
    ticker: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float = 0.0
    ask: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0 if self.bid and self.ask else self.close

    @property
    def spread_bps(self) -> float:
        if not self.bid or not self.ask or self.mid == 0:
            return 0.0
        return (self.ask - self.bid) / self.mid * 10_000


@dataclass
class UniverseSnapshot:
    timestamp: float
    prices: dict[str, float]
    returns: np.ndarray | None
    return_matrix: np.ndarray | None


class DataFeed(ABC):
    @abstractmethod
    async def subscribe(self, tickers: Sequence[str]) -> None: ...

    @abstractmethod
    async def get_snapshot(self) -> UniverseSnapshot | None: ...

    @abstractmethod
    async def stream_ticks(self) -> AsyncIterator[TickData]: ...

    @abstractmethod
    async def get_historical(
        self,
        tickers: Sequence[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]: ...

    @abstractmethod
    def build_return_matrix(
        self, historical: dict[str, pd.DataFrame], tickers: Sequence[str]
    ) -> np.ndarray: ...

    @abstractmethod
    def is_connected(self) -> bool: ...


class YFinanceFeed(DataFeed):
    def __init__(self, tickers: Sequence[str]) -> None:
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        self._tickers = list(tickers)
        self._connected = False
        self._price_cache: dict[str, float] = {}
        self._return_matrix: np.ndarray | None = None
        self._last_update: float = 0.0

    async def subscribe(self, tickers: Sequence[str]) -> None:
        self._tickers = list(tickers)
        logger.info(f"YFinanceFeed: subscribing {len(self._tickers)} tickers (EOD mode)")
        await self._refresh_historical(days=120)
        self._connected = True

    async def get_snapshot(self) -> UniverseSnapshot | None:
        if not self._connected or self._return_matrix is None:
            return None

        now = time.time()
        if now - self._last_update > 3600:
            await self._refresh_historical(days=5)

        return UniverseSnapshot(
            timestamp=time.time(),
            prices=self._price_cache.copy(),
            returns=(
                self._return_matrix[:, -1]
                if self._return_matrix.shape[1] > 0
                else np.zeros(len(self._tickers))
            ),
            return_matrix=self._return_matrix,
        )

    async def stream_ticks(self) -> AsyncIterator[TickData]:
        # YFinance is EOD-only — no tick stream available.
        # Use PolygonFeed for real-time data.
        while True:
            await asyncio.sleep(86400)
            yield  # type: ignore[misc]  # unreachable but satisfies AsyncIterator protocol

    async def get_historical(
        self,
        tickers: Sequence[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        data = {}
        try:
            df = self._yf.download(
                list(tickers), start=start, end=end, interval=interval,
                auto_adjust=True, progress=False, group_by="ticker"
            )
            for ticker in tickers:
                ticker_df = df[ticker] if len(tickers) > 1 and ticker in df else (
                    df if len(tickers) == 1 else pd.DataFrame()
                )
                if not ticker_df.empty:
                    ticker_df = ticker_df.ffill().bfill().dropna()
                    data[ticker] = ticker_df
        except Exception as e:
            logger.error(f"YFinance batch download failed: {e}")
        return data

    def build_return_matrix(
        self, historical: dict[str, pd.DataFrame], tickers: Sequence[str]
    ) -> np.ndarray:
        closes: dict[str, pd.Series] = {}
        for ticker in tickers:
            if ticker in historical and not historical[ticker].empty:
                closes[ticker] = historical[ticker]["Close"]

        if not closes:
            raise RuntimeError("No valid historical data from YFinance")

        df_closes = pd.DataFrame(closes).ffill().bfill().dropna()
        log_rets = np.log(df_closes / df_closes.shift(1)).dropna()
        matrix = np.clip(log_rets.values.T, -0.30, 0.30)
        return np.nan_to_num(matrix, nan=0.0)

    async def _refresh_historical(self, days: int = 120) -> None:
        try:
            end = pd.Timestamp.today().strftime("%Y-%m-%d")
            start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")

            hist = await self.get_historical(self._tickers, start, end)
            if not hist:
                logger.warning("YFinance returned empty data.")
                return

            for ticker, df in hist.items():
                if not df.empty:
                    self._price_cache[ticker] = float(df["Close"].iloc[-1])

            self._return_matrix = self.build_return_matrix(hist, self._tickers)

            if self._return_matrix is None or self._return_matrix.shape[1] < 2:
                logger.error("Insufficient data to build return matrix.")
                return

            self._last_update = time.time()
            logger.info(
                f"Data refreshed. Matrix: {self._return_matrix.shape[0]}×{self._return_matrix.shape[1]}"
            )

        except Exception as e:
            logger.error(f"YFinance refresh failed: {e}", exc_info=True)

    def is_connected(self) -> bool:
        return self._connected


class PolygonFeed(DataFeed):
    """
    Polygon.io WebSocket feed for real-time market data.
    Requires TOPARB_POLYGON_KEY env variable.
    """

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "Polygon API key required. Set TOPARB_POLYGON_KEY environment variable."
            )
        self._api_key = api_key
        self._tickers: list[str] = []
        self._connected = False
        self._ws = None
        self._tick_queue: asyncio.Queue[TickData] = asyncio.Queue(maxsize=10_000)
        self._snapshots: dict[str, TickData] = {}

    async def subscribe(self, tickers: Sequence[str]) -> None:
        self._tickers = list(tickers)
        try:
            import websockets
            self._websockets = websockets
        except ImportError:
            raise ImportError("websockets not installed. Run: pip install websockets")

        asyncio.create_task(self._ws_loop())
        logger.info(f"PolygonFeed: subscribing {len(self._tickers)} tickers")

    async def get_snapshot(self) -> UniverseSnapshot | None:
        if not self._snapshots:
            return None
        prices = {t: tick.close for t, tick in self._snapshots.items()}
        return UniverseSnapshot(
            timestamp=time.time(),
            prices=prices,
            returns=None,
            return_matrix=None,
        )

    async def stream_ticks(self) -> AsyncIterator[TickData]:
        while True:
            tick = await self._tick_queue.get()
            yield tick

    async def get_historical(
        self,
        tickers: Sequence[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        import urllib.request
        import json
        data = {}
        span = "day" if interval == "1d" else "minute"
        for ticker in tickers:
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{span}"
                f"/{start}/{end}?adjusted=true&sort=asc&limit=5000"
                f"&apiKey={self._api_key}"
            )
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    payload = json.loads(resp.read())
                results = payload.get("results", [])
                if not results:
                    continue
                df = pd.DataFrame(results)
                df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
                df = df.rename(columns={
                    "o": "Open", "h": "High", "l": "Low",
                    "c": "Close", "v": "Volume",
                })
                df.set_index("timestamp", inplace=True)
                data[ticker] = df
            except Exception as e:
                logger.error(f"Polygon historical fetch failed for {ticker}: {e}")
        return data

    def build_return_matrix(
        self, historical: dict[str, pd.DataFrame], tickers: Sequence[str]
    ) -> np.ndarray:
        closes: dict[str, pd.Series] = {}
        for ticker in tickers:
            if ticker in historical and not historical[ticker].empty:
                closes[ticker] = historical[ticker]["Close"]

        if not closes:
            raise RuntimeError("No valid historical data from Polygon")

        df_closes = pd.DataFrame(closes).ffill().bfill().dropna()
        log_rets = np.log(df_closes / df_closes.shift(1)).dropna()
        matrix = np.clip(log_rets.values.T, -0.30, 0.30)
        return np.nan_to_num(matrix, nan=0.0)

    def is_connected(self) -> bool:
        return self._connected

    async def _ws_loop(self) -> None:
        url = "wss://socket.polygon.io/stocks"
        while True:
            try:
                async with self._websockets.connect(url) as ws:
                    self._ws = ws
                    await ws.send(f'{{"action":"auth","params":"{self._api_key}"}}')
                    symbols = ",".join([f"AM.{t}" for t in self._tickers])
                    await ws.send(f'{{"action":"subscribe","params":"{symbols}"}}')
                    self._connected = True
                    logger.info("Polygon WebSocket connected")

                    async for message in ws:
                        import json
                        events = json.loads(message)
                        for event in events:
                            if event.get("ev") == "AM":
                                tick = TickData(
                                    ticker=event["sym"],
                                    timestamp=event["s"] / 1000.0,
                                    open=event["o"],
                                    high=event["h"],
                                    low=event["l"],
                                    close=event["c"],
                                    volume=event["v"],
                                )
                                self._snapshots[tick.ticker] = tick
                                await self._tick_queue.put(tick)

            except Exception as e:
                logger.error(f"Polygon WebSocket error: {e}. Reconnecting in 5s...")
                self._connected = False
                await asyncio.sleep(5)


def create_feed(mode: str, **kwargs) -> DataFeed:
    if mode == "yfinance":
        return YFinanceFeed(kwargs.get("tickers", []))
    elif mode == "polygon":
        api_key = kwargs.get("api_key", "")
        return PolygonFeed(api_key)
    else:
        raise ValueError(f"Unknown feed mode: {mode!r}. Use 'yfinance' or 'polygon'.")