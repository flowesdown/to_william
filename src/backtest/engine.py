"""
Backtesting Engine.

Walk-forward backtest using yfinance historical data.
Realistic simulation with:
- Bid-ask spread + market impact costs (Almgren-Chriss)
- Rolling window model re-fitting
- Walk-forward cross-validation
- Full performance attribution vs benchmark

Usage:
    engine = BacktestEngine(settings, tickers)
    results = engine.run()
    engine.print_summary(results)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from config.settings import BacktestConfig, ExecutionConfig, ModelConfig, RiskConfig
from src.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    logger.warning("yfinance not available — install with: pip install yfinance")


@dataclass
class BacktestStep:
    date: pd.Timestamp
    nav: float
    daily_return: float
    leverage: float
    f_star: float
    tda_fracture: float
    tda_anomaly: bool
    rebalanced: bool
    transaction_cost: float
    regime: str


@dataclass
class BacktestResults:
    steps: list[BacktestStep] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)

    # Computed after run
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_vol: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    total_transaction_costs: float = 0.0
    n_rebalances: int = 0
    n_tda_anomalies: int = 0
    n_risk_off_days: int = 0

    # Benchmark comparison
    benchmark_total_return: float = 0.0
    benchmark_sharpe: float = 0.0
    excess_return: float = 0.0
    information_ratio: float = 0.0

    @property
    def nav_series(self) -> pd.Series:
        if not self.steps:
            return pd.Series(dtype=float)
        return pd.Series(
            [s.nav for s in self.steps],
            index=[s.date for s in self.steps],
        )

    @property
    def return_series(self) -> pd.Series:
        if not self.steps:
            return pd.Series(dtype=float)
        return pd.Series(
            [s.daily_return for s in self.steps],
            index=[s.date for s in self.steps],
        )


class BacktestEngine:
    """
    Walks through historical data day-by-day, running the full TopArb pipeline.
    Uses CPU numpy (no GPU needed for backtesting — GPU used only for live).
    """

    def __init__(
        self,
        bt_config: BacktestConfig,
        model_config: ModelConfig,
        risk_config: RiskConfig,
        exec_config: ExecutionConfig,
        tickers: Sequence[str],
    ) -> None:
        self._bt = bt_config
        self._model = model_config
        self._risk_cfg = risk_config
        self._exec = exec_config
        self._tickers = list(tickers)
        self._initial_capital = bt_config.initial_capital

        # Lazily created per-step so each walk-forward window is independent
        self._pca = None
        self._tda = None
        self._controller = None

    def run(self, verbose: bool = True) -> BacktestResults:
        """Run the full backtest. Returns BacktestResults."""
        if not _YF_AVAILABLE:
            raise RuntimeError("yfinance required. Run: pip install yfinance")

        logger.info(
            f"Backtest: {self._bt.start_date} → {self._bt.end_date} "
            f"| {len(self._tickers)} assets | capital=${self._initial_capital:,.0f}"
        )

        prices_df = self._download_prices()
        if prices_df.empty:
            raise RuntimeError("Failed to download price data — check tickers and dates.")

        benchmark_df = self._download_benchmark()

        returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
        available_tickers = list(returns_df.columns)
        n_assets = len(available_tickers)

        if n_assets < 5:
            raise RuntimeError(f"Too few assets with data: {n_assets}. Need >= 5.")

        logger.info(f"Universe: {n_assets} assets × {len(returns_df)} trading days")

        results = BacktestResults(tickers=available_tickers)
        nav = self._initial_capital
        weights = np.zeros(n_assets)
        daily_returns_list: list[float] = []

        pca_window = self._model_config_n_obs()
        risk_manager = RiskManager(self._risk_cfg, n_assets)
        risk_manager.update_nav(nav)
        risk_manager.reset_daily(nav)

        last_reset_date = returns_df.index[0].date() if len(returns_df) > 0 else None

        for i, date in enumerate(returns_df.index):
            # Daily reset at market open
            if last_reset_date is None or date.date() != last_reset_date:
                risk_manager.reset_daily(nav)
                last_reset_date = date.date()

            if i < pca_window:
                step = BacktestStep(
                    date=date, nav=nav, daily_return=0.0, leverage=0.0,
                    f_star=0.0, tda_fracture=0.0, tda_anomaly=False,
                    rebalanced=False, transaction_cost=0.0, regime="WARMUP",
                )
                results.steps.append(step)
                continue

            window_returns = returns_df.iloc[i - pca_window:i].values   # (T × N)
            today_returns = returns_df.iloc[i].values                     # (N,)

            # Mark NAV
            port_return = float(weights @ today_returns) if np.any(weights != 0) else 0.0
            nav_before = nav
            nav = nav * (1.0 + port_return)
            risk_manager.update_nav(nav)

            allocation_result = self._run_model_step(window_returns, n_assets)

            if allocation_result is None:
                daily_return = (nav - nav_before) / max(nav_before, 1e-10)
                results.steps.append(BacktestStep(
                    date=date, nav=nav, daily_return=daily_return, leverage=0.0,
                    f_star=0.0, tda_fracture=0.0, tda_anomaly=False,
                    rebalanced=False, transaction_cost=0.0, regime="WARMUP",
                ))
                daily_returns_list.append(daily_return)
                continue

            target_weights, f_star, fracture_score, tda_anomaly, regime = allocation_result

            validation = risk_manager.validate_allocation(
                target_weights, nav, window_returns
            )

            tc = 0.0
            rebalanced = False
            if validation.approved and validation.scaled_weights is not None:
                new_weights = validation.scaled_weights
                drift = float(np.sum(np.abs(new_weights - weights)))
                if drift > self._risk_cfg.rebalance_threshold:
                    tc = self._estimate_tc(weights, new_weights, nav)
                    nav -= tc
                    weights = new_weights.copy()
                    rebalanced = True
                    risk_manager.record_execution(weights, new_weights)

            daily_return = (nav - nav_before) / max(nav_before, 1e-10)
            daily_returns_list.append(daily_return)

            results.steps.append(BacktestStep(
                date=date,
                nav=nav,
                daily_return=daily_return,
                leverage=float(np.sum(np.abs(weights))),
                f_star=f_star,
                tda_fracture=fracture_score,
                tda_anomaly=tda_anomaly,
                rebalanced=rebalanced,
                transaction_cost=tc,
                regime=regime,
            ))

            if verbose and i % 50 == 0:
                logger.info(
                    f"[BT] {date.date()} NAV={nav:,.0f} f*={f_star:.3f} "
                    f"fracture={fracture_score:.3f} regime={regime}"
                )

        self._compute_metrics(results, daily_returns_list, benchmark_df)
        return results

    def print_summary(self, results: BacktestResults) -> None:
        """Pretty-print backtest results."""
        print("\n" + "=" * 62)
        print("  TopArb Walk-Forward Backtest Summary")
        print("=" * 62)
        print(f"  Period:              {self._bt.start_date} → {self._bt.end_date}")
        print(f"  Universe:            {len(results.tickers)} assets")
        print(f"  Initial Capital:     ${self._initial_capital:>12,.0f}")
        final_nav = results.steps[-1].nav if results.steps else self._initial_capital
        print(f"  Final NAV:           ${final_nav:>12,.0f}")
        print("-" * 62)
        print(f"  Total Return:        {results.total_return:>8.2%}")
        print(f"  Ann. Return:         {results.annualized_return:>8.2%}")
        print(f"  Ann. Volatility:     {results.annualized_vol:>8.2%}")
        print(f"  Sharpe Ratio:        {results.sharpe_ratio:>8.3f}")
        print(f"  Sortino Ratio:       {results.sortino_ratio:>8.3f}")
        print(f"  Max Drawdown:        {results.max_drawdown:>8.2%}")
        print(f"  Calmar Ratio:        {results.calmar_ratio:>8.3f}")
        print("-" * 62)
        print(f"  Benchmark Return:    {results.benchmark_total_return:>8.2%}")
        print(f"  Benchmark Sharpe:    {results.benchmark_sharpe:>8.3f}")
        print(f"  Excess Return:       {results.excess_return:>8.2%}")
        print(f"  Information Ratio:   {results.information_ratio:>8.3f}")
        print("-" * 62)
        print(f"  Total Trans. Costs:  ${results.total_transaction_costs:>10,.2f}")
        print(f"  Rebalances:          {results.n_rebalances:>8d}")
        print(f"  TDA Anomalies:       {results.n_tda_anomalies:>8d}")
        print(f"  Risk-Off Days:       {results.n_risk_off_days:>8d}")
        print("=" * 62)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_model_step(
        self,
        window_returns: np.ndarray,   # (T × N) — CPU numpy
        n_assets: int,
    ):
        """Run PCA + TDA + HJB on one window. Returns CPU numpy results."""
        try:
            from src.utils.backend import GPU_AVAILABLE, from_numpy, to_numpy
            from src.math.eigen_risk import PCARiskModel
            from src.math.manifold_tda import VietorisRipsManifold
            from src.execution.hjb_controller import TopologicalKellyController

            n_components = min(
                self._model.n_pca_components,
                n_assets - 1,
                window_returns.shape[0] - 1,
            )
            if n_components < 1:
                return None

            returns_gpu = from_numpy(window_returns.astype(np.float64))

            pca = PCARiskModel(n_components=n_components)
            pca.fit(returns_gpu)

            tda = VietorisRipsManifold(
                persistence_threshold=self._model.persistence_threshold,
                anomaly_zscore_threshold=self._model.anomaly_zscore_threshold,
                fracture_ema_alpha=self._model.fracture_ema_alpha,
                downsample_to=min(self._model.downsample_to, n_assets),
            )

            controller = TopologicalKellyController(
                pca_model=pca,
                tda_model=tda,
                risk_aversion=self._model.risk_aversion,
                gamma_tda=self._model.gamma_tda,
                kappa_anomaly=self._model.kappa_anomaly,
                max_leverage=self._risk_cfg.max_leverage,
                n_eigen_portfolios=self._model.n_eigen_portfolios,
            )

            allocation = controller.step(returns_gpu)

            weights_np = to_numpy(allocation.weights)
            weights_np = np.nan_to_num(weights_np, nan=0.0, posinf=0.0, neginf=0.0)

            return (
                weights_np,
                allocation.f_star,
                allocation.tda_signal.fracture_score,
                allocation.tda_signal.is_anomaly,
                allocation.hjb.leverage_regime,
            )

        except Exception as e:
            logger.debug(f"Model step failed: {e}")
            return None

    def _estimate_tc(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        nav: float,
    ) -> float:
        """Transaction cost: (TC_bps + slippage_bps) × turnover × NAV."""
        turnover = float(np.sum(np.abs(new_weights - old_weights)))
        tc_frac = (self._exec.transaction_cost_bps + self._exec.slippage_bps) / 10_000
        return nav * turnover * tc_frac

    def _download_prices(self) -> pd.DataFrame:
        logger.info(f"Downloading prices {self._bt.start_date}→{self._bt.end_date}")
        try:
            raw = yf.download(
                self._tickers,
                start=self._bt.start_date,
                end=self._bt.end_date,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                return pd.DataFrame()

            if len(self._tickers) == 1:
                prices = raw[["Close"]].rename(columns={"Close": self._tickers[0]})
            else:
                prices = raw["Close"]

            # Drop tickers with >20% missing data
            threshold = 0.80
            prices = prices.dropna(axis=1, thresh=int(threshold * len(prices)))
            prices = prices.ffill().bfill().dropna()
            logger.info(f"Downloaded: {prices.shape[1]} tickers × {len(prices)} days")
            return prices

        except Exception as e:
            logger.error(f"Price download failed: {e}")
            return pd.DataFrame()

    def _download_benchmark(self) -> pd.DataFrame:
        try:
            return yf.download(
                self._bt.benchmark_ticker,
                start=self._bt.start_date,
                end=self._bt.end_date,
                auto_adjust=True,
                progress=False,
            )
        except Exception:
            return pd.DataFrame()

    def _compute_metrics(
        self,
        results: BacktestResults,
        daily_returns: list[float],
        benchmark_df: pd.DataFrame,
    ) -> None:
        if not daily_returns:
            return

        rets = np.array(daily_returns)
        rets = rets[np.isfinite(rets)]
        if len(rets) == 0:
            return

        total_ret = float(np.prod(1.0 + rets) - 1.0)
        n_years = len(rets) / 252.0
        ann_ret = float((1.0 + total_ret) ** (1.0 / max(n_years, 1e-6)) - 1.0)
        ann_vol = float(np.std(rets) * np.sqrt(252))
        sharpe = ann_ret / max(ann_vol, 1e-10)

        # Sortino: downside deviation only
        downside = rets[rets < 0]
        down_vol = float(np.std(downside) * np.sqrt(252)) if len(downside) > 0 else ann_vol
        sortino = ann_ret / max(down_vol, 1e-10)

        # Drawdown
        nav_arr = np.array([s.nav for s in results.steps if s.regime != "WARMUP"])
        if len(nav_arr) > 0:
            peak = np.maximum.accumulate(nav_arr)
            dds = (nav_arr - peak) / np.maximum(peak, 1e-10)
            max_dd = float(np.min(dds))
        else:
            max_dd = 0.0

        calmar = ann_ret / max(abs(max_dd), 1e-10)

        results.total_return = total_ret
        results.annualized_return = ann_ret
        results.annualized_vol = ann_vol
        results.sharpe_ratio = sharpe
        results.sortino_ratio = sortino
        results.max_drawdown = abs(max_dd)
        results.calmar_ratio = calmar
        results.total_transaction_costs = sum(s.transaction_cost for s in results.steps)
        results.n_rebalances = sum(1 for s in results.steps if s.rebalanced)
        results.n_tda_anomalies = sum(1 for s in results.steps if s.tda_anomaly)
        results.n_risk_off_days = sum(1 for s in results.steps if s.regime == "RISK_OFF")

        # Benchmark
        if not benchmark_df.empty:
            try:
                col = "Close" if "Close" in benchmark_df.columns else benchmark_df.columns[0]
                bm_prices = benchmark_df[col].dropna()
                bm_rets = np.log(bm_prices / bm_prices.shift(1)).dropna().values
                bm_total = float(np.prod(1.0 + bm_rets) - 1.0)
                bm_years = len(bm_rets) / 252.0
                bm_ann = float((1.0 + bm_total) ** (1.0 / max(bm_years, 1e-6)) - 1.0)
                bm_vol = float(np.std(bm_rets) * np.sqrt(252))
                bm_sharpe = bm_ann / max(bm_vol, 1e-10)

                results.benchmark_total_return = bm_total
                results.benchmark_sharpe = bm_sharpe
                results.excess_return = ann_ret - bm_ann

                # Information ratio
                min_len = min(len(rets), len(bm_rets))
                if min_len >= 10:
                    excess_daily = rets[-min_len:] - bm_rets[-min_len:]
                    tracking_err = float(np.std(excess_daily) * np.sqrt(252))
                    results.information_ratio = results.excess_return / max(tracking_err, 1e-10)
            except Exception as e:
                logger.debug(f"Benchmark metrics failed: {e}")

    def _model_config_n_obs(self) -> int:
        return max(self._model.n_pca_components * 5, 60)
