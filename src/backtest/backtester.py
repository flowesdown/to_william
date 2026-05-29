"""
src/backtest/backtester.py
"""
import numpy as np
import logging
from dataclasses import dataclass
from config.settings import ModelConfig, RiskConfig, BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestReport:
    nav_series: np.ndarray
    daily_returns: np.ndarray
    sharpe_ratio: float
    max_drawdown: float
    regime_breakdown: dict[str, float]
    n_rebalances: int
    avg_leverage: float
    tda_anomaly_rate: float

    def print_summary(self):
        print("\n" + "=" * 60)
        print("  TopArb Backtest Summary")
        print("=" * 60)
        total_ret = (self.nav_series[-1] / self.nav_series[0] - 1) * 100
        print(f"  Total Return:      {total_ret:.2f}%")
        print(f"  Sharpe Ratio:      {self.sharpe_ratio:.3f}")
        print(f"  Max Drawdown:      {self.max_drawdown * 100:.2f}%")
        print(f"  Avg Leverage:      {self.avg_leverage:.2f}x")
        print(f"  TDA Anomaly Rate:  {self.tda_anomaly_rate * 100:.1f}%")
        print(f"  Total Rebalances:  {self.n_rebalances}")
        print("=" * 60)


class TopArbBacktester:
    def __init__(self, model_config=None, risk_config=None, backtest_config=None, refit_every=10):
        self.model_config = model_config or ModelConfig()
        self.risk_config = risk_config or RiskConfig()
        self.backtest_config = backtest_config or BacktestConfig()
        self.refit_every = refit_every

    def run(self, return_matrix, tickers=None, train_window=60, verbose=False):
        n_assets, n_obs = return_matrix.shape
        if n_obs <= train_window:
            raise ValueError(f"Insufficient data: {n_obs} obs for {train_window} window.")

        from src.math.eigen_risk import PCARiskModel
        from src.math.manifold_tda import VietorisRipsManifold
        from src.execution.hjb_controller import TopologicalKellyController
        from src.utils.backend import from_numpy, to_numpy

        nav = [self.backtest_config.initial_capital]
        daily_rets = []
        leverages = []
        regimes = {"RISK_OFF": 0, "TRANSITION": 0, "RISK_ON": 0}
        anomalies = 0
        rebalances = 0

        # Инициализация математики
        pca = PCARiskModel(n_components=min(self.model_config.n_pca_components, n_assets - 1))
        tda = VietorisRipsManifold(
            persistence_threshold=self.model_config.persistence_threshold,
            anomaly_zscore_threshold=self.model_config.anomaly_zscore_threshold,
            downsample_to=min(self.model_config.downsample_to, n_assets)
        )
        controller = TopologicalKellyController(
            pca_model=pca, tda_model=tda,
            risk_aversion=self.model_config.risk_aversion,
            gamma_tda=self.model_config.gamma_tda,
            max_leverage=self.risk_config.max_leverage
        )

        weights = np.zeros(n_assets)

        # Симуляция Walk-Forward
        for t in range(train_window, n_obs):
            window_data = return_matrix[:, t - train_window:t]
            today_ret = return_matrix[:, t]

            # Считаем PnL портфеля за сегодня (по весам вчерашнего дня)
            port_ret = float(np.dot(weights, today_ret))
            current_nav = nav[-1] * (1.0 + port_ret)
            nav.append(current_nav)
            daily_rets.append(port_ret)

            # Шаг модели HJB (расчет новых весов)
            alloc = controller.step(from_numpy(window_data.T))
            new_weights = to_numpy(alloc.weights)
            new_weights = np.nan_to_num(new_weights, nan=0.0)

            # Сбор метрик
            regime = alloc.hjb.leverage_regime
            if regime in regimes:
                regimes[regime] += 1
            leverages.append(float(np.sum(np.abs(new_weights))))
            if alloc.tda_signal.is_anomaly:
                anomalies += 1

            # Ребалансировка
            if t % self.refit_every == 0:
                weights = new_weights
                rebalances += 1

        nav_arr = np.array(nav)
        rets_arr = np.array(daily_rets)

        # Финальные статистики
        ann_ret = np.mean(rets_arr) * 252
        ann_vol = np.std(rets_arr) * np.sqrt(252)
        sharpe = ann_ret / max(ann_vol, 1e-10)

        peak = np.maximum.accumulate(nav_arr)
        dd = (nav_arr - peak) / np.maximum(peak, 1e-10)
        max_dd = float(np.min(dd))

        total_steps = max(len(leverages), 1)
        regime_pct = {k: v / total_steps for k, v in regimes.items()}

        return BacktestReport(
            nav_series=nav_arr,
            daily_returns=rets_arr,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            regime_breakdown=regime_pct,
            n_rebalances=rebalances,
            avg_leverage=float(np.mean(leverages)),
            tda_anomaly_rate=anomalies / total_steps
        )