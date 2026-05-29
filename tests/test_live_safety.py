import pytest
import numpy as np
import numpy as cp
from src.execution.order_router import IBKROrderRouter
from config.settings import ExecutionConfig


class DummyIB:
    def __init__(self):
        self.orders_placed = []

    def placeOrder(self, contract, order):
        self.orders_placed.append(order)

        class MockTrade:
            class MockStatus:
                status = "Filled"
                avgFillPrice = order.lmtPrice
                filled = order.totalQuantity

            orderStatus = MockStatus()
            order = type('obj', (object,), {'orderId': 123})

        return MockTrade()


@pytest.mark.asyncio
async def test_ibkr_router_uses_limit_orders_only(risk_manager, n_assets, tickers):
    pytest.importorskip("ib_insync")
    config = ExecutionConfig()
    config.paper_trading = False
    router = IBKROrderRouter(config)
    router._ib = DummyIB()
    router._connected_flag = True

    target = np.zeros(n_assets)
    target[0] = 0.1  # BUY
    target[1] = -0.1  # SELL
    current = np.zeros(n_assets)
    prices = {t: 100.0 for t in tickers}

    await router.submit_rebalance(target, current, tickers, prices, 1_000_000.0, risk_manager)

    orders = router._ib.orders_placed
    assert len(orders) == 2
    for order in orders:
        assert order.orderType == "LMT", "CRITICAL: Market order detected!"
        assert order.algoStrategy == "Adaptive", "Missing execution algo"


@pytest.mark.asyncio
async def test_risk_manager_rejects_nan_prices(risk_manager, n_assets, tickers):
    target = np.ones(n_assets) / n_assets
    current = np.zeros(n_assets)
    prices = {t: 100.0 for t in tickers}
    prices[tickers[0]] = np.nan

    from src.execution.order_router import PaperOrderRouter
    router = PaperOrderRouter(ExecutionConfig())
    result = await router.submit_rebalance(target, current, tickers, prices, 1_000_000.0, risk_manager)
    assert result.orders_submitted == n_assets - 1


def test_controller_survives_infinite_volatility(n_assets):
    from src.execution.hjb_controller import TopologicalKellyController
    from src.math.eigen_risk import PCARiskModel
    from src.math.manifold_tda import VietorisRipsManifold

    controller = TopologicalKellyController(PCARiskModel(n_components=3), VietorisRipsManifold())
    crazy_returns = np.full((60, n_assets), -0.99)
    allocation = controller.step(crazy_returns)
    assert np.isfinite(allocation.f_star)
    assert allocation.f_star == 0.0