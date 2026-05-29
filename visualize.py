import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from src.backtest.backtester import TopArbBacktester
from config.settings import settings

# Настройки для графиков
plt.style.use('dark_background')  # Темы NVIDIA/RAPIDS стиле


def run_and_plot():
    print("Запуск бэктеста для визуализации...")

    # 1. Подготовка данных (как в main.py)
    import yfinance as yf
    tickers = settings.universe.default_tickers[:settings.universe.size]
    start, end = "2024-01-01", "2026-01-01"

    raw = yf.download(tickers, start=start, end=end, progress=False)
    close_series = raw['Close'].dropna(axis=1, thresh=len(raw) * 0.8).ffill().bfill()

    # Лог-доходности
    return_matrix = np.log(close_series / close_series.shift(1)).dropna().values.T
    return_matrix = np.clip(return_matrix, -0.30, 0.30)

    # 2. Инициализация бэктестера
    # Берем агрессивные параметры из окружения или ставим вручную
    bt = TopArbBacktester(refit_every=10)
    report = bt.run(return_matrix, train_window=60)

    # 3. Визуализация
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Верхний график: Equity Curve
    nav_series = pd.Series(report.nav_series)
    # Считаем бенчмарк (просто равномерное распределение по всем акциям)
    benchmark = (1 + pd.DataFrame(return_matrix).T.mean(axis=1)).cumprod() * settings.backtest.initial_capital

    ax1.plot(nav_series.index, nav_series.values, label='TopArb (GPU Accelerated)', color='#76b900', lw=2)
    ax1.plot(nav_series.index, benchmark.values[:len(nav_series)], label='Market (Equal Weight)', color='gray',
             alpha=0.6, linestyle='--')

    ax1.set_title(f"TopArb Performance: RA={os.getenv('TOPARB_RISK_AVERSION', '0.1')}", fontsize=16, color='#76b900')
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(alpha=0.2)

    # Нижний график: Drawdown
    peak = nav_series.cummax()
    dd = (nav_series - peak) / peak
    ax2.fill_between(dd.index, dd.values * 100, 0, color='red', alpha=0.3)
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Trading Days")
    ax2.grid(alpha=0.2)

    plt.tight_layout()

    # Сохраняем и показываем
    filename = "performance_chart.png"
    plt.savefig(filename)
    print(f"График сохранен в файл: {filename}")
    plt.show()


if __name__ == "__main__":
    run_and_plot()