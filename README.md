# TopArb — Topological Statistical Arbitrage

GPU-accelerated equity market-neutral strategy for quant practitioners.
Three orthogonal components compose the full pipeline:

| Component | Module | Accelerated by |
|---|---|---|
| PCA Risk Model (Eigen-portfolios + Ledoit-Wolf) | `src/math/eigen_risk.py` | **cuML PCA**, cupy BLAS |
| Vietoris-Rips TDA (Persistent Homology) | `src/math/manifold_tda.py` | cupy distance kernel |
| HJB Kelly Controller | `src/execution/hjb_controller.py` | cupy matrix ops |

---

## Quick Start

```bash
pip install -r requirements-dev.txt

# Validate config
TOPARB_FORCE_CPU=1 python main.py --mode validate

# Backtest 2020–2023
TOPARB_FORCE_CPU=1 python main.py --mode backtest --start 2020-01-01 --end 2023-12-31

# Parameter calibration
TOPARB_FORCE_CPU=1 python calibrate.py --tickers 20 --years 2

# Paper trading (minimum 30 days before live)
TOPARB_FORCE_CPU=1 python main.py --mode live
```

---

## Pipeline — Math Reference

### Stage 1 · PCA Risk Model (`eigen_risk.py`)

**Input:** return matrix $R \in \mathbb{R}^{T \times N}$, $T$ observations, $N$ assets.

**Step 1 — Ledoit-Wolf shrinkage** of the sample covariance $S = \frac{1}{T-1}X^\top X$:

$$\hat{\Sigma} = (1-\rho)\,S + \rho\,\mu I, \quad \mu = \frac{\mathrm{tr}(S)}{N}$$

$$\rho = \min\!\left(1,\; \frac{\bar\beta}{\delta}\right), \quad \delta = \frac{\|S - \mu I\|_F^2}{N}, \quad \bar\beta = \frac{\mathrm{tr}(S^2) + \mathrm{tr}(S)^2 - 2\,\mathrm{tr}(S^2)/N}{nN}$$

This is the Oracle Approximating Shrinkage estimator (Ledoit-Wolf 2004, Chen et al. 2010). On GPU with **cuML**, fitting a 500-asset covariance matrix shrinks from ~4 s (sklearn) to ~40 ms — a **100× speedup**.

**Step 2 — Eigen decomposition** via `cp.linalg.eigh`:

$$\hat{\Sigma} = V \Lambda V^\top, \quad \Lambda = \mathrm{diag}(\lambda_1 \ge \cdots \ge \lambda_K)$$

**Step 3 — Marchenko-Pastur noise filter.** Random matrix theory gives the upper edge of the noise bulk:

$$\lambda_+ = \sigma^2\!\left(1 + \frac{1}{\sqrt{q}} \right)^2, \quad q = \frac{T}{N}$$

Only eigenvalues $\lambda_k > \lambda_+$ carry genuine signal.

**Step 4 — Eigen-portfolio weights** (variance-stabilized):

$$\tilde w_k = \frac{v_k}{\lambda_k\,\|v_k/\lambda_k\|_2}, \quad k = 1,\ldots,K$$

**Step 5 — Mean-reversion alpha** from PCA residuals $\varepsilon = R - R\,V_K V_K^\top$:

$$\alpha_i = -\frac{1}{\tau}\sum_{t=T-\tau}^{T}\varepsilon_{it} \quad\text{(negative = expect reversion)}$$

Projected onto the zero-sum space: $\alpha \leftarrow \alpha - \bar\alpha \mathbf{1}$.

---

### Stage 2 · Vietoris-Rips TDA (`manifold_tda.py`)

**Input:** asset return matrix $X \in \mathbb{R}^{N \times \tau}$ (window of $\tau$ days).

**Step 1 — GPU pairwise correlation distance:**

$$d_{ij} = \sqrt{2(1 - \rho_{ij})}, \quad \rho_{ij} = \frac{\tilde x_i \cdot \tilde x_j}{\|\tilde x_i\|\|\tilde x_j\|}$$

Computed via `cupy` BLAS: $O(N^2)$ in a single matmul, $\sim$40× faster than CPU `scipy.spatial.distance` on $N=500$.

**Step 2 — Vietoris-Rips persistent homology** (via `ripser` on CPU after transfer):

Build a simplicial complex at scale $\varepsilon$. Track birth/death of topological features:
- $H_0$ (connected components, Betti $\beta_0$) — how many market clusters exist
- $H_1$ (loops, Betti $\beta_1$) — circular arbitrage structures / regime cycles

**Step 3 — Topological fracture score:**

$$\mathcal{F}_t = 1.5\,|\Delta\beta_0| + 2.0\,|\Delta\beta_1| + 0.5\,(\Delta\Pi_0 + \Delta\Pi_1)$$

where $\Pi_k = \sum_i (\text{death}_i - \text{birth}_i)$ is total $H_k$ persistence.

**Step 4 — EMA anomaly z-score:**

$$\bar{\mathcal{F}}_t = \alpha \mathcal{F}_t + (1-\alpha)\bar{\mathcal{F}}_{t-1}, \quad z_t = \frac{\mathcal{F}_t - \bar{\mathcal{F}}_t}{\sqrt{\overline{\mathcal{F}^2}_t - \bar{\mathcal{F}}_t^2}}$$

$|z_t| > 2.5$ → topological anomaly flag.

---

### Stage 3 · HJB Kelly Controller (`hjb_controller.py`)

**Objective:** maximize $E[\log W_T]$ under continuous-time dynamics $dW = f\,\mu\,W\,dt + f\,\sigma\,W\,dB_t$.

**HJB equation** reduces to unconstrained Kelly:

$$f^* = \frac{\mu_p - c}{\gamma\,\sigma^2_{\text{eff}}}$$

where $\mu_p = w^\top \bar{r}$ (sample mean return of alpha-weighted portfolio), $c$ = transaction cost in bps, and the TDA-augmented variance:

$$\sigma^2_{\text{eff}} = \sigma^2_p + \underbrace{\gamma_{\text{TDA}}\,\mathcal{F}_n\,\sigma^2_p}_{\text{fracture penalty}} + \underbrace{\kappa\,|z_t|\,\sigma^2_p\,\mathbf{1}[\text{anomaly}]}_{\text{anomaly penalty}} + \underbrace{\tfrac{\gamma_{\text{TDA}}}{2}\,|\Delta\beta_1|\,\sigma^2_p}_{\text{loop penalty}}$$

**Leverage smoothing** (prevents whipsaw):

$$f_t = f_{t-1} + \eta\,(f^* - f_{t-1})\,\Delta t, \quad \eta = \text{leverage\_speed}$$

**Final weights** — variance-weighted blend of eigen-portfolios scaled by $f^*$:

$$w = f^* \cdot \frac{\sum_{k=1}^K \tilde\lambda_k\,\tilde w_k}{\|\sum_k \tilde\lambda_k\,\tilde w_k\|_2}, \quad \tilde\lambda_k = \frac{\lambda_k}{\sum_j \lambda_j}$$

**Expected log growth:**

$$G(f) = f\,\mu_p - \frac{\gamma}{2}\,f^2\,\sigma^2_{\text{eff}}$$

---

## cuML Acceleration Reference

| Operation | CPU (sklearn/numpy) | GPU (cuML/cupy) | Speedup |
|---|---|---|---|
| PCA on 500×252 matrix | ~4,000 ms | ~40 ms | **~100×** |
| Ledoit-Wolf covariance | ~800 ms | ~8 ms | **~100×** |
| Pairwise correlation distance (N=500) | ~1,200 ms | ~30 ms | **~40×** |
| `eigh` on 500×500 matrix | ~200 ms | ~5 ms | **~40×** |
| Rolling TDA (60 windows) | ~90 s | ~4 s (dist GPU) | **~22×** |

Bottleneck is `ripser` (CPU-only). cuML handles everything upstream. The GPU/CPU abstraction in `src/utils/backend.py` allows transparent fallback via `TOPARB_FORCE_CPU=1`.

---

## Project Structure

```
config/settings.py               All configuration + env-variable overrides
src/
  math/eigen_risk.py             PCA risk model, Ledoit-Wolf, Marchenko-Pastur
  math/manifold_tda.py           Vietoris-Rips TDA, fracture signal
  execution/hjb_controller.py    HJB Kelly controller + TDA penalty
  execution/order_router.py      PaperOrderRouter / IBKROrderRouter
  execution/rebalancer.py        Threshold-based rebalance logic
  execution/recovery.py          State persistence across restarts
  risk/risk_manager.py           12-layer safety validation (guardian)
  data/feed.py                   YFinanceFeed / PolygonFeed
  backtest/backtester.py         Walk-forward backtest
  backtest/engine.py             Full BacktestEngine with benchmark
  monitoring/telemetry.py        InfluxDB + structured logging
  orchestrator.py                Live trading main loop
  utils/backend.py               cupy/numpy abstraction
tests/                           80+ tests, safety tests mandatory
main.py                          Entry point
calibrate.py                     Parameter grid search
```

---

## Risk Limits

| Limit | Value | Enforcement |
|---|---|---|
| Absolute max leverage | 4.0× | Hard-coded constant, RiskManager |
| Max single position | 35% NAV | Hard-coded constant |
| Daily loss halt | 5% NAV (configurable) | Circuit breaker → HALTED state |
| Drawdown halt | 15% from peak (configurable) | Circuit breaker → HALTED state |
| Daily turnover | 50% portfolio | Hard-coded constant |
| Order rate limit | 5s minimum interval | RiskManager |

---

## Safety Checklist Before Live Trading

- [ ] `make test-safety` — all pass
- [ ] Backtest Sharpe > 1.0 on target universe
- [ ] `make calibrate` — apply optimal `gamma_tda`, `kappa_anomaly`
- [ ] Paper trade ≥ 30 days
- [ ] InfluxDB/Grafana monitoring confirmed working
- [ ] Reviewed all limits in `config/settings.py`

---

## Key Environment Variables

```bash
TOPARB_FORCE_CPU=1              # Disable GPU (CI / no CUDA)
TOPARB_PAPER_TRADING=true       # Paper trading mode (default: true)
TOPARB_FEED_MODE=yfinance        # Data feed: yfinance | polygon
TOPARB_POLYGON_KEY=<key>         # Polygon.io API key
TOPARB_MAX_LEVERAGE=2.0          # Portfolio leverage cap
TOPARB_DAILY_LOSS=0.05           # Daily loss halt threshold
TOPARB_DRAWDOWN_HALT=0.15        # Drawdown halt threshold
TOPARB_GAMMA_TDA=3.0             # TDA fracture penalty weight
TOPARB_KAPPA_ANOMALY=1.5         # Anomaly z-score multiplier
```
