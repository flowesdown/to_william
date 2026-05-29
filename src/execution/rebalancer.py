"""
Threshold-based portfolio rebalancer.

Rebalancing philosophy:
- Don't rebalance on every tick — transaction costs erode alpha
- Rebalance when drift from target exceeds threshold OR TDA anomaly fires
- Emergency rebalance (deleveraging) when risk limits approach
- Compute optimal threshold dynamically from HJB model
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RebalanceDecision:
    should_rebalance: bool
    reason: str
    urgency: str   # "routine" | "drift" | "emergency"
    drift_magnitude: float
    estimated_turnover: float


class ThresholdRebalancer:
    """
    Decides WHEN to rebalance based on drift + TDA signals.

    The optimal threshold is computed from the HJB controller
    via compute_optimal_rebalance_threshold(), which balances
    transaction cost against tracking error.
    """

    def __init__(
        self,
        base_threshold: float = 0.02,    # 2% drift → rebalance
        tda_anomaly_override: bool = True, # Force rebalance on TDA anomaly
        emergency_leverage_frac: float = 0.90,  # Rebalance if leverage > 90% of limit
        min_rebalance_interval_steps: int = 5,  # Min steps between rebalances
    ) -> None:
        self._base_threshold = base_threshold
        self._tda_override = tda_anomaly_override
        self._emergency_frac = emergency_leverage_frac
        self._min_interval = min_rebalance_interval_steps
        self._steps_since_rebalance: int = min_rebalance_interval_steps

    def should_rebalance(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        current_f_star: float,
        max_leverage: float,
        tda_is_anomaly: bool = False,
        dynamic_threshold: float | None = None,
    ) -> RebalanceDecision:
        """
        Determine whether a rebalance should occur.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target weights from HJB controller
            current_f_star: Current leverage target
            max_leverage: Maximum allowed leverage
            tda_is_anomaly: Whether TDA fracture anomaly is active
            dynamic_threshold: Optional threshold from HJB model
        """
        threshold = dynamic_threshold if dynamic_threshold is not None else self._base_threshold
        drift = float(np.sum(np.abs(target_weights - current_weights)))
        current_leverage = float(np.sum(np.abs(current_weights)))

        # 1. Emergency: leverage approaching hard limit → deleverage now
        if current_leverage > max_leverage * self._emergency_frac:
            return RebalanceDecision(
                should_rebalance=True,
                reason=f"Emergency: leverage {current_leverage:.3f} > "
                       f"{max_leverage * self._emergency_frac:.3f} threshold",
                urgency="emergency",
                drift_magnitude=drift,
                estimated_turnover=drift,
            )

        # 2. TDA anomaly override — rebalance regardless of drift
        if self._tda_override and tda_is_anomaly:
            if self._steps_since_rebalance >= 2:  # Min 2 steps even on anomaly
                self._steps_since_rebalance = 0
                return RebalanceDecision(
                    should_rebalance=True,
                    reason="TDA topological anomaly detected → forced rebalance",
                    urgency="drift",
                    drift_magnitude=drift,
                    estimated_turnover=drift,
                )

        # 3. Minimum interval not met → skip
        if self._steps_since_rebalance < self._min_interval:
            self._steps_since_rebalance += 1
            return RebalanceDecision(
                should_rebalance=False,
                reason=f"Minimum interval: {self._steps_since_rebalance}/{self._min_interval} steps",
                urgency="routine",
                drift_magnitude=drift,
                estimated_turnover=drift,
            )

        # 4. Drift threshold
        if drift >= threshold:
            self._steps_since_rebalance = 0
            return RebalanceDecision(
                should_rebalance=True,
                reason=f"Drift {drift:.4f} >= threshold {threshold:.4f}",
                urgency="routine",
                drift_magnitude=drift,
                estimated_turnover=drift,
            )

        self._steps_since_rebalance += 1
        return RebalanceDecision(
            should_rebalance=False,
            reason=f"Drift {drift:.4f} < threshold {threshold:.4f}",
            urgency="routine",
            drift_magnitude=drift,
            estimated_turnover=drift,
        )

    def reset(self) -> None:
        self._steps_since_rebalance = self._min_interval