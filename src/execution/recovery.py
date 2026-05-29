"""
src/execution/recovery.py
"""
import pickle
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class RecoveryManager:
    """Гарантирует сохранение топологического и HJB состояния между рестартами."""

    def __init__(self, filepath: str = "data/state_dump.pkl"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def save_state(self, orchestrator) -> None:
        if not orchestrator._controller:
            return

        state_dict = {
            "step": orchestrator._step,
            "nav": orchestrator._nav,
            "current_weights": orchestrator._current_weights,
            "controller_state": orchestrator._controller.serialize_state(),
            "tda_beta0": orchestrator._controller._tda.beta0_history,
            "tda_beta1": orchestrator._controller._tda.beta1_history,
            "tda_fracture": orchestrator._controller._tda.fracture_history,
        }
        try:
            with open(self.filepath, "wb") as f:
                pickle.dump(state_dict, f)
            logger.debug("State saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self) -> Dict[str, Any] | None:
        if not os.path.exists(self.filepath):
            return None
        try:
            with open(self.filepath, "rb") as f:
                state = pickle.load(f)
            logger.info("Previous state successfully recovered.")
            return state
        except Exception as e:
            logger.critical(f"Failed to load state dump: {e}")
            return None