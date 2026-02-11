from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ml.src.models.keras_model import KerasModel

ARTIFACTS_ROOT = Path("ml/artifacts")

@lru_cache(maxsize=8)
def _load_model(model_id: str) -> KerasModel:
    run_dir = ARTIFACTS_ROOT / model_id
    cfg_path = run_dir / "config_used.yaml"
    weights_path = run_dir / "model.weights.h5"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    model = KerasModel(config_path=str(cfg_path), name=model_id)
    model.model.load_weights(str(weights_path))
    return model

def predict_h61(*, model_id: str, x_window: list[list[float]]) -> list[float]:
    """
    x_window: shape (T, input_dim)
    returns: list length 61
    """
    model = _load_model(model_id)

    x = np.asarray(x_window, dtype=np.float32)          # (T, D)
    x = x[None, ...]                                    # (1, T, D)

    y = model.model.predict(x, verbose=0)[0]            # (61,)
    return [float(v) for v in y]