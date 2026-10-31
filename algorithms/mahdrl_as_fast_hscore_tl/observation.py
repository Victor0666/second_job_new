"""Validate complete host batches; each network row retains its nine slots."""
import numpy as np
from .config import VM_INPUT_DIM, VM_CAPACITY


def validate_grouped(obs, mask, *, decision=True):
    obs, mask = np.asarray(obs), np.asarray(mask)
    if (obs.ndim != 2 or obs.shape[1] != VM_INPUT_DIM
            or mask.shape != (len(obs)*VM_CAPACITY,)):
        raise ValueError("Grouped VA requires Hx93 observations and a flat 9H mask")
    if (not np.isfinite(obs).all() or not np.isfinite(mask).all()
            or not np.isin(mask, (0, 1)).all()):
        raise ValueError("Non-finite grouped observation or non-binary mask")
    if decision and not np.any(mask > .5):
        raise ValueError("No legal global VM at a non-terminal decision")
