"""F17: normalized TA state and shared candidate validation."""
import numpy as np

TA_WEIGHT_MODEL = 'f17_normalized_weights_v1'
TA_ALGORITHMS = frozenset(('mahdrl', 'mahdrl_oe', 'ablation_hostrand', 'ablation_mgrrand',
                           'ablation_vmrand', 'ablation_nohost', 'ablation_tl2'))
WEIGHT_TOLERANCE = 1e-8


def normalize_initial_weights(weights):
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (5,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('TA initial weights must be five finite nonnegative values')
    scale = float(values.max())
    if scale <= 0:
        raise ValueError('TA initial weights must have positive total mass')
    scaled = values / scale
    return (scaled / scaled.sum()).astype(np.float32)


def weight_candidates(weights, deltas):
    """One rule for masking and execution; only roundoff-sized negatives are repaired."""
    w = np.asarray(weights, dtype=np.float64)
    d = np.asarray(deltas, dtype=np.float64)
    if (w.shape != (5,) or not np.isfinite(w).all() or np.any(w < 0)
            or not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-7)):
        raise ValueError('TA persistent weights must lie on the normalized nonnegative simplex')
    if d.ndim != 2 or d.shape[1] != 5:
        raise ValueError('TA increments must have five dimensions')
    finite = np.isfinite(d).all(axis=1)
    # Also reject direct calls outside the unchanged {-0.1, 0, +0.1} action space.
    action_values = np.isclose(d[..., None], [-.1, 0., .1], rtol=0, atol=WEIGHT_TOLERANCE).any(axis=-1).all(axis=1)
    u = w[None, :] + d
    valid = finite & action_values & np.isfinite(u).all(axis=1) & (u >= -WEIGHT_TOLERANCE).all(axis=1)
    cleaned = np.maximum(u, 0.0)
    total = cleaned.sum(axis=1)
    valid &= np.isfinite(total) & (total > 0)
    normalized = np.zeros_like(u)
    np.divide(cleaned, total[:, None], out=normalized, where=valid[:, None])
    normalized = normalized.astype(np.float32)
    valid &= np.isfinite(normalized).all(axis=1) & (normalized >= 0).all(axis=1)
    valid &= np.isclose(normalized.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=1e-7)
    # Preserve the exact existing representation for the zero-increment action.
    normalized[(d == 0).all(axis=1)] = w
    return normalized, valid
