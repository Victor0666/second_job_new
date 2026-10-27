"""Descriptive stability statistics from raw fixed-validation learning curves."""
import numpy as np


def training_stability(rows, *, threshold=.99, consecutive=5, tail_window=100):
    if consecutive < 1 or tail_window < 1 or not 0 <= threshold <= 1 or not rows:
        raise ValueError("Invalid stability window, threshold or empty learning curve")
    energy = np.asarray([row["validation_energy"] for row in rows], dtype=np.float64)
    success = np.asarray([row["validation_success_rate"] for row in rows], dtype=np.float64)
    if not np.isfinite(energy).all() or not np.isfinite(success).all() or np.any(energy < 0) or np.any((success < 0) | (success > 1)):
        raise ValueError("Invalid validation metrics in learning curve")
    feasible = success >= threshold
    streak, reached = 0, None
    for i, value in enumerate(feasible):
        streak = streak+1 if value else 0
        if reached is None and streak >= consecutive:
            reached = i
    tail_e, tail_s = energy[-tail_window:], success[-tail_window:]
    return dict(window=tail_window, observations=len(rows), success_threshold=threshold,
        consecutive_validations=consecutive,
        first_sustained_feasible_completed_episode=(int(rows[reached]["episode"])+1 if reached is not None else None),
        post_sustained_feasibility_drop_count=(int(np.sum(feasible[reached:-1] & ~feasible[reached+1:])) if reached is not None else 0),
        tail_validation_energy_mean=float(tail_e.mean()), tail_validation_energy_std=float(tail_e.std()),
        tail_validation_success_mean=float(tail_s.mean()), tail_validation_success_std=float(tail_s.std()),
        tail_mean_absolute_energy_change=float(np.abs(np.diff(tail_e)).mean()) if len(tail_e)>1 else 0.,
        statistics="raw_curve_within_one_training_seed;_not_a_cross_seed_confidence_interval")
