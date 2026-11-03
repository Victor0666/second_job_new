"""Evaluation-only, fixed-count version of the low-high-low Poisson workload."""
import numpy as np


PROFILE = 'bursty_p20_r5'
MEAN_RATE = .03
LOW_RATE = 1. / 60.
HIGH_RATE = 1. / 12.


def workload_metadata(count=50):
    horizon = count / MEAN_RATE
    return dict(profile=PROFILE, arrival_lambda=MEAN_RATE, reference_arrival_lambda=MEAN_RATE,
                arrival_source='piecewise_poisson_conditioned_on_count', workflows_per_episode=count,
                arrival_observation_horizon=horizon, peak_start=.4 * horizon, peak_end=.6 * horizon,
                low_arrival_lambda=LOW_RATE, high_arrival_lambda=HIGH_RATE, peak_to_low_ratio=5.,
                peak_time_fraction=.2, expected_peak_arrival_fraction=5. / 9.,
                source_workload_profile='standard', training=False, adaptation=False)


def apply_arrival_profile(arrivals, profile, seed, count, arrival_lambda):
    if profile == 'standard':
        return arrivals
    if profile != PROFILE:
        raise ValueError('Unknown arrival profile: ' + str(profile))
    if (isinstance(count, bool) or count is None or int(count) != count or count < 1
            or len(arrivals) != count or arrival_lambda != MEAN_RATE):
        raise ValueError('Bursty evaluation requires a fixed positive workflow count and mean rate 0.03')
    horizon = count / MEAN_RATE
    first, second = .4 * horizon, .6 * horizon
    mass_first = LOW_RATE * first
    mass_second = mass_first + HIGH_RATE * (second - first)
    # Conditional Poisson event times are ordered uniforms in cumulative-intensity space.
    mass = np.sort(np.random.default_rng(seed).uniform(0., MEAN_RATE * horizon, size=int(count)))
    times = np.where(mass < mass_first, mass / LOW_RATE,
                     np.where(mass < mass_second, first + (mass - mass_first) / HIGH_RATE,
                              second + (mass - mass_second) / LOW_RATE))
    return times.tolist()
