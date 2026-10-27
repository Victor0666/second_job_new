"""PD3QN-FE-R090: delay weight 0.9 and aggregate-success checkpoint selection."""

ALGORITHM = 'pd3qn_fe_r090'
DISPLAY_NAME = 'PD3QN-FE-R090'
MODEL_VERSION = 'pd3qn_fe_r090_v1'
DELAY_WEIGHT = .9


def reward_metadata():
    return dict(delay_weight=DELAY_WEIGHT, energy_weight=1. - DELAY_WEIGHT,
                task_baseline_norm=300., energy_reward_scale=.001)
