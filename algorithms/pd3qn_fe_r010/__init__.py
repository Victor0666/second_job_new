"""PD3QN-FE-R010: delay weight 0.1 and aggregate-success checkpoint selection."""

ALGORITHM = 'pd3qn_fe_r010'
DISPLAY_NAME = 'PD3QN-FE-R010'
MODEL_VERSION = 'pd3qn_fe_r010_v1'
DELAY_WEIGHT = .1


def reward_metadata():
    return dict(delay_weight=DELAY_WEIGHT, energy_weight=1. - DELAY_WEIGHT,
                task_baseline_norm=300., energy_reward_scale=.001)
