"""Only scheduling and checkpoint selection differ from Fast v5."""
from algorithms.drl_gnn_fast.config import (
    COMMON_REWARD, DRLGNNConfig, INPUT_ENCODING, POWER_REFERENCE_W,
    REWARD_PROTOCOL, SOURCE_COMMIT, SOURCE_REPOSITORY, TASK_FEATURES,
    VM_FEATURES, GRAPHS_PER_BACKWARD, PPO_UPDATE_EXECUTION,
)

ALGORITHM = 'drl_gnn_fast_idle'
DISPLAY_NAME = 'DRL-GNN-Fast-Idle'
MODEL_VERSION = 'drl_gnn_fast_idle_v6'
OUTPUT_TAG = 'drl_idle'
SCHEDULING_PROTOCOL = 'arrived_parent_finished_idle_vm_v1'
