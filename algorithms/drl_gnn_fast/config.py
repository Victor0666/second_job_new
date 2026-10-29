"""Unchanged DRL-GNN scientific settings; separate execution and selection identity."""
from algorithms.drl_gnn.config import (DRLGNNConfig, COMMON_REWARD, INPUT_ENCODING,
    POWER_REFERENCE_W, REWARD_PROTOCOL, SCHEDULING_PROTOCOL, SOURCE_COMMIT,
    SOURCE_REPOSITORY, TASK_FEATURES, VM_FEATURES)

ALGORITHM = 'drl_gnn_fast'
DISPLAY_NAME = 'DRL-GNN-Fast'
MODEL_VERSION = 'drl_gnn_fast_v5'
GRAPHS_PER_BACKWARD = 4
PPO_UPDATE_EXECUTION = 'independent_graph_groups_v1'
