"""Author-code GIN/PPO settings with public scheduling-state inputs."""
from dataclasses import asdict, dataclass


SOURCE_REPOSITORY = "https://github.com/kdsuneraavinash/workflow-cloudsim-drlgnn"
SOURCE_COMMIT = "be67762c3dd22c05733ebb808083375de3bd7d62"
MODEL_VERSION = "drl_gnn_source_be67762_public_state_v2"
REWARD_PROTOCOL = "common_energy_tardiness_v2"
INPUT_ENCODING = "public_raw_task8_vm9_bn_v2"
# Append public state to the native task4/VM3 prefix; keep raw units and graph BN.
TASK_FEATURES = ("assigned", "ready", "workload_mi", "estimated_completion_s",
                 "task_deadline_remaining_s", "workflow_deadline_remaining_s",
                 "input_bits", "output_bits")
VM_FEATURES = ("availability_time_s", "inverse_cpu_speed", "active_energy_rate",
               "bandwidth_mbps", "remaining_busy_s", "host_load",
               "host_current_power_w", "host_idle_power_w", "host_id")
COMMON_REWARD = dict(alpha_delay_vm=.75, energy_reward_scale=.001, task_baseline_norm=300.)


@dataclass(frozen=True)
class DRLGNNConfig:
    hidden_dim: int = 64
    embedding_dim: int = 16
    gin_layers: int = 3
    total_timesteps: int = 2_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = .99
    gae_lambda: float = .95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = .2
    clip_vloss: bool = True
    ent_coef: float = .01
    vf_coef: float = .5
    max_grad_norm: float = .5
    target_kl: float | None = None
    adam_eps: float = 1e-5
    seed: int = 1
    device: str = "cpu"

    def __post_init__(self):
        if (self.hidden_dim, self.embedding_dim, self.gin_layers) != (64, 16, 3):
            raise ValueError("The author-code baseline fixes hidden=64, embedding=16 and GIN layers=3")
        if min(self.total_timesteps, self.num_envs, self.num_steps,
               self.num_minibatches, self.update_epochs) < 1:
            raise ValueError("Training counts must be positive")
        if self.batch_size % self.num_minibatches or self.minibatch_size < 2:
            raise ValueError("PPO minibatches must divide the rollout and contain at least two samples")
        if not (0 <= self.gamma <= 1 and 0 <= self.gae_lambda <= 1 and 0 < self.clip_coef < 1):
            raise ValueError("Invalid PPO coefficients")
        if min(self.learning_rate, self.adam_eps, self.max_grad_norm) <= 0:
            raise ValueError("Learning rate, Adam epsilon and gradient limit must be positive")
        if min(self.ent_coef, self.vf_coef) < 0 or (self.target_kl is not None and self.target_kl <= 0):
            raise ValueError("Invalid PPO loss weights or KL limit")

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self):
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self):
        return self.total_timesteps // self.batch_size

    def as_dict(self):
        return asdict(self)
