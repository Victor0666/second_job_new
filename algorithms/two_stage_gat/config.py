"""Frozen author architecture/PPO; only the public problem and budget are adapted."""
from dataclasses import asdict, dataclass

SOURCE_REPOSITORY = 'https://github.com/kdsuneraavinash/2sd-gat-cloud-scheduler'
SOURCE_COMMIT = 'aae1397f11e8e75d5dcb70c9b38608dcf05ef773'
MODEL_VERSION = 'two_stage_gat_source_aae1397_selection_v2'
SELECTION_TAG = 'select_sr092_esp099_v2'
WORKFLOW_SUCCESS_THRESHOLD = .92
SATISFACTION_PROBABILITY_TARGET = .99
INPUT_ENCODING = 'task6_vm4_signed_slack_shared_v1'
REWARD_PROTOCOL = 'common_energy_tardiness_v2'
SCHEDULING_PROTOCOL = 'arrived_parent_finished_vm_tail_reservation_v1'
COMMON_REWARD = dict(alpha_delay_vm=.75, energy_reward_scale=.001, task_baseline_norm=300.)


@dataclass(frozen=True)
class TwoStageGATConfig:
    hidden_dim: int = 64
    embedding_dim: int = 8
    gat_layers: int = 3
    heads: int = 4
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
    adam_eps: float = 1e-5
    seed: int = 1
    device: str = 'cpu'

    def __post_init__(self):
        if (self.hidden_dim, self.embedding_dim, self.gat_layers, self.heads) != (64, 8, 3, 4):
            raise ValueError('The published implementation fixes hidden64/embedding8/GAT3/heads4')
        if min(self.num_envs, self.num_steps, self.num_minibatches, self.update_epochs) < 1:
            raise ValueError('Sampling and update counts must be positive')
        if self.batch_size % self.num_minibatches or self.batch_size // self.num_minibatches < 2:
            raise ValueError('Full PPO minibatches must divide the rollout and have at least two rows')
        if not (0 <= self.gamma <= 1 and 0 <= self.gae_lambda <= 1 and 0 < self.clip_coef < 1):
            raise ValueError('Invalid PPO coefficients')
        if min(self.learning_rate, self.adam_eps, self.max_grad_norm) <= 0 or min(self.ent_coef, self.vf_coef) < 0:
            raise ValueError('Invalid optimizer/loss coefficients')

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    def as_dict(self):
        return asdict(self)
