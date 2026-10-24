"""IRWS learning parameters and the fixed common cloud reward settings."""
from dataclasses import dataclass, asdict

ALGORITHM = 'irws_r'
DISPLAY_NAME = 'IRWS-R'
MODEL_VERSION = 'irws_r_v2_batch'
IMPLEMENTATION = dict(
    graph_aggregation='predecessor_mean_feature_softmax_v2',
    update_rule='interval_minibatch_mean_v2',
    meta_estimator='observed_interval_mean_local_intrinsic_partials_v2',
    boundary_rule='old_policy_expectation_then_post_update_selection_v2',
    task_score='fixed_seeded_Q',
)
BEST_CHECKPOINT = 'best_irws_r.pth'
FINAL_CHECKPOINT = 'final_irws_r.pth'
REWARD_PROTOCOL = 'common_energy_tardiness_paper_fusion_v1'
COMMON_REWARD = dict(alpha_delay_vm=.75, energy_reward_scale=.001, task_baseline_norm=300.)


@dataclass(frozen=True)
class PaperConfig:
    # Table III, printed p.907; clip from Eq.(23) discussion, printed p.905.
    gamma: float = .2
    clip: float = .2
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4
    intrinsic_lr: float = 1e-5
    embedding_lr: float = 1e-6
    embedding_dim: int = 5
    trajectory_length: int = 16
    batch_size: int = 64
    actor_hidden: tuple = (128,64)
    critic_hidden: tuple = (64,64,32)
    # Not specified by the paper: one topological pass, TD(0)=Eq.(31),
    # one update round and GRU width equal to embedding width.
    message_passes: int = 1
    update_rounds: int = 1
    rnn_hidden: int = 5
    device: str = 'cpu'

    def __post_init__(self):
        if not (0<=self.gamma<=1 and 0<self.clip<1):
            raise ValueError('Invalid paper algorithm coefficients')
        if min(self.actor_lr,self.critic_lr,self.intrinsic_lr,self.embedding_lr)<=0:
            raise ValueError('Learning rates must be positive')
        if min(self.batch_size,self.trajectory_length,self.message_passes,self.embedding_dim,self.update_rounds,self.rnn_hidden)<1:
            raise ValueError('Invalid network/training dimensions')

    def as_dict(self):return asdict(self)
