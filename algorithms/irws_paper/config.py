"""Published algorithm parameters and explicitly identified engineering defaults."""
from dataclasses import dataclass, asdict

MODEL_VERSION = 'irws_paper_cloud_v1'


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
    alpha: float = .5
    # Not specified by the paper: one topological pass, TD(0)=Eq.(31),
    # one update round, GRU width equal to embedding width, repo entropy setting.
    message_passes: int = 1
    gae_lambda: float = 0.
    update_rounds: int = 1
    entropy_coef: float = .01
    rnn_hidden: int = 5
    device: str = 'cpu'

    def __post_init__(self):
        if not (0<=self.gamma<=1 and 0<=self.gae_lambda<=1 and 0<self.clip<1 and 0<=self.alpha<=1):
            raise ValueError('Invalid paper algorithm coefficients')
        if min(self.actor_lr,self.critic_lr,self.intrinsic_lr,self.embedding_lr)<=0:
            raise ValueError('Learning rates must be positive')
        if min(self.batch_size,self.trajectory_length,self.message_passes,self.embedding_dim,self.update_rounds,self.rnn_hidden)<1:
            raise ValueError('Invalid network/training dimensions')

    def as_dict(self):return asdict(self)