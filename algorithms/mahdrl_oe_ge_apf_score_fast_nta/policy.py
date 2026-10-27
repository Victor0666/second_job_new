"""An episode-local random stream, independent of environment and learner RNGs."""
import numpy as np
from .config import ta_policy_metadata


def manager_seed(instance_seed):
    return (int(instance_seed) * 1000003 + 20260325 + 9176) & 0xFFFFFFFF


class RandomWeightPolicy:
    def __init__(self, instance_seed):
        self.instance_seed = int(instance_seed)
        self.seed = manager_seed(self.instance_seed)
        self.rng = np.random.RandomState(self.seed)
        self.decisions = 0
        self.legal_count = 0

    def select_action(self, mask):
        mask = np.asarray(mask)
        if (mask.shape != (243,) or not np.isfinite(mask).all()
                or np.any((mask != 0) & (mask != 1))):
            raise ValueError("Random TA requires a finite binary 243-action mask")
        valid = np.flatnonzero(mask > .5)
        if not valid.size:
            raise ValueError("Random TA has no legal weight increment")
        self.legal_count = int(valid.size)
        action = int(self.rng.choice(valid))
        self.decisions += 1
        return action

    def summary(self):
        return dict(policy=ta_policy_metadata()["version"], instance_seed=self.instance_seed,
                    seed=self.seed, decisions=self.decisions)
