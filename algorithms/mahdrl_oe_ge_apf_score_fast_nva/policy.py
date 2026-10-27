"""Local VM slots are sampled uniformly using an independent per-instance RNG."""
import numpy as np
from .config import vm_policy_metadata


def vm_seed(instance_seed):
    return (int(instance_seed) * 1000003 + 20260325) & 0xFFFFFFFF


class RandomVMPolicy:
    def __init__(self, instance_seed):
        self.instance_seed = int(instance_seed)
        self.seed = vm_seed(instance_seed)
        self.rng = np.random.RandomState(self.seed)
        self.decisions = self.multiple_choices = self.last_legal_count = 0

    def select_action(self, mask):
        mask = np.asarray(mask)
        if (mask.ndim != 1 or not np.isfinite(mask).all()
                or np.any((mask != 0) & (mask != 1))):
            raise ValueError("Random VA requires a finite binary VM mask")
        valid = np.flatnonzero(mask > .5)
        if not valid.size:
            raise ValueError("Random VA has no legal idle VM in the selected host")
        self.last_legal_count = int(valid.size)
        action = int(self.rng.choice(valid))
        self.decisions += 1
        self.multiple_choices += int(valid.size > 1)
        return action

    def summary(self):
        return dict(policy=vm_policy_metadata()["version"], instance_seed=self.instance_seed,
            seed=self.seed, decisions=self.decisions, multiple_choice_decisions=self.multiple_choices,
            multiple_choice_ratio=self.multiple_choices/max(1, self.decisions))
