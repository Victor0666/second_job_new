"""Uniform host sampling uses one independent stream per physical instance."""
import numpy as np
from .config import ha_policy_metadata


def host_seed(instance_seed):
    return (int(instance_seed) * 1000003 + 20260325 + 123456 * 97) & 0xFFFFFFFF


class RandomHostPolicy:
    def __init__(self, instance_seed):
        self.instance_seed = int(instance_seed)
        self.seed = host_seed(instance_seed)
        self.rng = np.random.RandomState(self.seed)
        self.decisions = self.multiple_choices = self.last_legal_count = 0

    def select_action(self, mask):
        mask = np.asarray(mask)
        if (mask.ndim != 1 or not np.isfinite(mask).all()
                or np.any((mask != 0) & (mask != 1))):
            raise ValueError("Random HA requires a finite binary host mask")
        valid = np.flatnonzero(mask > .5)
        if not valid.size:
            raise ValueError("Random HA has no legal idle host")
        self.last_legal_count = int(valid.size)
        action = int(self.rng.choice(valid))
        self.decisions += 1
        self.multiple_choices += int(valid.size > 1)
        return action

    def summary(self):
        return dict(policy=ha_policy_metadata()["version"], instance_seed=self.instance_seed,
            seed=self.seed, decisions=self.decisions, multiple_choice_decisions=self.multiple_choices,
            multiple_choice_ratio=self.multiple_choices/max(1, self.decisions))
