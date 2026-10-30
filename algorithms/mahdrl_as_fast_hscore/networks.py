"""A fixed-output Dueling network with a legal-action advantage mean."""
from common.d3qn_agent_EXDehid import DuelingQNet
from .config import VM_INPUT_DIM, VM_CAPACITY, TRUNK


class FixedVMDuelingQNet(DuelingQNet):
    def __init__(self, hidden_dims=TRUNK):
        super().__init__(VM_INPUT_DIM, VM_CAPACITY, hidden_dims)

    def forward(self, x, mask):
        if x.ndim != 2 or x.shape[1] != VM_INPUT_DIM or mask.shape != (x.shape[0], VM_CAPACITY):
            raise ValueError("Fixed VA network shape mismatch")
        h = self.feature(x)
        advantage = self.adv(h)
        legal = (mask > .5).to(advantage.dtype)
        center = (advantage * legal).sum(1, keepdim=True) / legal.sum(1, keepdim=True).clamp_min(1)
        return self.val(h) + advantage - center
