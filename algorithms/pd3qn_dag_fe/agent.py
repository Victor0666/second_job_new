"""Keep the shared PD3QN learner; simplify only this variant's dueling heads."""
import torch
from torch import nn

from common.d3qn_agent_PER import D3QNAgent


class DAGFEAgent(D3QNAgent):
    def __init__(self, *args, **kwargs):
        if not kwargs.get('paper_mode'):
            raise ValueError('PD3QN-DAG-FE requires the paper learner')
        super().__init__(*args, **kwargs)
        width = self.hidden_dims[-1]
        # Keep network objects so their existing diagnostic hooks remain attached.
        for network in (self.online, self.target):
            network.val = nn.Linear(width, 1).to(self.device)
            network.adv = nn.Linear(width, self.output_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        # The optimizer must reference the new heads rather than the discarded ones.
        self.optim = torch.optim.Adam(self.online.parameters(), lr=self.optim.param_groups[0]['lr'])
