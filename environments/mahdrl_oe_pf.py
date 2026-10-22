"""OE assignment rewards and PF-consistent summaries; the policy has no GE."""
from algorithms.mahdrl_oe_pf.agent import OE_PF_POLICY
from environments.mahdrl_oe_ge_pf import CloudWorkflowEnv_VMAgents as _FilteredOE
from environments.mahdrl_oe_ge_pf import MANAGER_ACTION_TABLE


class CloudWorkflowEnv_VMAgents(_FilteredOE):
    # GE lives in the agent/trainer, so reusing the filtered environment adds no guidance.
    algorithm = "mahdrl_oe_pf"
    policy = OE_PF_POLICY
    components = ("OE", "PF")
