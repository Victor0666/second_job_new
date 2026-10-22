"""OE observations and assignment rewards; GE changes only the training policy."""
from dataclasses import asdict

from common.mahdrl_combinations import OE_GE_POLICY
from environments.mahdrl_oe import CloudWorkflowEnv_VMAgents as _OE
from environments.mahdrl_oe import MANAGER_ACTION_TABLE


class CloudWorkflowEnv_VMAgents(_OE):
    algorithm = "mahdrl_oe_ge"
    policy = OE_GE_POLICY
    components = ("OE", "GE")

    def variant_metadata(self):
        metadata = super().variant_metadata()
        return dict(metadata, algorithm=self.algorithm, oe_base_scheme=metadata["scheme"],
                    scheme=self.policy.scheme, components=list(self.components),
                    performance_policy=asdict(self.policy),
                    performance_action_filter="PF" in self.components,
                    host_summary_scope="performance_candidates" if "PF" in self.components else "all_legal")
