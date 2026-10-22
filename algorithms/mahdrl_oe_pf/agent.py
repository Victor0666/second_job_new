"""Original epsilon-greedy and D3QN updates, with OE/PF checkpoint isolation."""
from copy import deepcopy
from dataclasses import asdict

from algorithms.mahdrl_oe.agent import D3QNAgent as OEAgent
from common.mahdrl_combinations import write_variant_metadata
from common.mahdrl_performance import FILTER_POLICY, PerformanceFilterAgent, PerformancePolicy
from environments.mahdrl_oe import VARIANT_SCHEME as OE_SCHEME


OE_PF_POLICY = PerformancePolicy(
    "mahdrl_oe_pf_v1", max_duration_ratio=FILTER_POLICY.max_duration_ratio,
)


class D3QNAgent(PerformanceFilterAgent, OEAgent):
    # No GuidedExplorationAgent in the MRO: even training uses original uniform exploration.
    policy = OE_PF_POLICY


def bind_variant(agent, metadata, role):
    if (agent.policy != OE_PF_POLICY
            or metadata.get("scheme") != OE_PF_POLICY.scheme
            or metadata.get("algorithm") != "mahdrl_oe_pf"
            or metadata.get("oe_base_scheme") != OE_SCHEME
            or metadata.get("changes") != [1, 2, 3, 5]
            or metadata.get("deadline_action_filter") is not False
            or metadata.get("components") != ["OE", "PF"]
            or metadata.get("performance_action_filter") is not True
            or metadata.get("host_summary_scope") != "performance_candidates"
            or metadata.get("performance_policy") != asdict(agent.policy)):
        raise ValueError("Expected the matching MAHDRL-OE-PF environment without GE")
    if role not in ("TA", "HA", "VA"):
        raise ValueError("Unknown MAHDRL-OE-PF agent role")
    agent.variant_metadata = dict(deepcopy(metadata), role=role)
    agent.normalization_metadata = dict(
        getattr(agent, "normalization_metadata", {"normalization_version": "legacy"}),
        algorithm_variant=deepcopy(agent.variant_metadata),
    )
    return agent
