"""OE + guided exploration, with a separately identified optional PF variant."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from algorithms.mahdrl_oe.agent import D3QNAgent as OEAgent
from common.mahdrl_performance import (
    FILTER_POLICY, GUIDED_POLICY, GuidedExplorationAgent, PerformancePolicy,
)
from environments.mahdrl_oe import VARIANT_SCHEME as OE_SCHEME


OE_GE_POLICY = PerformancePolicy(
    "mahdrl_oe_ge_v1", guided_probability=GUIDED_POLICY.guided_probability,
)
OE_GE_PF_POLICY = PerformancePolicy(
    "mahdrl_oe_ge_pf_v1", guided_probability=GUIDED_POLICY.guided_probability,
    max_duration_ratio=FILTER_POLICY.max_duration_ratio,
)


class OEGuidedAgent(GuidedExplorationAgent, OEAgent):
    """GE action selection, original D3QN updates, and both checkpoint checks."""
    policy = OE_GE_POLICY


class OEGuidedFilterAgent(OEGuidedAgent):
    policy = OE_GE_PF_POLICY


def bind_variant(agent, metadata, role):
    policy = asdict(agent.policy)
    filtered = agent.policy == OE_GE_PF_POLICY
    components = ["OE", "GE", "PF"] if filtered else ["OE", "GE"]
    if (agent.policy not in (OE_GE_POLICY, OE_GE_PF_POLICY)
            or metadata.get("scheme") != agent.policy.scheme
            or metadata.get("algorithm") != agent.policy.scheme.removesuffix("_v1")
            or metadata.get("oe_base_scheme") != OE_SCHEME
            or metadata.get("changes") != [1, 2, 3, 5]
            or metadata.get("deadline_action_filter") is not False
            or metadata.get("components") != components
            or metadata.get("performance_action_filter") is not filtered
            or metadata.get("host_summary_scope") != ("performance_candidates" if filtered else "all_legal")
            or metadata.get("performance_policy") != policy):
        raise ValueError("Expected the matching MAHDRL OE/GE combination environment")
    if role not in ("TA", "HA", "VA"):
        raise ValueError("Unknown MAHDRL combination agent role")
    agent.variant_metadata = dict(deepcopy(metadata), role=role)
    agent.normalization_metadata = dict(
        getattr(agent, "normalization_metadata", {"normalization_version": "legacy"}),
        algorithm_variant=deepcopy(agent.variant_metadata),
    )
    return agent


def write_variant_metadata(directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "algorithm_variant.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8",
    )
