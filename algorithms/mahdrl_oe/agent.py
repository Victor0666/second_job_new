"""Baseline D3QN with explicit MAHDRL-OE checkpoint compatibility checks."""

from copy import deepcopy
import json
from pathlib import Path

import torch

from common.d3qn_agent_EXDehid import D3QNAgent as _BaseD3QNAgent


VARIANT_SCHEME = "mahdrl_oe_1235_v1"


def bind_variant(agent, metadata, role):
    """Bind the actual environment schema, including the unchanged TA's identity."""
    if (metadata.get("scheme") != VARIANT_SCHEME
            or metadata.get("changes") != [1, 2, 3, 5]
            or metadata.get("deadline_action_filter") is not False):
        raise ValueError("Expected the MAHDRL-OE changes 1/2/3/5 environment")
    if role not in ("TA", "HA", "VA"):
        raise ValueError("Unknown MAHDRL-OE agent role")
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
        json.dumps(dict(algorithm="mahdrl_oe", **metadata), indent=2, allow_nan=False),
        encoding="utf-8",
    )


class D3QNAgent(_BaseD3QNAgent):
    """Same network and learning rules; foreign or unmarked weights are rejected."""

    def _expected_variant(self):
        expected = getattr(self, "variant_metadata", None)
        if expected is None:
            raise ValueError("Bind MAHDRL-OE environment metadata before saving or loading")
        if self.normalization_metadata.get("algorithm_variant") != expected:
            raise ValueError("MAHDRL-OE checkpoint metadata was overwritten")
        return expected

    def save(self, path):
        self._expected_variant()
        super().save(path)

    def load(self, path, strict=True, require_learning_model=False):
        expected = self._expected_variant()
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        normalization = checkpoint.get("normalization", {}) if isinstance(checkpoint, dict) else {}
        actual = normalization.get("algorithm_variant") if isinstance(normalization, dict) else None
        if actual != expected:
            raise ValueError(
                f"Checkpoint MAHDRL-OE variant/role mismatch: {path}; "
                "train MAHDRL-OE or select its matching checkpoint directory"
            )
        del checkpoint
        super().load(path, strict=strict, require_learning_model=require_learning_model)
