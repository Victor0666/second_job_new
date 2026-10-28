"""Reuse exactly the ablation SR>092 selector and evidence checks."""
from common.mahdrl_ablation import (
    SELECTION_TAG, WORKFLOW_SUCCESS_THRESHOLD, selection_metadata,
    validation_selection, selection_key,
    validate_checkpoint_selection as base_validate_checkpoint_selection,
)


def validate_checkpoint_selection(record, expected):
    if not isinstance(record, dict):
        raise ValueError('Missing MAHDRL-AS-Fast-R010-NoPF checkpoint selection/reward metadata')
    return base_validate_checkpoint_selection(record, expected)
