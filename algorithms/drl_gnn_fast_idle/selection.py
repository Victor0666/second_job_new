"""Use the same aggregate SR >= .92 and minimum-energy rule as VA-aware."""
from algorithms.mahdrl_oe_ge_apf_score_fast.selection import (
    SELECTION_TAG, WORKFLOW_SUCCESS_THRESHOLD, selection_metadata,
    validation_selection, selection_key,
    validate_checkpoint_selection as _validate_checkpoint_selection,
)


def validate_checkpoint_selection(record, expected):
    try:
        return _validate_checkpoint_selection(record, expected, validation_selection)
    except ValueError as error:
        raise ValueError(str(error).replace('MAHDRL-AS-Fast', 'DRL-GNN-Fast-Idle')) from error
