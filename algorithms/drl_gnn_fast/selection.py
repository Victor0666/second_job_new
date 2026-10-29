"""Keep the independent 99% workflow-success gate and minimum-energy ranking."""
from algorithms.pd3qn_dag_fe.selection import (
    SELECTION_TAG, WORKFLOW_SUCCESS_THRESHOLD, selection_metadata,
    validation_selection, selection_key)
from algorithms.mahdrl_oe_ge_apf_score_fast.selection import (
    validate_checkpoint_selection as _validate_checkpoint_selection)


def validate_checkpoint_selection(record, expected):
    try:
        return _validate_checkpoint_selection(record, expected, validation_selection)
    except ValueError as error:
        raise ValueError(str(error).replace('MAHDRL-AS-Fast v2', 'DRL-GNN-Fast v5')
                         .replace('MAHDRL-AS-Fast', 'DRL-GNN-Fast')
                         .replace('use v1 for legacy weights', 'use drl_gnn for legacy weights')) from error
