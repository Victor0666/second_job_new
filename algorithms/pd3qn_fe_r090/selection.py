"""Reuse the pure single-SR selector; the learner remains historical PD3QN-FE."""
from algorithms.pd3qn_dag_fe.selection import (
    SELECTION_TAG, WORKFLOW_SUCCESS_THRESHOLD,
    selection_key, selection_metadata, validation_selection,
)
