"""Use exactly the same dual-99% rule as PD3QN-FE without changing its code."""
from algorithms.pd3qn_fe.selection import (
    SATISFACTION_PROBABILITY_TARGET, SELECTION_TAG, WORKFLOW_SUCCESS_THRESHOLD,
    selection_key, selection_metadata, validation_selection,
)
