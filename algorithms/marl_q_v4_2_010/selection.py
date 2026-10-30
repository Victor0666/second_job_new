"""MARL-only 92% success gate; retain the shared energy ranking and validation."""
from algorithms.pd3qn_dag_fe import selection as _shared

SELECTION_TAG = 'select_sr092_minenergy_v1'
WORKFLOW_SUCCESS_THRESHOLD = .92
selection_key = _shared.selection_key


def selection_metadata():
    return dict(_shared.selection_metadata(), version=SELECTION_TAG,
                workflow_success_threshold=WORKFLOW_SUCCESS_THRESHOLD)


def validation_selection(rows):
    candidate = _shared.validation_selection(rows)
    candidate['feasible'] = candidate['workflow_success_rate'] >= WORKFLOW_SUCCESS_THRESHOLD
    return candidate
