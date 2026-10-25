"""Two distinct empirical thresholds; strict comparison preserves earlier ties."""
import math

SELECTION_TAG = 'select_sr099_esp099_v1'
WORKFLOW_SUCCESS_THRESHOLD = .99
SATISFACTION_PROBABILITY_TARGET = .99


def selection_metadata():
    return dict(version=SELECTION_TAG, rule='validation_feasible_then_minimum_energy',
        workflow_success_threshold=WORKFLOW_SUCCESS_THRESHOLD,
        satisfaction_probability_target=SATISFACTION_PROBABILITY_TARGET,
        feasibility_metric='validation_empirical_satisfaction_probability',
        energy_metric='mean_validation_instance_total_energy',
        success_metric='total_successful_workflows_divided_by_total_workflows',
        fallback='maximum_ESP_then_workflow_success_then_minimum_energy',
        tie_break='higher_workflow_success_then_earlier_checkpoint',
        population_probability_guaranteed=False)


def validation_selection(rows):
    if not rows:
        raise ValueError('Checkpoint selection requires validation instances')
    successes = total = passed = 0
    energies = []
    for row in rows:
        count, success = row['wf_count'], row['wf_success_count']
        energy = float(row['total_energy'])
        if (not math.isfinite(count) or not math.isfinite(success)
                or count <= 0 or int(count) != count or int(success) != success
                or not 0 <= success <= count or not math.isfinite(energy) or energy < 0):
            raise ValueError('Invalid validation workflow counts or energy')
        rate = success / count
        if not math.isclose(float(row['wf_success_rate']), rate, rel_tol=0., abs_tol=1e-12):
            raise ValueError('Validation workflow success rate/count mismatch')
        successes += int(success)
        total += int(count)
        passed += rate >= WORKFLOW_SUCCESS_THRESHOLD
        energies.append(energy)
    esp = passed / len(rows)
    return dict(workflow_success_rate=successes / total, workflow_success_count=successes,
        workflow_count=total, mean_total_energy=sum(energies) / len(rows),
        validation_ESP=esp, validation_instances=len(rows), validation_instances_passed=passed,
        feasible=esp >= SATISFACTION_PROBABILITY_TARGET)


def selection_key(candidate):
    if candidate['feasible']:
        return (0, candidate['mean_total_energy'], -candidate['workflow_success_rate'])
    return (1, -candidate['validation_ESP'], -candidate['workflow_success_rate'],
            candidate['mean_total_energy'])
