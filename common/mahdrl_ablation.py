"""AS-Fast local selector: aggregate workflow SR > .92, then minimum energy."""
import math

SELECTION_TAG = 'select_srgt092_minenergy_v1'
WORKFLOW_SUCCESS_THRESHOLD = .92


def selection_metadata():
    return dict(version=SELECTION_TAG, rule='validation_feasible_then_minimum_energy',
        workflow_success_threshold=WORKFLOW_SUCCESS_THRESHOLD, workflow_success_comparison='strictly_greater',
        feasibility_metric='total_successful_workflows_divided_by_total_workflows',
        energy_metric='mean_validation_instance_total_energy',
        success_metric='total_successful_workflows_divided_by_total_workflows',
        fallback='maximum_workflow_success_then_minimum_energy',
        tie_break='higher_workflow_success_then_earlier_checkpoint',
        population_probability_guaranteed=False)


def validation_selection(rows):
    if not rows:
        raise ValueError('Checkpoint selection requires validation instances')
    successes = total = 0
    energies = []
    for row in rows:
        count, success = row['wf_count'], row['wf_success_count']
        energy = float(row['total_energy'])
        if (not math.isfinite(count) or not math.isfinite(success)
                or count <= 0 or int(count) != count or int(success) != success
                or not 0 <= success <= count or not math.isfinite(energy) or energy < 0):
            raise ValueError('Invalid validation workflow counts or energy')
        if not math.isclose(float(row['wf_success_rate']), success / count, rel_tol=0., abs_tol=1e-12):
            raise ValueError('Validation workflow success rate/count mismatch')
        successes += int(success)
        total += int(count)
        energies.append(energy)
    return dict(workflow_success_rate=successes / total, workflow_success_count=successes,
        workflow_count=total, mean_total_energy=sum(energies) / len(rows),
        validation_instances=len(rows), feasible=successes / total > WORKFLOW_SUCCESS_THRESHOLD)


def selection_key(candidate):
    if candidate['feasible']:
        return (0, candidate['mean_total_energy'], -candidate['workflow_success_rate'])
    return (1, -candidate['workflow_success_rate'], candidate['mean_total_energy'])


def validate_checkpoint_selection(record, expected):
    """Recompute saved evidence before accepting any new-rule checkpoint."""
    if not isinstance(record, dict):
        raise ValueError('Missing Fast ablation checkpoint selection metadata; retrain under the new rule')
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError('MAHDRL-AS-Fast checkpoint selection mismatch: ' + key)
    episode = record.get('episode')
    selected = record.get('selected_episode')
    if (type(episode) is not int or episode < 0 or type(selected) is not int
            or not 0 <= selected <= episode or record.get('completed_episodes') != episode + 1):
        raise ValueError('Invalid checkpoint selection episode')
    train_seeds = expected['protocol'].get('train')
    if train_seeds is not None and episode >= len(train_seeds):
        raise ValueError('Checkpoint episode outside training protocol')
    summaries = []
    for name in ('validation_rows', 'best_validation_rows'):
        rows = record.get(name)
        if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
                or [row.get('seed') for row in rows] != expected['validation_seeds']):
            raise ValueError('Checkpoint validation seeds mismatch: ' + name)
        try:
            summaries.append(validation_selection(rows))
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError('Invalid checkpoint validation evidence: ' + name) from error
    candidate, best_metrics = summaries
    best = dict(best_metrics, episode=selected, completed_episodes=selected + 1)
    if record.get('validation_selection') != candidate or record.get('best_validation_selection') != best:
        raise ValueError('Checkpoint validation metrics mismatch')
    kind = record.get('checkpoint_kind')
    if (kind not in ('best', 'final') or selection_key(best) > selection_key(candidate)
            or (selected == episode and best_metrics != candidate)
            or (kind == 'best' and selected != episode)):
        raise ValueError('Checkpoint best/final selection mismatch')
    return record


def no_apf_metadata(metadata):
    """Declare raw legal candidates consistently in every ablation identity."""
    metadata = dict(metadata)
    metadata.setdefault("guided_probability", metadata.get("apf", {}).get("guided_probability", .5))
    metadata.update(host_apf="disabled", vm_apf="disabled",
        host_apf_enabled=False, vm_apf_enabled=False, performance_action_filter=False,
        candidate_filter="physical_legality_only",
        apf=dict(scheme="physical_legality_only_v1", host_enabled=False, vm_enabled=False),
        host_summary_scope=("not_used" if metadata.get("host_summary_scope") == "not_used"
                            else "all_physically_legal_idle_vms"))
    metadata["components"] = [value for value in metadata.get("components", []) if "APF" not in value and value != "PF"]
    return metadata
