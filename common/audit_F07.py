"""Reproducible per-workflow payload and FCFS reference measurements."""
import argparse
import json
from pathlib import Path
from common.precompute_deadlines import build_env_kwargs, run_baseline_episode_fcfs_vmfirst
from common.scenarios import Scenario, DAX_GROUPS, ROOT
from environments.base_fcfs import CloudWorkflowEnv_VMAgents

def measure():
    rows = []
    cases = [('SS_50', None)] + [(name, name) for group in DAX_GROUPS for name in group]
    for case, template in cases:
        kwargs = build_env_kwargs(Scenario())
        kwargs['random_seed'] = 2
        if template:
            kwargs.update(dax_paths=[str(ROOT / 'data' / 'dax' / template)], workflows_per_episode=1)
        env = CloudWorkflowEnv_VMAgents(**kwargs)
        env.reset()
        makespans, names, _ = run_baseline_episode_fcfs_vmfirst(env)
        for wf, name, makespan in zip(env.workflows, names, makespans):
            edges = [bits for t in wf.tasks for bits in t.parent_in_bits.values()]
            rows.append(dict(case=case, seed=2, workflow_id=wf.workflow_id, template=name,
                             nodes=len(wf.tasks), edges=wf.graph.number_of_edges(),
                             mean_edge_bits=sum(edges) / len(edges) if edges else 0,
                             max_task_input_bits=max(t.in_file_size_sum_bits for t in wf.tasks),
                             logical_communication_bits=sum(edges), reference_makespan_s=makespan))
    return rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    Path(args.out).write_text(json.dumps(measure(), indent=2), encoding='utf-8')
