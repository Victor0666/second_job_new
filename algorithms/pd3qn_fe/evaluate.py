"""Evaluate only this variant's selected checkpoint on the frozen test partition."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from algorithms.pd3qn_fe import ALGORITHM, DISPLAY_NAME
from algorithms.pd3qn_fe.runtime import (env_kwargs, evaluate_instances, load_best_checkpoint,
    make_agent, set_seed)
from algorithms.pd3qn_fe.scenario import parse_scenario
from algorithms.pd3qn_fe.selection import selection_metadata
from common.f11 import protect_training_outputs
from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost


def evaluate(scenario):
    directory = scenario.evaluation_dir
    path = scenario.evaluation_csv
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite evaluation CSV: {path}')
    protect_training_outputs(directory, directory)
    kwargs = env_kwargs(scenario)
    set_seed(scenario.model_seed)
    env = CloudWorkflowEnv_PD3QN_Cost(**kwargs)
    try:
        agent = make_agent(scenario, env)
    finally:
        if hasattr(env, 'close'):
            env.close()
    checkpoint = scenario.checkpoint_dir() / 'best_pd3qn_fe.pth'
    selected = load_best_checkpoint(agent, checkpoint, scenario)
    print(f'[{DISPLAY_NAME} selected checkpoint] episode={selected["episode"]} '
          f'validation={selected["validation_selection"]}', flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = evaluate_instances(agent, scenario, scenario.seeds, split='test', kwargs=kwargs,
                              directory=directory / 'measurements')
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(algorithm=ALGORITHM, checkpoint_selection=selection_metadata(),
        protocol=scenario.selection_protocol(), source_checkpoint=str(checkpoint),
        selected_checkpoint_validation=selected,
        summary={key: dict(mean=float(np.mean([row[key] for row in rows])),
            std=float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.)
            for key in rows[0] if key != 'seed'})
    (directory / 'summary_pd3qn_fe.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'[{DISPLAY_NAME} eval completed] csv={path}', flush=True)
    return rows, summary


if __name__ == '__main__':
    evaluate(parse_scenario('eval'))
