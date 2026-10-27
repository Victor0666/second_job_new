"""Paired checkpoints and isolated validation for the independent MARL learners."""
from contextlib import contextmanager
import random
import uuid
import numpy as np
import torch
from algorithms.pd3qn_dag_fe.selection import selection_metadata, validation_selection
from algorithms.marl.identity import OBSERVATION_LAYOUT, SCHEDULING_POLICY, QUEUE_CAPACITY


@contextmanager
def isolated_evaluation(global_agent, locals):
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    modules = [m for a in (global_agent, *locals.agents) for m in (a.actor, a.critic)]
    modes = [m.training for m in modules]
    try:
        for module in modules:
            module.eval()
        with torch.no_grad():
            yield
    finally:
        for module, mode in zip(modules, modes):
            module.train(mode)
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])


def selection_record(scenario, episode, candidate, best, rows, role):
    return dict(pair_id=str(uuid.uuid4()), episode=episode, completed_episodes=episode+1, role=role,
        checkpoint_selection=selection_metadata(), protocol=scenario.selection_protocol(),
        observation_layout=OBSERVATION_LAYOUT, scheduling_policy=SCHEDULING_POLICY,
        queue_capacity=QUEUE_CAPACITY,
        source_train_scene=scenario.normalization_kwargs()['source_train_scene'],
        model_seed=scenario.model_seed, validation_selection=candidate,
        best_validation_selection=best, validation=rows)


def validate_pair(global_agent, locals, scenario):
    metadata = getattr(global_agent, 'training_selection', None)
    if not metadata or metadata != getattr(locals, 'training_selection', None):
        raise ValueError('MARL Global/Local checkpoints are not a selected pair')
    expected = dict(role='best', checkpoint_selection=selection_metadata(),
        observation_layout=OBSERVATION_LAYOUT, scheduling_policy=SCHEDULING_POLICY,
        queue_capacity=QUEUE_CAPACITY,
        protocol=scenario.selection_protocol(), model_seed=scenario.model_seed,
        source_train_scene=scenario.normalization_kwargs()['source_train_scene'])
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError('MARL checkpoint selection mismatch: ' + key)
    rows = metadata.get('validation', [])
    if [row['seed'] for row in rows] != scenario.validation_seeds():
        raise ValueError('MARL checkpoint validation seeds mismatch')
    if validation_selection(rows) != metadata.get('validation_selection'):
        raise ValueError('MARL checkpoint validation results mismatch')
    return metadata
