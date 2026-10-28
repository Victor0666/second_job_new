"""Deterministic ordering of already arrived, currently Ready tasks."""
import math
from .config import get_config


def ordered_ready_tasks(env, rule):
    get_config(rule)
    ids = [int(tid) for tid in env.ready_task_ids if env.task_state[tid] == "Ready"]
    if len(ids) != len(set(ids)):
        raise ValueError("Ready task IDs must be unique")

    def key(tid):
        ready = float(env.task_ready_time[tid])
        if rule == "ertf":
            result = (ready, tid)
        elif rule == "edf":
            result = (float(env.task_baseline_finish[tid]), ready, tid)
        else:
            result = (-float(env.task_up_rank[tid]), ready, tid)
        if not all(math.isfinite(value) for value in result):
            raise ValueError("Fixed task ordering requires finite priority attributes")
        return result

    return sorted(ids, key=key)
