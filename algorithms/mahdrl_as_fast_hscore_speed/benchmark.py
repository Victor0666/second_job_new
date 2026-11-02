"""Bounded, synchronized production-width learner benchmark; no training outputs."""
import argparse
import gc
import json
from pathlib import Path
import platform
import random
import statistics
import sys
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from algorithms.mahdrl_as_fast_hscore import agent as original
from algorithms.mahdrl_as_fast_hscore.runtime import set_seed
from algorithms.mahdrl_as_fast_hscore.config import TA_HIDDEN, learning_kwargs
from algorithms.mahdrl_as_fast_hscore_speed import agent as accelerated


def build(module, role, device):
    options = dict(learning_kwargs(role, torch.device(device)), profile_mode="off")
    if role == "TA":
        return module.TAAgent(15, 243, hidden_dims=TA_HIDDEN, **options)
    if role == "HA":
        return module.HAAgent(23, 14, **options)
    return module.FixedVAAgent(**options)


def populate(old, new, role, *, hosts=3):
    rng = np.random.default_rng(8)
    for i in range(512):
        if role == "HA":
            state, following = [dict(context=rng.normal(size=23).astype(np.float32),
                candidates=rng.normal(size=(hosts, 14)).astype(np.float32),
                exists_mask=np.ones(hosts, np.float32)) for _ in range(2)]
            width = hosts
        else:
            dim, width = (15, 243) if role == "TA" else (93, 9)
            state, following = rng.normal(size=(2, dim)).astype(np.float32)
        mask, next_mask = rng.integers(0, 2, size=(2, width)).astype(np.float32)
        mask[0] = next_mask[0] = 1
        done = i % 7 == 0
        if done:
            next_mask[:] = 0
        for learner in (old, new):
            learner.remember(state, mask, 0, -.1, following, next_mask, done)
    # Reuse immutable records to bound benchmark RAM while exercising a full buffer.
    old_rows, new_rows = list(old.buffer), list(new.buffer)
    for i in range(old.buffer.maxlen-len(old.buffer)):
        old.buffer.append(old_rows[i % len(old_rows)])
        new.buffer.append(new_rows[i % len(new_rows)])


def elapsed(device, operation):
    if device == "cuda":
        torch.cuda.synchronize()
    start = perf_counter()
    operation()
    if device == "cuda":
        torch.cuda.synchronize()
    return perf_counter()-start


def updates(learner, count):
    for _ in range(count):
        learner.update()


def benchmark_role(role, device, count, repeats):
    set_seed(1)
    old = build(original, role, device)
    set_seed(1)
    new = build(accelerated, role, device)
    populate(old, new, role)
    for learner in (old, new):
        set_seed(2)
        updates(learner, 5)
    timings = {"original": [], "speed": []}
    for repeat in range(repeats):
        order = [("original", old), ("speed", new)]
        if repeat % 2:
            order.reverse()
        for label, learner in order:
            set_seed(100+repeat)
            timings[label].append(elapsed(device, lambda: updates(learner, count))/count)
    medians = {key: statistics.median(values) for key, values in timings.items()}
    sample = {}
    for label, learner in (("original", old), ("speed", new)):
        set_seed(7)
        sample[label] = elapsed(device, lambda: [random.sample(learner.buffer, learner.batch_size)
                                                  for _ in range(100)])/100
    identical = all(torch.equal(value, new.online.state_dict()[key])
                    for key, value in old.online.state_dict().items())
    if not identical:
        raise RuntimeError("Benchmark learners diverged")
    return dict(role=role, batch_size=old.batch_size, buffer_capacity=old.buffer.maxlen,
        online_parameters=sum(p.numel() for p in old.online.parameters()),
        median_update_seconds=medians, update_speedup=medians["original"]/medians["speed"],
        repeat_seconds_per_update=timings, sample_seconds=sample,
        sample_speedup=sample["original"]/sample["speed"], final_online_parameters_exactly_equal=identical)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.updates, args.repeats) < 1:
        parser.error("updates and repeats must be positive")
    torch.set_num_threads(1)
    result = dict(scope="synthetic_full_replay_production_width_learner_updates_not_end_to_end_training",
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        device=args.device, gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
        cpu_threads=torch.get_num_threads(), updates_per_repeat=args.updates, repeats=args.repeats,
        timing="wall_clock_cuda_synchronized_alternating_order_after_5_warmup_updates",
        data="512 distinct immutable transitions repeated to full capacity; 1/7 terminal; HA has 3 hosts",
        roles=[])
    for role in ("TA", "HA", "VA"):
        row = benchmark_role(role, args.device, args.updates, args.repeats)
        result["roles"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        gc.collect()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
