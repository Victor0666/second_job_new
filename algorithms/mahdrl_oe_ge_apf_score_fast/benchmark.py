"""Legacy APF-backbone execution timings; this does not measure NoAPF v3 episodes."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import platform
import statistics
from time import perf_counter
import numpy as np
import torch
from algorithms.mahdrl_oe_ge_apf_score import agent as original, runtime as old_runtime
from algorithms.mahdrl_oe_ge_apf_score.networks import collate as old_collate
from algorithms.mahdrl_oe_ge_apf_score_fast import agent as fast, runtime
from algorithms.mahdrl_oe_ge_apf_score_fast.packing import collate
from algorithms.mahdrl_oe_ge_apf_score_fast.scenario import ScoreScenario
from environments.mahdrl_oe_ge_apf_score import CloudWorkflowEnv_VMAgents as OriginalEnv
from environments.mahdrl_oe_ge_apf_score_fast import CloudWorkflowEnv_VMAgents as FastEnv


def observation(n, c=23, f=10):
    return dict(context=np.arange(c, dtype=np.float32)/c,
                candidates=np.arange(n*f, dtype=np.float32).reshape(n, f)/max(1, n*f),
                exists_mask=np.ones(n, np.float32))


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def measure(function, repeats, device="cpu"):
    samples = []
    for _ in range(repeats):
        sync(device)
        start = perf_counter()
        function()
        sync(device)
        samples.append(perf_counter()-start)
    return dict(median_ms=statistics.median(samples)*1000, mean_ms=statistics.mean(samples)*1000,
                p95_ms=float(np.percentile(samples, 95))*1000, repeats=repeats)


def paired(values):
    return dict(**values, speedup=values["original"]["median_ms"]/values["fast"]["median_ms"])


def fill(agent, role, count, n):
    for i in range(count):
        stop = i % 5 == 0
        if role == "TA":
            s, m = np.arange(15, dtype=np.float32)/15, np.ones(243, np.float32)
            nxt, nm = s*.9, np.zeros_like(m) if stop else m
        else:
            c, f = (23, 10) if role == "HA" else (30, 7)
            s, m = observation(n, c, f), np.ones(n, np.float32)
            nxt, nm = observation(0 if stop else n, c, f), np.ones(0 if stop else n, np.float32)
        agent.remember(s, m, 0, -.1*(i%7), nxt, nm, stop)


def digest(agent):
    h = hashlib.sha256()
    for tensor in agent.online.state_dict().values():
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def make_agent(module, role, device):
    args = dict(device=torch.device(device), **({"profile_mode": "off"} if module is fast else {}))
    if role == "TA":
        return module.TAAgent(15, 243, hidden_dims=(512, 512, 256, 128), **runtime.learning_kwargs(role, torch.device(device)),
                              **{k: v for k, v in args.items() if k != "device"})
    return module.SetD3QNAgent(*( (23, 10) if role == "HA" else (30, 7)),
                              **runtime.learning_kwargs(role, torch.device(device)),
                              **{k: v for k, v in args.items() if k != "device"})


def benchmark(device, updates=20, pack_repeats=300, episode=False, resource="S"):
    result = dict(python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
                  algorithm_scope="legacy_apf_backbone_only_not_noapf_v3",
                  torch=torch.__version__, cuda=torch.version.cuda, device=device,
                  gpu=torch.cuda.get_device_name(device) if device == "cuda" else None,
                  threads=torch.get_num_threads(), batch_size=256,
                  scope="synthetic_replay_formal_network_widths; warmups_excluded; not_600_episode_training",
                  diagnostics="original_default_full_vs_fast_default_every100; profiler_off", pack={}, updates={})
    for name, sizes in (("equal6", [6]*256), ("mixed", [3, 6, 9, 12]*64)):
        obs, masks = [observation(n) for n in sizes], [np.ones(n, np.float32) for n in sizes]
        values = {}
        for label, function in (("original", old_collate), ("fast", collate)):
            for _ in range(5): function(obs, masks, "cpu")
            values[label] = measure(lambda: function(obs, masks, "cpu"), pack_repeats)
        result["pack"][name] = paired(values)
    for role, n in (("TA", 243), ("HA", 6), ("VA", 9)):
        values, hashes, losses = {}, {}, {}
        for label, module in (("original", original), ("fast", fast)):
            runtime.set_seed(123)
            agent = make_agent(module, role, device)
            fill(agent, role, 288, n)
            for _ in range(3): agent.update()
            history = []
            values[label] = measure(lambda: history.append(agent.update()), updates, device)
            losses[label], hashes[label] = history, digest(agent)
            del agent
        if hashes["original"] != hashes["fast"] or losses["original"] != losses["fast"]:
            raise AssertionError("Training equivalence failed in benchmark: " + role)
        result["updates"][role] = dict(paired(values), final_weights_exact=True, losses_exact=True)
    if episode:
        scene = ScoreScenario(mode="train", resource=resource, legacy_split=True, device=device, profile_mode="off")
        values, hashes, metrics = {}, {}, {}
        for label, module, envcls, runner in (("original", original, OriginalEnv, old_runtime), ("fast", fast, FastEnv, runtime)):
            runtime.set_seed(123)
            args = dict(runtime.build_env_kwargs(scene, 1), workflows_per_episode=2, max_ready_tasks=100)
            with redirect_stdout(io.StringIO()):
                env = envcls(**args, **({"profile_mode": "off"} if label == "fast" else {}))
                env.reset()
            agents = {r: make_agent(module, r, device) for r in ("TA", "HA", "VA")}
            for role, agent in agents.items(): fill(agent, role, 288, scene.num_hosts if role == "HA" else max(scene.vms_per_host))
            sync(device)
            started = perf_counter()
            with redirect_stdout(io.StringIO()): metrics[label] = runner.run_episode(env, agents, training=True, warmup=0.)
            sync(device)
            values[label] = dict(median_ms=(perf_counter()-started)*1000,
                                 updates={r: a._updates for r, a in agents.items()})
            hashes[label] = {r: digest(a) for r, a in agents.items()}
            del agents, env
        if hashes["original"] != hashes["fast"] or metrics["original"] != metrics["fast"]:
            raise AssertionError("Full episode equivalence failed")
        result["short_episode"] = dict(paired(values), resource=resource, workflows=2,
            scope="one_short_training_episode_with_synthetic_prefilled_replay_and_warmup0; excludes_setup_validation_and_IO",
            final_weights_exact=True, scheduling_metrics_exact=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--pack-repeats", type=int, default=300)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--episode", action="store_true", help="Needs existing S/M/L reference cache and normalization config")
    p.add_argument("--resource", choices="SML", default="S")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if min(args.updates, args.pack_repeats, args.threads) < 1:
        p.error("Counts must be positive")
    if args.out and args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(args.threads)
    report = benchmark(args.device, args.updates, args.pack_repeats, args.episode, args.resource)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream: stream.write(text+"\n")
    print(text)


if __name__ == "__main__":
    main()
