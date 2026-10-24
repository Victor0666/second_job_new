"""Isolated real-graph GPU probes; no formal training/checkpoints are produced."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from algorithms.two_stage_gat.config import TwoStageGATConfig
from algorithms.two_stage_gat.environment import TwoStageGATEnv
from algorithms.two_stage_gat.ppo import PPOAgent, Transition
from algorithms.two_stage_gat.runner import env_kwargs, seed_everything
from common.read_xml_opt_Tsize import load_workflow_from_dax
from common.scenarios import Scenario


def real_state(task, resource, mode):
    scene = Scenario(mode=mode, task=task, train_task='S', resource=resource)
    env = TwoStageGATEnv(**env_kwargs(scene, 0 if mode == 'train' else 700))
    counts = {Path(p).name.casefold(): len(load_workflow_from_dax(p, seed=0)[1])
              for p in dict.fromkeys(env.dax_paths)}
    seeds = scene.protocol().train if mode == 'train' else scene.protocol().test
    def size(seed):
        return sum(counts[Path(p).name.casefold()]
                   for p in env._deadline_cache[int(seed)]['wf_dax_names'][:50])
    seed = max(seeds, key=size)
    env.reset(seed=int(seed))
    total = size(seed)
    while env.assignment_count < total - 1:
        gid = env.ready_task_ids[0]
        vm = env.assignment_count % env.num_vms
        env.pair_step(gid * env.num_vms + vm)
    state = env.graph_state()
    assert len(env.workflows) == 50 and len(state.task_features) == total
    assert state.task_mask.any()
    return state, dict(task=task, resource=resource, instance_seed=int(seed),
        task_nodes=total, edges=len(state.dependencies), vms=env.num_vms,
        snapshot='last physical assignment; all 50 workflows arrived; round-robin VM reservations')


def probe(state, batch, mode):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    agent = records = None
    started = time.perf_counter()
    try:
        seed_everything(1)
        # One full minibatch isolates production activation memory, without 600 episodes.
        cfg = TwoStageGATConfig(device='cuda', update_epochs=1, num_minibatches=1)
        agent = PPOAgent(cfg)
        action, logp, value = agent.act(state, deterministic=True, count_step=False)
        if mode == 'train':
            records = [Transition(state, action, logp, value, -1., True, 0.) for _ in range(batch)]
            for _ in range(2):  # Include Adam-state residency on the second backward pass.
                agent.update(records)
        else:
            for _ in range(3):
                agent.act(state, deterministic=True, count_step=False)
        torch.cuda.synchronize()
        return dict(batch=batch, completed=True,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            seconds=time.perf_counter() - started)
    except torch.cuda.OutOfMemoryError:
        return dict(batch=batch, completed=False, reason='profiling allocator safety cap/OOM',
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            seconds=time.perf_counter() - started)
    finally:
        del records, agent
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', default='docs/Two_stage_GAT_memory_profile.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for GPU memory measurement')
    fraction = .80  # Full 128-row probe while retaining space for the display/context.
    torch.cuda.set_per_process_memory_fraction(fraction)
    free, total = torch.cuda.mem_get_info()
    result = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        gpu_total_mib=total/2**20, free_at_start_mib=free/2**20,
        allocator_fraction_cap=fraction, production_minibatch=128,
        method='real late-episode graphs repeated across a minibatch; two Adam backward/update passes',
        limitations='PyTorch allocated/reserved only; excludes CUDA context and other processes; capped probes may require extrapolation',
        scenarios=[])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for task, resource, mode in [('S',r,'train') for r in 'SML'] + [('L','L','eval')]:
        state, meta = real_state(task, resource, mode)
        case = dict(**meta, mode=mode, probes=[])
        result['scenarios'].append(case)
        print(json.dumps(dict(stage='state_ready', **meta, mode=mode)), flush=True)
        for batch in ((128,) if mode == 'train' else (1,)):
            sample = probe(state,batch,mode)
            case['probes'].append(sample)
            out.write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(dict(task=task,resource=resource,mode=mode,**sample)),flush=True)
            if not sample['completed']:
                break
        del state
