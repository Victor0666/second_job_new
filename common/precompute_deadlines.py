"""Generate FCFS/FIFS reference caches for any task/resource combination."""

import os
import sys
import json
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from common.scenarios import Scenario, parse_scenario
from common.read_xml_opt_Tsize import DAX_TEMPLATE_ID_SCHEME, data_model_metadata, workflow_instance_hash
from common.workflow_opt import COMMUNICATION_MODEL
from environments.base_fcfs import CloudWorkflowEnv_VMAgents


def build_env_kwargs(scenario):
    return dict(
        dax_paths=scenario.dax_paths(),
        horizon=1e9,
        arrival_lambda=0.03,
        random_seed=0,
        max_ready_tasks="auto",
        normalize=True,
        workflows_per_episode=50,
        num_hosts=scenario.num_hosts,
        vms_per_host=scenario.vms_per_host,
        pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),
        deadline_mode="none",
        deadline_cache_path=None,
        deadline_alpha=1.5,
        deadline_cache_strict=True,
    )


def _pick_fcfs_task(env) -> int:
    # FCFS：ready_time 最小，tie 用 tid
    return int(min(env.ready_task_ids, key=lambda tid: (env.task_ready_time[tid], tid)))


def _pick_vm_first_idle(env) -> int:
    idle = np.where(env._vm_idle_mask())[0]
    # 先空闲先服务：按 (vm_available_at, vm_idx) 最小
    return int(min(idle, key=lambda j: (env.vm_available_at[j], j)))


def run_baseline_episode_fcfs_vmfirst(env: CloudWorkflowEnv_VMAgents):
    """
    在当前 env（已 reset）上跑 baseline，直到 done_flag=True
    """
    # 先推进到第一个 decision point（或结束）
    env._advance_until_decision_energy_only()

    while not bool(getattr(env, "done_flag", False)):
        # 在同一时刻尽可能分配（只要还有 ready 且有 idle VM）
        while env._has_decision_point():
            tid = _pick_fcfs_task(env)
            vm_idx = _pick_vm_first_idle(env)
            env._assign_task_to_specific_vm(tid, vm_idx)

        # 推进到下一次 decision / 或结束
        env._advance_until_decision_energy_only()

    # episode 完成后，统计 makespan
    wf_makespans = []
    wf_dax_names = []
    for wf_id in range(int(env.workflows_per_episode)):
        wf = env.workflows[wf_id]
        finish_t = float(env.wf_finish_time.get(wf_id, env.current_time))
        makespan = max(0.0, finish_t - float(wf.arrival_time))
        wf_makespans.append(float(makespan))
        wf_dax_names.append(os.path.basename(getattr(wf, "dax_path", "")))

    arrival_times = [float(x) for x in env.arrival_times]
    return wf_makespans, wf_dax_names, arrival_times


def main(scenario):
    output = scenario.precompute_path
    # Even explicit --out must never overwrite historical or existing cache artifacts.
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite deadline cache: {output}")
    env_kwargs = build_env_kwargs(scenario)
    data = []
    for seed in scenario.seeds:
        env_kwargs["random_seed"] = seed
        env = CloudWorkflowEnv_VMAgents(**env_kwargs)
        env.reset()
        makespans, names, arrivals = run_baseline_episode_fcfs_vmfirst(env)
        data.append(
            dict(seed=seed, wf_makespans=makespans, wf_dax_names=names, arrival_times=arrivals,
                 wf_instance_hashes=[workflow_instance_hash(wf.graph, wf.tasks) for wf in env.workflows])
        )
        print(f"[precompute] seed={seed} avg_makespan={float(np.mean(makespans)):.6f}s", flush=True)
    result = dict(
        meta=dict(
            policy="FCFS_task + VM_first_idle",
            communication_model=COMMUNICATION_MODEL,
            **data_model_metadata(),
            dax_template_id_scheme=DAX_TEMPLATE_ID_SCHEME,
            alpha=1.5,
            arrival_lambda=0.03,
            workflows_per_episode=50,
            num_hosts=scenario.num_hosts,
            vms_per_host=list(scenario.vms_per_host),
            pc_tiers=list(env_kwargs["pc_tiers"]),
            bw_tiers=list(env_kwargs["bw_tiers"]),
            dax_paths=[Path(p).name for p in scenario.dax_paths()],
            task=scenario.task,
            resource=scenario.resource,
        ),
        data=data,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(f"Saved cache: {output}")


if __name__ == "__main__":
    main(parse_scenario("precompute"))
