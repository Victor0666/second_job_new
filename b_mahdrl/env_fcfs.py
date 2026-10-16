# -*- coding: utf-8 -*-
"""
env_fcfs.py

基于离线 cache 的 workflow deadline 环境：

1) workflow 级 deadline:
   wf.deadline = arrival_time + ddl_alpha * makespan_ref
   其中 ddl_alpha 为“每个 workflow 独立采样”：
      - 20% 概率取 2.0
      - 80% 概率取 3.0

2) task 级 deadline:
   仍由 wf.deadline 做 Latest-Finish 反推得到 task_baseline_finish

3) Manager reward（phase 级）:
   r_mgr = - [ alpha * phase_delay_cost + (1-alpha) * phase_energy_cost ]

   其中：
     phase_energy_cost = phase_delta_energy * energy_reward_scale
     phase_delay_cost  = phase_delay / task_baseline_norm

说明：
- phase_delay 默认采用 tardiness（只累计 max(0, finish-deadline)）
- 若 manager_delay_mode="lateness"，则累计带符号 lateness
"""

import os
import json
import numpy as np

try:
    import gymnasium as gym
except Exception:
    import gym

from a_set.workflow_opt import Workflow
from a_set.read_xml_opt_Tsize import load_workflow_from_dax

# 继承你已有的 HEFT 基类环境
from b_mahdrl.env_heft import (
    CloudWorkflowEnv_VMAgents as _BaseEnv,
    MANAGER_ACTION_TABLE,
    _topo_sort_local,
)


def _avg_task_dur_over_all_vms(tasks, vm_pc, vm_bw_bps):
    """
    计算每个任务在“所有 VM 上 duration 的均值”：
        dur(i,p)=in/bw + mi/pc + out/bw
    返回 avg_dur: shape=(n_tasks,)
    """
    n = len(tasks)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    mi = np.array([float(t.workload_mi) for t in tasks], dtype=np.float64)
    in_bits = np.array(
        [float((getattr(t, "ext_in_bits", 0.0) or 0.0) + (getattr(t, "from_parents_bits", 0.0) or 0.0)) for t in tasks],
        dtype=np.float64,
    )
    out_bits = np.array([float(getattr(t, "out_file_size_sum_bits", 0.0) or 0.0) for t in tasks], dtype=np.float64)

    bw = vm_bw_bps.reshape(1, -1)
    pc = vm_pc.reshape(1, -1)
    dur = (
        in_bits.reshape(-1, 1) / np.maximum(bw, 1e-12)
        + mi.reshape(-1, 1) / np.maximum(pc, 1e-12)
        + out_bits.reshape(-1, 1) / np.maximum(bw, 1e-12)
    )
    return dur.mean(axis=1)


class CloudWorkflowEnv_VMAgents(_BaseEnv):
    """
    deadline_mode:
      - "cache_fcfs": 从 deadline_cache_path 查表
      - "none": 不使用 cache，deadline 给一个很大的占位值（通常用于离线预计算）

    workflow deadline:
      wf.deadline = arrival_time + ddl_alpha * makespan_ref
      ddl_alpha 为每个 workflow 独立采样：
        - deadline_alpha_small_prob 概率取 deadline_alpha_small
        - 其余概率取 deadline_alpha_large

    manager reward:
      r_mgr = - [ alpha * phase_delay_cost + (1-alpha) * phase_energy_cost ]
    """

    def __init__(
        self,
        *args,
        deadline_mode: str = "cache_fcfs",
        deadline_cache_path: str = None,
        deadline_cache_strict: bool = True,
        deadline_alpha_small: float = 2.0,
        deadline_alpha_large: float = 3.0,
        deadline_alpha_small_prob: float = 0.2,
        manager_alpha_delay: float = None,
        manager_alpha_energy: float = None,   # 兼容旧代码；若传这个，则自动换算为 delay 权重
        manager_delay_mode: str = "tardiness",
        **kwargs,
    ):
        self.deadline_mode = str(deadline_mode)
        self.deadline_cache_path = deadline_cache_path
        self.deadline_cache_strict = bool(deadline_cache_strict)

        self.deadline_alpha_small = float(deadline_alpha_small)
        self.deadline_alpha_large = float(deadline_alpha_large)
        self.deadline_alpha_small_prob = float(deadline_alpha_small_prob)

        if manager_alpha_delay is not None and manager_alpha_energy is not None:
            raise ValueError("manager_alpha_delay 与 manager_alpha_energy 只能传一个")

        if manager_alpha_delay is None and manager_alpha_energy is None:
            manager_alpha_delay = 0.5
        elif manager_alpha_delay is None:
            manager_alpha_delay = 1.0 - float(manager_alpha_energy)

        self.manager_alpha_delay = float(manager_alpha_delay)
        self.manager_delay_mode = str(manager_delay_mode).lower()

        self._deadline_cache = None
        super().__init__(*args, **kwargs)

        if self.deadline_mode == "cache_fcfs":
            if not self.deadline_cache_path:
                raise ValueError("deadline_mode='cache_fcfs' 但未提供 deadline_cache_path")
            self._deadline_cache = self._load_deadline_cache(self.deadline_cache_path)

        self._mgr_prev_total_energy = 0.0
        self._mgr_finished_task_ids = set()

    # ---------------- cache IO ----------------
    @staticmethod
    def _load_deadline_cache(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"deadline cache not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        cache = {}
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            for rec in obj["data"]:
                sd = int(rec["seed"])
                cache[sd] = rec
        elif isinstance(obj, dict) and "seed_to_record" in obj:
            for k, rec in obj["seed_to_record"].items():
                cache[int(k)] = rec
        else:
            raise ValueError("deadline cache json 格式不正确：必须包含 data(list) 或 seed_to_record(dict)")

        return cache

    def _cache_lookup_makespan(self, episode_seed: int, wf_id: int, dax_name: str = None) -> float:
        if self._deadline_cache is None:
            raise RuntimeError("deadline cache not loaded")

        rec = self._deadline_cache.get(int(episode_seed), None)
        if rec is None:
            msg = f"[deadline-cache] 缺少 seed={episode_seed} 的记录"
            if self.deadline_cache_strict:
                raise KeyError(msg)
            print("WARNING:", msg, "-> fallback makespan_ref=0")
            return 0.0

        ms_list = rec.get("wf_makespans", None)
        if ms_list is None:
            raise ValueError(f"[deadline-cache] seed={episode_seed} record 缺少 wf_makespans")

        if wf_id >= len(ms_list):
            msg = f"[deadline-cache] seed={episode_seed} wf_id={wf_id} 越界（cache里只有 {len(ms_list)} 个workflow）"
            if self.deadline_cache_strict:
                raise IndexError(msg)
            print("WARNING:", msg, "-> fallback makespan_ref=0")
            return 0.0

        if dax_name is not None:
            names = rec.get("wf_dax_names", None)
            if isinstance(names, list) and wf_id < len(names):
                if str(names[wf_id]) != str(dax_name):
                    print(
                        f"WARNING: [deadline-cache] seed={episode_seed} wf_id={wf_id} dax mismatch: "
                        f"runtime={dax_name} cache={names[wf_id]}"
                    )

        return float(ms_list[wf_id])

    # ---------------- ddl alpha 采样 ----------------
    def _sample_deadline_alpha_for_workflow(self, wf_id: int) -> float:
        """
        为每个 workflow 独立决定 ddlAlpha。
        注意：不能使用 self.rng，否则会污染 DAX 模板抽样序列。

        做法：
        - 基于 (episode_seed=random_seed, wf_id) 构造独立 RandomState
        - 再按 deadline_alpha_small_prob 进行采样
        """
        p = float(np.clip(self.deadline_alpha_small_prob, 0.0, 1.0))

        local_seed = (
                             int(self.random_seed) * 1000003
                             + int(wf_id) * 9176
                             + 20260316
                     ) & 0xFFFFFFFF

        rs = np.random.RandomState(local_seed)
        u = float(rs.rand())

        if u < p:
            return float(self.deadline_alpha_small)
        return float(self.deadline_alpha_large)

    # ---------------- reset ----------------
    def reset(self, *, seed=None, options=None):
        self.rng = np.random.RandomState(int(self.random_seed))
        ret = super().reset(seed=seed, options=options)

        self._mgr_prev_total_energy = float(getattr(self, "total_energy", 0.0))
        self._mgr_finished_task_ids = set()

        return ret

    # ---------------- phase 内新完成任务 ----------------
    def _collect_new_finished_task_ids(self):
        old_finished = set(self._mgr_finished_task_ids)
        cur_finished = set()
        new_finished = []

        n_tasks = len(getattr(self, "task_state", []))
        for tid in range(n_tasks):
            if self.task_state[tid] == "Finished":
                cur_finished.add(tid)
                if tid not in old_finished:
                    new_finished.append(tid)

        self._mgr_finished_task_ids = cur_finished
        return new_finished

    # ---------------- phase delay 计算 ----------------
    def _compute_phase_delay(self, finished_tids):
        phase_delay = 0.0
        for tid in finished_tids:
            if tid >= len(self.task_baseline_finish):
                continue

            finish_t = float(self.task_end_time[tid])
            ddl_t = float(self.task_baseline_finish[tid])
            late = finish_t - ddl_t

            if self.manager_delay_mode == "tardiness":
                phase_delay += max(0.0, late)
            elif self.manager_delay_mode == "lateness":
                phase_delay += late
            else:
                raise ValueError(f"Unsupported manager_delay_mode: {self.manager_delay_mode}")

        return float(phase_delay)

    # ---------------- 重写 Manager reward ----------------
    def finish_phase_and_advance(self):
        r_old, pinfo = super().finish_phase_and_advance()
        pinfo = dict(pinfo or {})

        cur_total_energy = float(getattr(self, "total_energy", 0.0))
        phase_energy = cur_total_energy - float(self._mgr_prev_total_energy)
        self._mgr_prev_total_energy = cur_total_energy

        new_finished_tids = self._collect_new_finished_task_ids()
        phase_delay = self._compute_phase_delay(new_finished_tids)

        energy_scale = float(getattr(self, "energy_reward_scale", 1.0))
        delay_norm = max(float(getattr(self, "task_baseline_norm", 1.0)), 1e-12)

        phase_energy_cost = phase_energy * energy_scale
        phase_delay_cost = phase_delay / delay_norm

        alpha_delay = float(np.clip(self.manager_alpha_delay, 0.0, 1.0))
        alpha_energy = 1.0 - alpha_delay

        # ===== 修改后的核心公式 =====
        r_manager_new = - (alpha_delay * phase_delay_cost + alpha_energy * phase_energy_cost)

        pinfo["phase_energy"] = float(phase_energy)
        pinfo["phase_delay"] = float(phase_delay)
        pinfo["phase_energy_cost"] = float(phase_energy_cost)
        pinfo["phase_delay_cost"] = float(phase_delay_cost)
        pinfo["manager_alpha_delay"] = float(alpha_delay)
        pinfo["manager_alpha_energy"] = float(alpha_energy)
        pinfo["manager_reward_old"] = float(r_old)
        pinfo["manager_reward_new"] = float(r_manager_new)
        pinfo["manager_new_finished_task_cnt"] = int(len(new_finished_tids))

        return float(r_manager_new), pinfo

    # ---------------- 核心：arrival 时设置 workflow deadline ----------------
    def _add_workflow_if_arrived(self):
        added = False
        while (
            self.next_arrival_idx < len(self.arrival_times)
            and self.arrival_times[self.next_arrival_idx] <= self.current_time
        ):
            arr_t = self.arrival_times[self.next_arrival_idx]

            if self.dax_probs is None:
                idx = self.rng.randint(len(self.dax_paths))
            else:
                idx = self.rng.choice(len(self.dax_paths), p=self.dax_probs)
            path = self.dax_paths[idx]
            dax_name = os.path.basename(str(path))

            G, tasks = load_workflow_from_dax(
                path,
                seed=self.random_seed + self.next_arrival_idx,
                randomize_payloads=True,
            )
            wf = Workflow(
                workflow_id=len(self.workflows),
                graph=G,
                tasks=tasks,
                arrival_time=arr_t,
                source_device_id=0,
            )
            wf.is_arrived = True
            wf.dax_path = str(path)

            pc_mean = float(np.mean([self.vms[v].pc for v in self.vms]))
            bw_mean = float(np.mean([self.vms[v].bw for v in self.vms]))
            wf.compute_upward_ranks(pc_mean, bw_mean)
            wf.compute_downward_ranks(pc_mean, bw_mean)

            self.wf_remaining_tasks[wf.workflow_id] = len(tasks)

            # ===== 每个 workflow 独立采样 ddlAlpha =====
            ddl_alpha = self._sample_deadline_alpha_for_workflow(wf.workflow_id)
            wf.deadline_alpha = float(ddl_alpha)

            if self.deadline_mode == "cache_fcfs":
                makespan_ref = self._cache_lookup_makespan(
                    episode_seed=int(self.random_seed),
                    wf_id=int(wf.workflow_id),
                    dax_name=dax_name,
                )
                wf.ref_makespan = float(makespan_ref)
                wf.deadline = float(arr_t + ddl_alpha * float(makespan_ref))

            elif self.deadline_mode == "none":
                wf.ref_makespan = None
                wf.deadline = float(arr_t + max(self.horizon, 1.0))

            else:
                raise ValueError(f"Unsupported deadline_mode: {self.deadline_mode}")

            # ===== task 子 deadline 反推 =====
            n_local = len(tasks)
            topo, succ = _topo_sort_local(tasks)

            avg_dur = _avg_task_dur_over_all_vms(tasks, self._heft_vm_pc, self._heft_vm_bw_bps)
            D_wf_rel = float(wf.deadline - arr_t)

            LF = np.zeros((n_local,), dtype=np.float64)
            LS = np.zeros((n_local,), dtype=np.float64)
            is_sink = np.array([len(succ[i]) == 0 for i in range(n_local)], dtype=bool)

            for i in range(n_local):
                if is_sink[i]:
                    LF[i] = D_wf_rel
                    LS[i] = LF[i] - avg_dur[i]

            for u in reversed(topo):
                if is_sink[u]:
                    continue
                if len(succ[u]) > 0:
                    min_LS_child = float(np.min(LS[np.array(succ[u], dtype=np.int32)]))
                    LF[u] = min_LS_child
                else:
                    LF[u] = D_wf_rel
                LS[u] = LF[u] - avg_dur[u]

            for local_id in range(n_local):
                abs_deadline = float(arr_t + LF[local_id])
                self.task_baseline_finish.append(abs_deadline)

            base = len(self.task_meta)
            for t in tasks:
                self.task_meta.append((wf.workflow_id, t.task_id))
                self.task_state.append("unReady")
                self.task_parents.append(list(t.parents))
                self.task_children.append(list(t.children))
                self.task_mi.append(float(t.workload_mi))
                in_bits_total = float((t.ext_in_bits or 0.0) + (t.from_parents_bits or 0.0))
                self.task_in_bits.append(in_bits_total)
                self.task_out_bits.append(float(t.out_file_size_sum_bits))
                self.task_up_rank.append(float(t.upward_rank or 0.0))
                self.task_down_rank.append(float(t.downward_rank or 0.0))
                self.task_ready_time.append(0.0)
                self.task_end_time.append(0.0)

            for local_id, t in enumerate(tasks):
                if len(t.parents) == 0:
                    gid = base + local_id
                    self.task_state[gid] = "Ready"
                    self.task_ready_time[gid] = self.current_time
                    self.ready_task_ids.append(gid)

            self.workflows.append(wf)
            self.next_arrival_idx += 1
            added = True

        return added