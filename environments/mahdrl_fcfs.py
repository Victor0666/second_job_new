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

from common.workflow_opt import validate_deadline_cache, task_subdeadlines, sample_deadline_alpha
from common.workflow_opt import Workflow
from common.read_xml_opt_Tsize import validate_dax_template_name
from common.read_xml_opt_Tsize import load_workflow_from_dax

# 继承你已有的 HEFT 基类环境
from environments.mahdrl_heft import (
    CloudWorkflowEnv_VMAgents as _BaseEnv,
    MANAGER_ACTION_TABLE,
)


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
        validate_deadline_cache(obj, path)

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
                validate_dax_template_name(names[wf_id], dax_name, strict=self.deadline_cache_strict)

        return float(ms_list[wf_id])

    # ---------------- ddl alpha 采样 ----------------
    def _sample_deadline_alpha_for_workflow(self, wf_id: int) -> float:
        p = float(np.clip(self.deadline_alpha_small_prob, 0.0, 1.0))
        return sample_deadline_alpha(
            self.random_seed, wf_id,
            (self.deadline_alpha_small, self.deadline_alpha_large), (p, 1.0 - p),
        )

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
            subdeadlines = task_subdeadlines(
                tasks, self._heft_vm_pc, self._heft_vm_bw_bps, arr_t, wf.deadline,
            )
            self.task_baseline_finish.extend(subdeadlines.tolist())

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