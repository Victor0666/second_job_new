# -*- coding: utf-8 -*-
"""
env_cloud_workflow_d3qn_state_reworked_multiagent_Tsize_alpha15_DDLCACHE_HEFT.py

基于“动态基线调度（FCFS任务顺序 + 先空闲先服务分配VM）”离线预计算的 makespan，
在运行期通过 cache 查表设置 workflow deadline：
    wf.deadline = arrival_time + alpha * makespan_ref

并保持：任务子 deadline 仍由 wf.deadline 倒推（Latest-Finish 反推），P 使用“平均 VM 时长”的 avg_dur。

依赖：
- 复用你现有 HEFT 版本 env 的绝大部分逻辑（继承），仅覆写：
  1) reset()：每 episode 重新播种 self.rng（保证 seed->模板序列可复现）
  2) _add_workflow_if_arrived()：不跑 HEFT，改为从 cache 读取 makespan_ref，并倒推任务子 deadline

注意：
- 该 env 仍会维护 wf.deadline 以供 EDF 特征、manager slack 等使用。
- 若 deadline_mode="none"（用于离线预计算阶段），则 wf.deadline 用一个“很大”的占位值，避免特征/奖励崩。
"""

import os
import json
import heapq
import numpy as np

try:
    import gymnasium as gym
except Exception:
    import gym

from common.workflow_opt import validate_deadline_cache, task_subdeadlines
from common.workflow_opt import Workflow
from common.read_xml_opt_Tsize import validate_cached_instance, validate_dax_template_name
from common.read_xml_opt_Tsize import load_workflow_from_dax, poisson_arrival_times

# 继承你已有的 HEFT deadline env（不改动原文件）
from environments.base_heft import (
    CloudWorkflowEnv_VMAgents as _BaseEnv,
    MANAGER_ACTION_TABLE,  # 训练脚本还会用
)


class CloudWorkflowEnv_VMAgents(_BaseEnv):
    """
    deadline_mode:
      - "cache_fcfs" : 从 deadline_cache_path 查表（严格推荐）
      - "none"       : 不用 cache，deadline 用大值占位（用于离线预计算 baseline）
    """

    def __init__(
        self,
        *args,
        deadline_mode: str = "cache_fcfs",
        deadline_cache_path: str = None,
        deadline_alpha: float = 1.5,
        deadline_cache_strict: bool = True,
        **kwargs,
    ):
        self.deadline_mode = str(deadline_mode)
        self.deadline_cache_path = deadline_cache_path
        self.deadline_alpha = float(deadline_alpha)
        self.deadline_cache_strict = bool(deadline_cache_strict)

        self._deadline_cache = None  # seed -> record
        super().__init__(*args, **kwargs)

        # 加载 cache（仅 cache_fcfs 模式需要）
        if self.deadline_mode == "cache_fcfs":
            if not self.deadline_cache_path:
                raise ValueError("deadline_mode='cache_fcfs' 但未提供 deadline_cache_path")
            self._deadline_cache = self._load_deadline_cache(self.deadline_cache_path)

    # ---------------- cache IO ----------------
    @staticmethod
    def _load_deadline_cache(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"deadline cache not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        validate_deadline_cache(obj, path)

        # 兼容两种结构：
        # 1) {"meta":..., "data":[{"seed":..,"wf_makespans":[..],...}, ...]}
        # 2) {"seed_to_record": {"1": {...}, "2": {...}}}
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

        # 可选一致性检查：dax 模板是否一致
        if dax_name is not None:
            names = rec.get("wf_dax_names", None)
            if isinstance(names, list) and wf_id < len(names):
                validate_dax_template_name(names[wf_id], dax_name, strict=self.deadline_cache_strict)

        return float(ms_list[wf_id])

    # ---------------- reset：每episode重置rng（保证seed绑定模板序列） ----------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.random_seed = int(seed)
        # 关键：episode 的模板抽样 rng 必须与 self.random_seed 绑定，否则 cache 对不上
        self.rng = np.random.RandomState(int(self.random_seed))
        return super().reset(seed=seed, options=options)

    # ---------------- 核心覆写：arrival时创建workflow并设置deadline ----------------
    def _add_workflow_if_arrived(self):
        added = False
        while (
            self.next_arrival_idx < len(self.arrival_times)
            and self.arrival_times[self.next_arrival_idx] <= self.current_time
        ):
            arr_t = self.arrival_times[self.next_arrival_idx]

            # 选择 dax 模板（与原 env 一致）
            if self.dax_probs is None:
                idx = self.rng.randint(len(self.dax_paths))
            else:
                idx = self.rng.choice(len(self.dax_paths), p=self.dax_probs)
            path = self.dax_paths[idx]
            dax_name = os.path.basename(str(path))

            # 解析DAX并随机化payload（与原 env 一致：seed = random_seed + arrival_idx）
            G, tasks = load_workflow_from_dax(
                path,
                seed=self.random_seed + self.next_arrival_idx,
                randomize_payloads=True,
            )
            if self.deadline_mode == "cache_fcfs":
                validate_cached_instance(self._deadline_cache, self.random_seed, len(self.workflows), G, tasks)

            wf = Workflow(
                workflow_id=len(self.workflows),
                graph=G,
                tasks=tasks,
                arrival_time=arr_t,
                source_device_id=0,
            )
            wf.is_arrived = True
            wf.dax_path = str(path)

            # rank 计算（与原 env 一致：均值资源估计）
            pc_mean = float(np.mean([self.vms[v].pc for v in self.vms]))
            bw_mean = float(np.mean([self.vms[v].bw for v in self.vms]))
            wf.compute_upward_ranks(pc_mean, bw_mean)
            wf.compute_downward_ranks(pc_mean, bw_mean)

            self.wf_remaining_tasks[wf.workflow_id] = len(tasks)

            # ========== 1) workflow deadline：cache 查表 or none占位 ==========
            alpha = float(self.deadline_alpha)

            if self.deadline_mode == "cache_fcfs":
                makespan_ref = self._cache_lookup_makespan(
                    episode_seed=int(self.random_seed),
                    wf_id=int(wf.workflow_id),
                    dax_name=dax_name,
                )
                wf.ref_makespan = float(makespan_ref)
                wf.deadline = float(arr_t + alpha * float(makespan_ref))

            elif self.deadline_mode == "none":
                # 占位：用很大的 deadline，避免 EDF/slack 特征崩
                wf.ref_makespan = None
                wf.deadline = float(arr_t + max(self.horizon, 1.0))

            else:
                raise ValueError(f"Unsupported deadline_mode: {self.deadline_mode}")

            # ========== 2) 任务子 deadline：由 wf.deadline 倒推（Latest-Finish 反推） ==========
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

            # Ready：入度=0
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
