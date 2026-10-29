# -*- coding: utf-8 -*-
"""
MARL-Q V4.2：仅向空闲 VM 分配与精简观测。

Ready 任务全局 FCFS 等待，选择空闲 VM 后立即开始，无 VM 等待队列。
复用公共任务、通信、LF 子截止期和 Host 任务窗口能耗定义。
固定源训练尺度；Global=task4+ready1+host2H，Local=task4+vm5M。
工作流截止期沿用离线 FCFS 参考缓存及配对混合 alpha，不改奖励公式。
"""

from common.f18_state_normalization import checked_observation, load_normalizer, observe_rr

import os
import json
import heapq
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import gymnasium as gym
except Exception:
    import gym

from common.workflow_opt import validate_deadline_cache, task_subdeadlines, sample_deadline_alpha
from common.workflow_opt import (
    LoadRecord,
    energy_from_records,
    Workflow,
    exec_time_components,
)
from common.read_xml_opt_Tsize import validate_cached_instance, load_workflow_from_dax, poisson_arrival_times, dax_template_id
from common.resource_opt import create_cluster


class CloudWorkflowEnv_MARL(gym.Env):
    def __init__(
        self,
        dax_paths,
        horizon=36000.0,
        arrival_lambda=0.1,
        random_seed=0,
        normalize=True,
        # 资源
        num_hosts=3,
        vms_per_host=(9, 8, 8),
        pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
        bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0),
        # 每 episode 工作流数
        workflows_per_episode=50,
        # reward 配置（对齐 HRL0.75：alpha_delay=0.75）
        energy_reward_scale=1e-3,
        task_baseline_norm=300.0,
        alpha_delay=0.75,
        # 观测参数
        max_ready_tasks=64,
        # ---------- deadline cache 对齐（A） ----------
        deadline_mode: str = "cache_fcfs",  # "cache_fcfs" or "poisson_heftlike"
        deadline_cache_path: Optional[str] = None,
        alpha_deadline: float = 1.5,  # 单一 ddlAlpha 的后备值
        strict_cache: bool = True,
        # ---------- 混合 ddlAlpha ----------
        alpha_deadline_choices: Optional[Tuple[float, ...]] = None,
        alpha_deadline_probs: Optional[Tuple[float, ...]] = None,
        # ---------- ddl 打印&落盘（强制） ----------
        dump_deadlines: bool = True,
        print_deadlines: bool = True,
        deadlines_out_dir: Optional[str] = None,
        deadlines_file_prefix: str = "marl",
        normalization_mode='legacy', normalization_config=None, source_train_scene=None,
        arrival_source='cache',
    ):
        super().__init__()
        self.state_normalizer = load_normalizer(normalization_mode, normalization_config, source_train_scene)
        self.dax_paths = list(dax_paths)
        self.dax_map = {}
        for path in self.dax_paths:
            key = dax_template_id(path)
            previous = self.dax_map.get(key)
            if previous is not None and os.path.normcase(os.path.abspath(previous)) != os.path.normcase(os.path.abspath(path)):
                raise ValueError(f"Ambiguous DAX template ID '{key}': {previous} and {path}")
            self.dax_map[key] = path
        self.horizon = float(horizon)
        self.arrival_lambda = float(arrival_lambda)
        if arrival_source not in ('cache', 'poisson'):
            raise ValueError("arrival_source must be 'cache' or 'poisson'")
        self.arrival_source = arrival_source
        self.random_seed = int(random_seed)
        self.normalize = bool(normalize)

        self.workflows_per_episode = int(workflows_per_episode) if workflows_per_episode is not None else None

        # reward 参数
        self.energy_reward_scale = float(energy_reward_scale)
        self.task_baseline_norm = float(task_baseline_norm)
        self.alpha_delay = float(alpha_delay)

        # deadline cache 参数
        self.deadline_mode = str(deadline_mode)
        self.deadline_cache_path = deadline_cache_path
        self.alpha_deadline = float(alpha_deadline)
        self.strict_cache = bool(strict_cache)

        # 混合 ddlAlpha 参数
        if (alpha_deadline_choices is None) != (alpha_deadline_probs is None):
            raise ValueError("alpha_deadline_choices and alpha_deadline_probs must be both None or both provided")

        if alpha_deadline_choices is not None:
            if len(alpha_deadline_choices) == 0:
                raise ValueError("alpha_deadline_choices must not be empty")
            if len(alpha_deadline_choices) != len(alpha_deadline_probs):
                raise ValueError("len(alpha_deadline_choices) != len(alpha_deadline_probs)")

            probs = np.asarray(alpha_deadline_probs, dtype=np.float64)
            if np.any(probs < 0):
                raise ValueError("alpha_deadline_probs must be non-negative")
            s = float(probs.sum())
            if s <= 0:
                raise ValueError("sum(alpha_deadline_probs) must be > 0")

            probs = probs / s
            self.alpha_deadline_choices = tuple(float(x) for x in alpha_deadline_choices)
            self.alpha_deadline_probs = tuple(float(x) for x in probs.tolist())
        else:
            self.alpha_deadline_choices = None
            self.alpha_deadline_probs = None

        # ddl dump 参数
        self.dump_deadlines = bool(dump_deadlines)
        self.print_deadlines = bool(print_deadlines)
        self.deadlines_out_dir = deadlines_out_dir
        self.deadlines_file_prefix = str(deadlines_file_prefix)

        # 集群
        self.hosts, self.vms = create_cluster(
            num_hosts=num_hosts,
            vms_per_host=vms_per_host,
            pc_tiers=pc_tiers,
            bw_tiers=bw_tiers,
        )
        self.host_ids = sorted(self.hosts.keys())
        self.vm_ids = sorted(self.vms.keys())
        self.num_hosts = len(self.host_ids)
        self.num_vms = len(self.vm_ids)

        # host -> vm_indices（在 vm_ids 序中的索引）
        self.host_vm_indices = {h: [] for h in self.host_ids}
        for j, vid in enumerate(self.vm_ids):
            self.host_vm_indices[self.vms[vid].host_id].append(j)
        self.max_vms_per_host = max(len(v) for v in self.host_vm_indices.values())

        self.max_ready = int(max_ready_tasks) if max_ready_tasks is not None else 64

        # V4: task(4), Ready count(1), mean speed/utilization per Host(2H).
        self.global_obs_dim = 5 + 2 * self.num_hosts
        self.global_act_dim = self.num_hosts

        self.block_task_dim = 4
        self.block_queue_dim = 0
        self.block_host_dim = 0
        self.vm_feat_dim = 5
        self.local_obs_dim = (
            self.block_task_dim
            + self.block_queue_dim
            + self.block_host_dim
            + self.vm_feat_dim * self.max_vms_per_host
        )
        self.local_act_dim = self.max_vms_per_host

        # deadline cache 一次性加载（A）
        self._deadline_cache_by_seed = None
        self._cache_seeds_sorted = None
        if self.deadline_mode == "cache_fcfs":
            if not self.deadline_cache_path:
                raise ValueError("deadline_mode=cache_fcfs but deadline_cache_path is None")
            self._load_deadline_cache_once(self.deadline_cache_path)

        # 运行时 RNG（reset(seed) 会重置）
        self.rng = np.random.RandomState(self.random_seed)
        self._reset_internal()

    @property
    def records(self):
        return self._records

    # =========================================================
    # ddlAlpha sampler
    # =========================================================
    def _sample_deadline_alpha(self, wf_id: int) -> float:
        if self.alpha_deadline_choices is None:
            return float(self.alpha_deadline)
        return sample_deadline_alpha(
            self.episode_seed, wf_id, self.alpha_deadline_choices, self.alpha_deadline_probs,
        )

    # =========================================================
    # cache loader (A)
    # =========================================================
    def _load_deadline_cache_once(self, path: str):
        if self._deadline_cache_by_seed is not None:
            return
        if not os.path.isfile(path):
            raise FileNotFoundError(f"deadline cache not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        validate_deadline_cache(obj, path)

        data = obj.get("data", [])
        cache = {}
        seeds = []
        for rec in data:
            for name in rec.get("wf_dax_names", []):
                if dax_template_id(name) not in self.dax_map:
                    raise FileNotFoundError(f"dax_name '{name}' not found in configured dax_paths")
            s = int(rec["seed"])
            cache[s] = rec
            seeds.append(s)
        seeds = sorted(set(seeds))
        self._deadline_cache_by_seed = cache
        self._cache_seeds_sorted = seeds

    def _pick_cache_seed(self, seed: int) -> int:
        """严格对齐时：必须 seed 存在；否则按 seeds_sorted 取模（可关闭 strict_cache）。"""
        seed = int(seed)
        if seed in self._deadline_cache_by_seed:
            return seed
        if self.strict_cache:
            raise KeyError(f"seed={seed} not found in deadline cache (strict_cache=True)")
        idx = int(seed) % max(1, len(self._cache_seeds_sorted))
        return int(self._cache_seeds_sorted[idx])

    # =========================================================
    # reset/step
    # =========================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        options = options or {}
        self.episode_id = int(options.get("episode_id", 0))
        self.episode_tag = str(options.get("tag", "train"))
        self.dump_deadlines = bool(options.get("dump_deadlines", self.dump_deadlines))
        self.print_deadlines = bool(options.get("print_deadlines", self.print_deadlines))

        if seed is None:
            self.episode_seed = int(self.random_seed)
        else:
            self.episode_seed = int(seed)

        self.rng = np.random.RandomState(self.episode_seed)

        self._reset_internal()
        self._prepare_deadline_log_paths()

        if self.deadline_mode == "cache_fcfs":
            cache_seed = self._pick_cache_seed(self.episode_seed)
            rec = self._deadline_cache_by_seed[cache_seed]
            arr = list(rec.get("arrival_times", []))
            dax_names = list(rec.get("wf_dax_names", []))
            makespans_ref = list(rec.get("wf_makespans", rec.get("wf_makespans_ref", [])))

            n = int(self.workflows_per_episode) if self.workflows_per_episode is not None else len(arr)
            self.arrival_times = [float(x) for x in arr[:n]]
            self._wf_dax_names_episode = dax_names[:n]
            self._wf_makespans_ref_episode = [float(x) for x in makespans_ref[:n]]

            if self.strict_cache:
                if len(self.arrival_times) != n or len(self._wf_dax_names_episode) != n or len(self._wf_makespans_ref_episode) != n:
                    raise ValueError(
                        f"cache record length mismatch: expect n={n}, "
                        f"arr={len(self.arrival_times)} dax={len(self._wf_dax_names_episode)} mref={len(self._wf_makespans_ref_episode)}"
                    )
            if self.arrival_source == 'poisson':
                # Regenerate only arrivals; templates and 0.03 FCFS durations stay paired.
                self.arrival_times = poisson_arrival_times(self.arrival_lambda, n_max=n, seed=self.episode_seed)
        else:
            self.arrival_times = poisson_arrival_times(
                self.arrival_lambda,
                n_max=self.workflows_per_episode,
                seed=self.episode_seed,
            )
            if self.workflows_per_episode is not None:
                n = int(self.workflows_per_episode)
                if n > 0 and len(self.arrival_times) > n:
                    self.arrival_times = self.arrival_times[:n]
            self._wf_dax_names_episode = None
            self._wf_makespans_ref_episode = None

        self.next_arrival_idx = 0

        _ = self._advance_until_decision()

        gobs, gmask = self.get_global_state()
        info = {
            "current_time": self.current_time,
            "wf_target": self.workflows_per_episode,
            "episode_seed": int(self.episode_seed),
        }
        return {"global_obs": gobs, "global_mask": gmask}, info

    def step(self, host_action: int, vm_action: int):
        """
        一个决策步：
        1) Global：为“当前任务(FCFS)”选择 Host（必须 host_mask=1）
        2) Local：在该 Host 内选真实且空闲的 VM
        3) 分配 -> 推进到下一个决策点 -> 计算能耗差分 + 延迟差分奖励
        """
        if self.done_flag:
            gobs, gmask = self.get_global_state()
            return {"global_obs": gobs, "global_mask": gmask}, 0.0, 0.0, True, {"current_time": self.current_time}

        if not self._has_decision_point():
            _ = self._advance_until_decision()
            if self.done_flag:
                gobs, gmask = self.get_global_state()
                return {"global_obs": gobs, "global_mask": gmask}, 0.0, 0.0, True, {"current_time": self.current_time}

        if self._cur_task_gid is None:
            self._update_current_task_fcfs()
        if self._cur_task_gid is None:
            _ = self._advance_until_decision()
            gobs, gmask = self.get_global_state()
            return {"global_obs": gobs, "global_mask": gmask}, 0.0, 0.0, bool(self.done_flag), {}

        tid = int(self._cur_task_gid)

        gmask = self._global_mask()
        if host_action < 0 or host_action >= self.num_hosts or gmask[host_action] <= 0.5:
            raise ValueError('Invalid MARL Host action')
        host_id = int(self.host_ids[int(host_action)])
        host_index = int(host_action)

        lobs, lmask, vm_map = self.get_local_state(host_id)
        if len(vm_map) == 0:
            rE_total, rE_by_host = self._advance_until_decision()
            gobs2, gmask2 = self.get_global_state()
            return {"global_obs": gobs2, "global_mask": gmask2}, float(rE_total), float(rE_total), bool(self.done_flag), {}

        if vm_action < 0 or vm_action >= self.local_act_dim or lmask[vm_action] <= 0.5:
            raise ValueError('Invalid MARL VM action')

        vm_index_global = int(vm_map[int(vm_action)])
        vm_id = int(self.vm_ids[vm_index_global])
        end_time = self._assign_task_to_vm(tid, vm_index_global)

        task_deadline = float(self.task_baseline_finish[tid]) if tid < len(self.task_baseline_finish) else float(end_time)
        lateness = max(0.0, float(end_time) - float(task_deadline))
        delay_norm = 0.0
        if self.task_baseline_norm > 0:
            delay_norm = min(1.0, lateness / max(self.task_baseline_norm, 1e-9))
        r_delay = -float(delay_norm)

        rE_total, rE_by_host = self._advance_until_decision()
        r_energy_total = float(rE_total)
        r_energy_host = float(rE_by_host.get(host_id, 0.0))

        a = float(self.alpha_delay)
        r_global = (1.0 - a) * r_energy_total + a * r_delay
        r_local = (1.0 - a) * r_energy_host + a * r_delay

        gobs2, gmask2 = self.get_global_state()
        done = bool(self.done_flag)

        wf_id, local_task_id = self.task_meta[tid] if tid < len(self.task_meta) else (-1, -1)
        wf_deadline = float(getattr(self.workflows[int(wf_id)], "deadline", 0.0)) if (0 <= int(wf_id) < len(self.workflows)) else 0.0
        wf_deadline_alpha = float(self.wf_deadline_alpha.get(int(wf_id), self.alpha_deadline))

        info = {
            "current_time": float(self.current_time),
            "episode_seed": int(self.episode_seed),

            "terminated": bool(done and self._episode_should_end()),
            "truncated": bool(done and not self._episode_should_end()),
            "assigned_task_id": int(tid),
            "wf_id": int(wf_id),
            "local_task_id": int(local_task_id),

            "host_id": int(host_id),
            "host_index": int(host_index),
            "vm_id": int(vm_id),
            "vm_index_global": int(vm_index_global),

            "task_end_time": float(end_time),
            "task_deadline": float(task_deadline),
            "wf_deadline": float(wf_deadline),
            "wf_deadline_alpha": float(wf_deadline_alpha),

            "lateness": float(lateness),
            "delay_norm": float(delay_norm),

            "r_energy_total": float(r_energy_total),
            "r_energy_host": float(r_energy_host),
            "total_energy_J": float(self.total_energy),

            "completed_wf": int(self.completed_workflows),
            "wf_target": int(self.workflows_per_episode) if self.workflows_per_episode is not None else None,
        }
        return {"global_obs": gobs2, "global_mask": gmask2}, float(r_global), float(r_local), done, info

    # =========================================================
    # 外部：state / metrics（E）
    # =========================================================
    @property
    def records(self):
        return self._records

    def get_global_state(self):
        obs = self._build_global_obs()
        mask = self._global_mask()
        return obs.astype(np.float32), mask.astype(np.float32)

    def get_local_state(self, host_id: int):
        if self.done_flag or not self._has_decision_point():
            return (np.zeros(self.local_obs_dim, np.float32),
                    np.zeros(self.local_act_dim, np.float32), [])
        if self._cur_task_gid is None:
            self._update_current_task_fcfs()
        tid = int(self._cur_task_gid)

        vm_indices = list(self.host_vm_indices[int(host_id)])
        vm_map = []
        for j in range(self.max_vms_per_host):
            vm_map.append(int(vm_indices[j]) if j < len(vm_indices) else -1)

        obs = self._build_local_obs(tid, int(host_id), vm_map)

        mask = np.zeros((self.local_act_dim,), dtype=np.float32)
        idle = self._vm_idle_mask()
        for slot, vmg in enumerate(vm_map):
            if vmg >= 0 and idle[vmg]:
                mask[slot] = 1.0

        return obs.astype(np.float32), mask.astype(np.float32), vm_map

    def get_episode_metrics(self) -> dict:
        n_tasks = len(self.task_end_time)
        if n_tasks == 0:
            return {
                "total_energy_J": float(self.total_energy),
                "makespan": float(0.0),
                "task_avg_lateness": 0.0,
                "task_avg_tardiness": 0.0,
                "pct_tardy": 0.0,
                "wf_avg_lateness": 0.0,
                "wf_avg_tardiness": 0.0,
                "wf_pct_tardy": 0.0,
            }

        t_first_arr = float(self.arrival_times[0]) if len(self.arrival_times) > 0 else 0.0
        makespan = float(self.current_time - t_first_arr)

        lateness_list = []
        tardy_list = []
        for gid in range(n_tasks):
            end_t = float(self.task_end_time[gid])
            ddl = float(self.task_baseline_finish[gid]) if gid < len(self.task_baseline_finish) else end_t
            late = end_t - ddl
            tardy = max(0.0, late)
            lateness_list.append(late)
            tardy_list.append(tardy)

        task_avg_late = float(np.mean(lateness_list))
        task_avg_tardy = float(np.mean(tardy_list))
        pct_tardy = float(np.mean([1.0 if x > 1e-12 else 0.0 for x in tardy_list]))

        wf_late = []
        wf_tardy = []
        for wf_id in range(len(self.workflows)):
            ends = []
            for gid, (w, _lid) in enumerate(self.task_meta):
                if int(w) == int(wf_id):
                    ends.append(float(self.task_end_time[gid]))
            if len(ends) == 0:
                continue
            wf_end = max(ends)
            wf_ddl = float(getattr(self.workflows[wf_id], "deadline", wf_end))
            late = wf_end - wf_ddl
            tardy = max(0.0, late)
            wf_late.append(late)
            wf_tardy.append(tardy)

        wf_avg_late = float(np.mean(wf_late)) if len(wf_late) > 0 else 0.0
        wf_avg_tardy = float(np.mean(wf_tardy)) if len(wf_tardy) > 0 else 0.0
        wf_pct_tardy = float(np.mean([1.0 if x > 1e-12 else 0.0 for x in wf_tardy])) if len(wf_tardy) > 0 else 0.0

        return {
            "total_energy_J": float(self.total_energy),
            "makespan": float(makespan),
            "task_avg_lateness": float(task_avg_late),
            "task_avg_tardiness": float(task_avg_tardy),
            "pct_tardy": float(pct_tardy),
            "wf_avg_lateness": float(wf_avg_late),
            "wf_avg_tardiness": float(wf_avg_tardy),
            "wf_pct_tardy": float(wf_pct_tardy),
        }

    # =========================================================
    # 内部：reset state
    # =========================================================
    def _reset_internal(self):
        self.current_time = 0.0
        self.done_flag = False

        self.workflows: List[Workflow] = []
        self.arrival_times: List[float] = []
        self.next_arrival_idx = 0

        self._wf_dax_names_episode = None
        self._wf_makespans_ref_episode = None

        self.task_meta: List[Tuple[int, int]] = []
        self.gid_of: Dict[Tuple[int, int], int] = {}

        self.task_state: List[str] = []
        self.task_ready_time: List[float] = []
        self.task_end_time: List[float] = []

        self.ready_task_ids: List[int] = []

        self.event_heap: List[Tuple[float, str, int, int]] = []
        self.vm_available_at = np.zeros(self.num_vms, dtype=np.float64)
        # available_at is the current task's finish time; busy VMs cannot accept work.
        self.vm_running_task = np.full(self.num_vms, -1, dtype=np.int64)

        self.completed_workflows = 0
        self.wf_remaining_tasks: Dict[int, int] = {}

        self.task_baseline_finish: List[float] = []
        self.wf_deadline_abs: Dict[int, float] = {}
        self.wf_dax_name: Dict[int, str] = {}
        self.wf_deadline_alpha: Dict[int, float] = {}

        self.parents_loc_by_wf: Dict[int, Dict[int, Tuple[int, int]]] = {}

        self._records: List[LoadRecord] = []
        self.total_energy = 0.0
        self.lifetime_energy = 0.0

        self._host_running_pc = {h: 0.0 for h in self.host_ids}
        self._energy_last_t = float(self.current_time)
        self._energy_by_host_J = {h: 0.0 for h in self.host_ids}
        self._energy_last_busy = {}

        self._cur_task_gid = None

        self._ddl_txt_path = None
        self._ddl_jsonl_path = None

        self._no_event_streak = 0
        self._no_event_limit = 10

    # =========================================================
    # decision point
    # =========================================================
    def _has_decision_point(self) -> bool:
        return bool(self.ready_task_ids) and bool(np.any(self._vm_idle_mask()))

    def _vm_idle_mask(self):
        return (self.vm_running_task < 0) & (self.vm_available_at <= self.current_time + 1e-12)

    def _global_mask(self):
        mask = np.zeros((self.num_hosts,), dtype=np.float32)
        idle = self._vm_idle_mask()
        for hi, h in enumerate(self.host_ids):
            idxs = self.host_vm_indices[h]
            if any(idle[j] for j in idxs):
                mask[hi] = 1.0
        return mask

    def _update_current_task_fcfs(self):
        if len(self.ready_task_ids) == 0:
            self._cur_task_gid = None
            return
        best = None
        for gid in self.ready_task_ids:
            rt = self.task_ready_time[gid]
            key = (rt, gid)
            if best is None or key < best[0]:
                best = (key, gid)
        self._cur_task_gid = int(best[1]) if best is not None else None

    # =========================================================
    # ddl dump helpers
    # =========================================================
    def _prepare_deadline_log_paths(self):
        if not self.dump_deadlines:
            self._ddl_txt_path = None
            self._ddl_jsonl_path = None
            return

        outdir = self.deadlines_out_dir
        if outdir is None:
            outdir = os.path.join(os.getcwd(), "outputs", "run", "marl", "deadlines")
        os.makedirs(outdir, exist_ok=True)

        base = f"{self.deadlines_file_prefix}_{self.episode_tag}_seed{int(self.episode_seed)}_ep{int(self.episode_id)}"
        self._ddl_txt_path = os.path.join(outdir, f"{base}.txt")
        self._ddl_jsonl_path = os.path.join(outdir, f"{base}.jsonl")

        with open(self._ddl_txt_path, "w", encoding="utf-8") as f:
            f.write(f"[DDL-DUMP] tag={self.episode_tag} seed={self.episode_seed} episode_id={self.episode_id}\n")
        with open(self._ddl_jsonl_path, "w", encoding="utf-8") as f:
            pass

    def _log_workflow_deadlines(
        self,
        wf_id: int,
        arrival_t: float,
        wf_deadline: float,
        dax_name: str,
        task_deadlines_abs: List[Tuple[int, float]],
        ddl_alpha: float,
        makespan_ref: Optional[float],
    ):
        # if self.print_deadlines:
        #     print(
        #         f"[DDL] wf={wf_id} dax={dax_name} arrival={arrival_t:.6f} "
        #         f"ddl_alpha={ddl_alpha:.6f} makespan_ref={float(makespan_ref) if makespan_ref is not None else -1.0:.6f} "
        #         f"wf_deadline={wf_deadline:.6f}"
        #     )
        #     for (local_tid, ddl) in task_deadlines_abs:
        #         print(f"  [DDL]   task local_id={int(local_tid)} task_deadline={float(ddl):.6f}")

        if not self.dump_deadlines:
            return

        if self._ddl_txt_path:
            with open(self._ddl_txt_path, "a", encoding="utf-8") as f:
                f.write(
                    f"[DDL] wf={wf_id} dax={dax_name} arrival={arrival_t:.6f} "
                    f"ddl_alpha={ddl_alpha:.6f} makespan_ref={float(makespan_ref) if makespan_ref is not None else -1.0:.6f} "
                    f"wf_deadline={wf_deadline:.6f}\n"
                )
                for (local_tid, ddl) in task_deadlines_abs:
                    f.write(f"  [DDL] task local_id={int(local_tid)} task_deadline={float(ddl):.6f}\n")

        if self._ddl_jsonl_path:
            rec = {
                "tag": self.episode_tag,
                "episode_id": int(self.episode_id),
                "episode_seed": int(self.episode_seed),
                "wf_id": int(wf_id),
                "dax_name": str(dax_name),
                "arrival_time": float(arrival_t),
                "ddl_alpha": float(ddl_alpha),
                "makespan_ref": None if makespan_ref is None else float(makespan_ref),
                "wf_deadline": float(wf_deadline),
                "task_deadlines": [{"local_task_id": int(tid), "task_deadline": float(dl)} for (tid, dl) in task_deadlines_abs],
            }
            with open(self._ddl_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # =========================================================
    # 时间推进（能耗差分）
    # =========================================================
    def _episode_should_end(self) -> bool:
        if self.workflows_per_episode is None:
            return (self.current_time >= self.horizon and len(self.event_heap) == 0 and self._no_more_arrivals())
        all_arrived = self._no_more_arrivals()
        all_done = self.completed_workflows >= self.workflows_per_episode
        no_running = len(self.event_heap) == 0
        no_ready = len(self.ready_task_ids) == 0
        return all_arrived and all_done and no_running and no_ready

    def _no_more_arrivals(self):
        return self.next_arrival_idx >= len(self.arrival_times)

    def _energy_rewards_to_current_time(self):
        t_now = float(self.current_time)
        t_last = float(self._energy_last_t)
        if t_now <= t_last:
            return 0.0, {h: 0.0 for h in self.host_ids}
        dE_by_host = {h: 0.0 for h in self.host_ids}
        for h in self.host_ids:
            running_pc = float(self._host_running_pc.get(h, 0.0))
            if running_pc <= 1e-12:
                continue
            host = self.hosts[h]
            ratio = max(0.0, min(1.0, running_pc / max(float(host.total_pc), 1e-9)))
            last_busy = self._energy_last_busy.get(h)
            # Count deferred idle only when a subsequent task makes it an internal gap.
            gap = max(0.0, t_last - last_busy) if last_busy is not None else 0.0
            dE_by_host[h] = float(host.power(ratio)) * (t_now - t_last) + float(host.power(0.0)) * gap
            self._energy_last_busy[h] = t_now
            self._energy_by_host_J[h] += dE_by_host[h]
        dE_total = sum(dE_by_host.values())
        self.total_energy += dE_total
        self.lifetime_energy += dE_total
        self._energy_last_t = t_now
        r_total = -dE_total * self.energy_reward_scale
        r_by_host = {h: -dE_by_host[h] * self.energy_reward_scale for h in self.host_ids}
        return float(r_total), r_by_host

    def _advance_until_decision(self):
        rE_total_acc = 0.0
        rE_host_acc = {h: 0.0 for h in self.host_ids}

        while True:
            self._process_finish_events()
            self._add_workflow_if_arrived()

            if self._episode_should_end() or self.current_time >= self.horizon:
                self.done_flag = True
                break

            if self._has_decision_point():
                self._no_event_streak = 0
                self._update_current_task_fcfs()
                break

            next_times = []
            if len(self.event_heap) > 0:
                next_times.append(self.event_heap[0][0])
            if not self._no_more_arrivals():
                next_times.append(self.arrival_times[self.next_arrival_idx])

            if len(next_times) == 0:
                if self._episode_should_end():
                    self.done_flag = True
                    break
                raise RuntimeError("Unfinished MARL episode has no arrival/completion event")

            next_t = min(next_times)
            self.current_time = float(min(next_t, self.horizon))

            r0, rb = self._energy_rewards_to_current_time()
            rE_total_acc += r0
            for h in self.host_ids:
                rE_host_acc[h] += rb[h]

        return float(rE_total_acc), rE_host_acc

    # =========================================================
    # workflow 到达
    # =========================================================
    def _add_workflow_if_arrived(self):
        added = False
        while (
            self.next_arrival_idx < len(self.arrival_times)
            and self.arrival_times[self.next_arrival_idx] <= self.current_time
        ):
            arr_t = float(self.arrival_times[self.next_arrival_idx])

            if self.deadline_mode == "cache_fcfs":
                dax_name = str(self._wf_dax_names_episode[self.next_arrival_idx])
                dax_path = self.dax_map.get(dax_template_id(dax_name))
                if dax_path is None:
                    raise FileNotFoundError(f"dax_name '{dax_name}' not found in configured dax_paths")
                dax_name = os.path.basename(str(dax_path))
                makespan_ref = float(self._wf_makespans_ref_episode[self.next_arrival_idx])
            else:
                idx = self.rng.randint(len(self.dax_paths))
                dax_path = self.dax_paths[idx]
                dax_name = os.path.basename(dax_path)
                makespan_ref = None

            payload_seed = int(self.episode_seed + self.next_arrival_idx)

            G, tasks = load_workflow_from_dax(
                dax_path,
                seed=payload_seed,
                randomize_payloads=True,
            )
            if self.deadline_mode == "cache_fcfs":
                validate_cached_instance(self._deadline_cache_by_seed, self._pick_cache_seed(self.episode_seed),
                                         len(self.workflows), G, tasks)

            wf_id = len(self.workflows)
            wf = Workflow(
                workflow_id=wf_id,
                graph=G,
                tasks=tasks,
                arrival_time=arr_t,
                source_device_id=0,
            )
            wf.is_arrived = True

            pc_mean = float(np.mean([self.vms[v].pc for v in self.vms])) if len(self.vms) > 0 else 1.0
            bw_mean = float(np.mean([self.vms[v].bw for v in self.vms])) if len(self.vms) > 0 else 1.0

            self.wf_remaining_tasks[wf_id] = len(tasks)

            # ===== 核心改动：每个 workflow 独立采样 ddlAlpha =====
            ddl_alpha_i = self._sample_deadline_alpha(wf_id)

            if self.deadline_mode == "cache_fcfs":
                wf.deadline = float(arr_t + ddl_alpha_i * makespan_ref)
            else:
                bw_mean_bps = bw_mean * 1e6
                fallback_durations = []
                for t in tasks:
                    in_bits_total = float((t.ext_in_bits or 0.0) + (t.from_parents_bits or 0.0))
                    dur = (
                        in_bits_total / max(bw_mean_bps, 1e-9)
                        + float(t.workload_mi) / max(pc_mean, 1.0)
                        + float(t.out_file_size_sum_bits) / max(bw_mean_bps, 1e-9)
                    )
                    fallback_durations.append(dur)
                M_avg = float(np.sum(fallback_durations))
                wf.deadline = float(arr_t + ddl_alpha_i * M_avg)

            self.wf_deadline_abs[wf_id] = float(wf.deadline)
            self.wf_dax_name[wf_id] = str(dax_name)
            self.wf_deadline_alpha[wf_id] = float(ddl_alpha_i)

            self.parents_loc_by_wf[wf_id] = {}

            for t in tasks:
                gid = len(self.task_meta)
                self.task_meta.append((wf_id, t.task_id))
                self.gid_of[(wf_id, t.task_id)] = gid

                self.task_state.append("unReady")
                self.task_ready_time.append(0.0)
                self.task_end_time.append(0.0)

            subdeadlines = task_subdeadlines(
                tasks,
                [self.vms[vid].pc for vid in self.vm_ids],
                [self.vms[vid].bw * 1e6 for vid in self.vm_ids],
                arr_t, wf.deadline,
            )
            self.task_baseline_finish.extend(subdeadlines.tolist())
            task_deadlines_abs = [(i, float(deadline)) for i, deadline in enumerate(subdeadlines)]

            self._log_workflow_deadlines(
                wf_id=int(wf_id),
                arrival_t=float(arr_t),
                wf_deadline=float(wf.deadline),
                dax_name=str(dax_name),
                task_deadlines_abs=task_deadlines_abs,
                ddl_alpha=float(ddl_alpha_i),
                makespan_ref=makespan_ref,
            )

            for local_id, t in enumerate(tasks):
                if len(t.parents) == 0:
                    gid = self.gid_of[(wf_id, local_id)]
                    self.task_state[gid] = "Ready"
                    self.task_ready_time[gid] = self.current_time
                    self.ready_task_ids.append(gid)

            self.workflows.append(wf)
            self.next_arrival_idx += 1
            added = True

        return added

    def _process_finish_events(self):
        while len(self.event_heap) > 0 and self.event_heap[0][0] <= self.current_time + 1e-12:
            event_time, etype, gid, vm_index_global = heapq.heappop(self.event_heap)
            vm = self.vms[self.vm_ids[vm_index_global]]
            host_id = int(vm.host_id)
            if etype != 'finish' or self.vm_running_task[vm_index_global] != gid:
                raise RuntimeError('Invalid MARL VM execution event')
            self.task_state[gid] = 'Finished'
            self.task_end_time[gid] = float(event_time)
            self.vm_running_task[vm_index_global] = -1
            self._host_running_pc[host_id] -= float(vm.pc)

            wf_id, local_tid = self.task_meta[gid]

            if wf_id in self.wf_remaining_tasks:
                self.wf_remaining_tasks[wf_id] -= 1
                if self.wf_remaining_tasks[wf_id] == 0:
                    self.completed_workflows += 1

            wf = self.workflows[int(wf_id)]
            task_obj = wf.tasks[int(local_tid)]
            for c_local in task_obj.children:
                c_gid = self.gid_of[(int(wf_id), int(c_local))]
                if self.task_state[c_gid] != "unReady":
                    continue
                ok = True
                child_obj = wf.tasks[int(c_local)]
                for p_local in child_obj.parents:
                    p_gid = self.gid_of[(int(wf_id), int(p_local))]
                    if self.task_state[p_gid] != "Finished":
                        ok = False
                        break
                if ok:
                    self.task_state[c_gid] = "Ready"
                    self.task_ready_time[c_gid] = self.current_time
                    self.ready_task_ids.append(c_gid)

        self.ready_task_ids = [x for x in self.ready_task_ids if self.task_state[x] == "Ready"]

    # =========================================================
    # 分配
    # =========================================================
    def _task_duration(self, task_id: int, vm_index: int) -> float:
        wf_id, local_id = self.task_meta[task_id]
        vm_id = self.vm_ids[vm_index]
        vm = self.vms[vm_id]
        return float(exec_time_components(
            self.workflows[int(wf_id)].tasks[int(local_id)],
            vm_pc_mi_s=float(vm.pc), vm_bw_Mb_s=float(vm.bw),
            parents_loc=self.parents_loc_by_wf.get(int(wf_id), {}),
            child_vm_id=int(vm_id), child_host_id=int(vm.host_id),
            zero_if_same_vm=True, zero_if_same_host=False,
        )[-1])

    def _assign_task_to_vm(self, gid: int, vm_index_global: int) -> float:
        if not 0 <= vm_index_global < self.num_vms or not self._vm_idle_mask()[vm_index_global]:
            raise ValueError('MARL task requires an idle VM')
        if not 0 <= gid < len(self.task_state) or self.task_state[gid] != 'Ready':
            raise ValueError('MARL task is not Ready')
        vid = int(self.vm_ids[vm_index_global])
        now = float(self.current_time)

        vm = self.vms[vid]
        host_id = int(vm.host_id)

        wf_id, local_task_id = self.task_meta[gid]
        wf_id = int(wf_id)
        local_task_id = int(local_task_id)


        tau_exec = self._task_duration(gid, vm_index_global)

        start_time = now
        end_time = float(start_time + float(tau_exec))

        self.task_state[gid] = 'Running'
        self.task_end_time[gid] = end_time
        self.vm_available_at[vm_index_global] = end_time

        if gid in self.ready_task_ids:
            self.ready_task_ids.remove(gid)

        self.vm_running_task[vm_index_global] = gid
        self._host_running_pc[host_id] += float(vm.pc)
        heapq.heappush(self.event_heap, (end_time, 'finish', gid, vm_index_global))

        self._records.append(LoadRecord(float(start_time), float(end_time), int(host_id), float(vm.pc)))

        self.parents_loc_by_wf[wf_id][local_task_id] = (int(vid), int(host_id))

        self._cur_task_gid = None
        return float(end_time)

    # =========================================================
    # 观测构建
    # =========================================================
    def _build_global_obs(self):
        if self.done_flag or not self.ready_task_ids:
            return np.zeros(self.global_obs_dim, dtype=np.float32)
        if self._cur_task_gid is None:
            self._update_current_task_fcfs()
        ready = len(self.ready_task_ids)
        if self.state_normalizer:
            observe_rr(self, ready)
            ready_value = self.state_normalizer.rr(ready)
        else:
            ready_value = ready / max(1, self.max_ready)
        hosts = [self._host_features(h) for h in self.host_ids]
        return checked_observation(self, np.concatenate([
            self._task_features(self._cur_task_gid), [ready_value], *hosts]).astype(np.float32))

    def _task_features(self, tid):
        wf_id, local_id = self.task_meta[tid]
        wf = self.workflows[int(wf_id)]
        task_obj = wf.tasks[int(local_id)]
        in_bits = float((task_obj.ext_in_bits or 0.0) + (task_obj.from_parents_bits or 0.0))
        out_bits = float(task_obj.out_file_size_sum_bits)
        mi = float(task_obj.workload_mi)
        return np.array([
            mi / self._obs_ref('workload_mi_ref', 1.),
            in_bits / self._obs_ref('input_bits_ref', 1.),
            out_bits / self._obs_ref('output_bits_ref', 1.),
            (wf.deadline - self.current_time) / self._obs_ref('T_ref', max(self.horizon, 1.)),
        ], dtype=np.float32)

    def _host_features(self, host_id):
        idxs = self.host_vm_indices[int(host_id)]
        mean_pc = np.mean([self.vms[self.vm_ids[j]].pc for j in idxs])
        return np.array([mean_pc / self._obs_ref('vm_pc_ref', 1.),
            self._host_running_pc[host_id] / self.hosts[host_id].total_pc], dtype=np.float32)

    def _build_local_obs(self, tid: int, host_id: int, vm_map):
        vm_block = np.zeros((self.max_vms_per_host, self.vm_feat_dim), dtype=np.float32)
        idle = self._vm_idle_mask()
        for slot, vmg in enumerate(vm_map):
            if vmg < 0:
                continue
            vm_id = self.vm_ids[vmg]
            vm = self.vms[vm_id]
            vm_block[slot] = np.array(
                [
                    vm.pc / self._obs_ref('vm_pc_ref', 1.),
                    vm.bw / self._obs_ref('vm_bw_ref', 1.),
                    float(idle[vmg]),
                    max(0., self.vm_available_at[vmg] - self.current_time) / self._obs_ref('T_ref', max(self.horizon, 1.)),
                    self._task_duration(tid, vmg) / self._obs_ref('T_ref', max(self.horizon, 1.)),
                ],
                dtype=np.float32,
            )

        obs = np.concatenate([self._task_features(tid), vm_block.reshape(-1)], axis=0)
        return checked_observation(self, obs)

    def _obs_ref(self, key, legacy):
        normalizer = getattr(self, 'state_normalizer', None)
        return normalizer.scales[key] if normalizer else legacy
