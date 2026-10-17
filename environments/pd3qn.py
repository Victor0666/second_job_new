# -*- coding: utf-8 -*-
"""PD3QN with DAG completion events and full-cluster episode energy.

The step energy reward includes assignment energy and all hosts' idle energy
added while advancing to the next decision or episode end. Legacy reward
parameters remain supported.
"""

import os
import json
from typing import Dict, Any, Tuple, List, Optional

import numpy as np

try:
    import gymnasium as gym
except Exception:
    import gym

from common.read_xml_opt_Tsize import load_workflow_from_dax, poisson_arrival_times
from common.resource_opt import create_cluster
from common.workflow_opt import Workflow, exec_time_components, LoadRecord


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _pick_fcfs_task(ready_task_ids: List[int], task_ready_time: List[float]) -> int:
    best = None
    best_key = None
    for gid in ready_task_ids:
        rt = float(task_ready_time[gid])
        key = (rt, int(gid))
        if (best_key is None) or (key < best_key):
            best_key = key
            best = int(gid)
    return int(best)


def _topo_sort_local(tasks) -> Tuple[List[int], List[List[int]]]:
    n = len(tasks)
    indeg = np.zeros((n,), dtype=np.int32)
    succ = [[] for _ in range(n)]
    for i, t in enumerate(tasks):
        for c in getattr(t, "children", []):
            succ[i].append(int(c))
        indeg[i] = len(getattr(t, "parents", []))
    q = [i for i in range(n) if indeg[i] == 0]
    topo = []
    while q:
        u = q.pop(0)
        topo.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(topo) != n:
        topo = list(range(n))
    return topo, succ


def _avg_task_dur_over_all_vms(tasks, vm_pc_mi_s: np.ndarray, vm_bw_bps: np.ndarray) -> np.ndarray:
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
    pc = vm_pc_mi_s.reshape(1, -1)

    dur = (
        in_bits.reshape(-1, 1) / np.maximum(bw, 1e-12)
        + mi.reshape(-1, 1) / np.maximum(pc, 1e-12)
        + out_bits.reshape(-1, 1) / np.maximum(bw, 1e-12)
    )
    return dur.mean(axis=1)


class CloudWorkflowEnv_PD3QN_Cost(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
            self,
            dax_paths: List[str],
            horizon: float = 1e9,
            arrival_lambda: float = 0.03,
            random_seed: int = 1,
            workflows_per_episode: int = 50,
            max_ready_tasks: Any = "auto",
            normalize: bool = True,

            # resources
            num_hosts: int = 3,
            vms_per_host: Tuple[int, ...] = (9, 8, 8),
            pc_tiers: Tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 8.0),
            bw_tiers: Tuple[float, ...] = (1000.0, 2000.0, 4000.0, 6000.0, 8000.0),

            # shallow-align reward params (推荐使用这组)
            alpha_delay_vm: float = 0.75,
            task_baseline_norm: float = 300.0,
            energy_reward_scale: float = 1e-3,

            # compatibility with old params
            w_delay: Optional[float] = None,
            w_energy: Optional[float] = None,
            delay_norm_ref: Optional[float] = None,
            energy_norm_per_mi_ref: float = 20.0,  # 兼容保留，本版 reward 中不再使用

            # deadline cache aligned
            deadline_mode: str = "cache_fcfs",
            deadline_cache_path: Optional[str] = None,
            deadline_alpha: float = 1.5,
            deadline_alpha_choices: Optional[List[float]] = None,
            deadline_alpha_probs: Optional[List[float]] = None,
            deadline_cache_strict: bool = True,

            # workflow 级 alpha 独立采样，不污染 self.rng
            deadline_alpha_seed_offset: int = 20260317,
    ):
        super().__init__()

        self.dax_paths = list(dax_paths)
        self.horizon = float(horizon)
        self.arrival_lambda = float(arrival_lambda)
        self.random_seed = int(random_seed)
        self.workflows_per_episode = int(workflows_per_episode)
        self.max_ready_tasks = max_ready_tasks
        self.normalize = bool(normalize)

        self.num_hosts = int(num_hosts)
        self.vms_per_host = tuple(int(x) for x in vms_per_host)
        self.pc_tiers = tuple(float(x) for x in pc_tiers)
        self.bw_tiers = tuple(float(x) for x in bw_tiers)

        # ---------------- reward config ----------------
        self.alpha_delay_vm = float(alpha_delay_vm)
        self.task_baseline_norm = float(task_baseline_norm)
        self.energy_reward_scale = float(energy_reward_scale)

        # 兼容旧字段
        self.energy_norm_per_mi_ref = float(energy_norm_per_mi_ref)

        if (w_delay is not None) or (w_energy is not None):
            wd = 0.0 if w_delay is None else float(w_delay)
            we = 0.0 if w_energy is None else float(w_energy)
            s = wd + we
            if s > 1e-12:
                self.alpha_delay_vm = float(wd / s)

        if delay_norm_ref is not None:
            self.task_baseline_norm = float(delay_norm_ref)

        # 兼容地保留 old-style 权重字段，便于外部调试打印
        self.w_delay = float(self.alpha_delay_vm)
        self.w_energy = float(1.0 - self.alpha_delay_vm)

        self.deadline_mode = str(deadline_mode)
        self.deadline_cache_path = deadline_cache_path
        self.deadline_alpha = float(deadline_alpha)
        self.deadline_cache_strict = bool(deadline_cache_strict)
        self._deadline_cache = None

        # 任意离散分布配置
        self.deadline_alpha_choices = None if deadline_alpha_choices is None else [float(x) for x in deadline_alpha_choices]
        self.deadline_alpha_probs = None if deadline_alpha_probs is None else [float(x) for x in deadline_alpha_probs]
        self.deadline_alpha_seed_offset = int(deadline_alpha_seed_offset)
        self._validate_deadline_alpha_config()

        if self.deadline_mode == "cache_fcfs":
            if not self.deadline_cache_path:
                raise ValueError("deadline_mode='cache_fcfs' 但未提供 deadline_cache_path")
            self._deadline_cache = self._load_deadline_cache(self.deadline_cache_path)

        # resources
        self.hosts, self.vms = create_cluster(
            num_hosts=self.num_hosts,
            vms_per_host=self.vms_per_host,
            pc_tiers=self.pc_tiers,
            bw_tiers=self.bw_tiers,
        )
        self.vm_ids = sorted(list(self.vms.keys()))
        self.num_vms = len(self.vm_ids)

        self._vm_pc = np.array([float(self.vms[i].pc) for i in self.vm_ids], dtype=np.float64)  # MI/s
        self._vm_bw_bps = np.array([float(self.vms[i].bw) * 1e6 for i in self.vm_ids], dtype=np.float64)  # bits/s

        self.action_space = gym.spaces.Discrete(self.num_vms)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # runtime RNG
        self.rng = np.random.RandomState(self.random_seed)

        # time / energy
        self.current_time = 0.0
        self.total_energy = 0.0
        self.done_flag = False

        # only count energy after the first workflow arrival time
        self.energy_window_started = False
        self.energy_window_start_time = 0.0

        # arrivals/episode
        self.arrival_times = []
        self.next_arrival_idx = 0
        self.completed_workflows = 0

        self.workflows: List[Workflow] = []
        self.wf_remaining_tasks: Dict[int, int] = {}
        self.wf_task_base: Dict[int, int] = {}

        # flattened task arrays
        self.task_meta: List[Tuple[int, int]] = []
        self.task_state: List[str] = []
        self.task_parents: List[List[int]] = []
        self.task_children: List[List[int]] = []
        self.task_mi: List[float] = []
        self.task_in_bits: List[float] = []
        self.task_out_bits: List[float] = []
        self.task_ready_time: List[float] = []
        self.task_end_time: List[float] = []

        # deadlines
        self.task_baseline_finish: List[float] = []

        # placement
        self.task_assigned_vm_id: List[int] = []
        self.task_assigned_host_id: List[int] = []

        self.ready_task_ids: List[int] = []
        self.vm_available_at = np.zeros((self.num_vms,), dtype=np.float64)

        self.records: List[LoadRecord] = []
        self._cur_task_gid: Optional[int] = None

        # host_id -> List[(start, end, load_pc_sum)]
        self._host_segments: Dict[int, List[Tuple[float, float, float]]] = {
            int(hid): [] for hid in self.hosts.keys()
        }

        obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs.shape[0],), dtype=np.float32)

    # ---------------- deadline alpha config ----------------
    def _validate_deadline_alpha_config(self):
        if self.deadline_alpha_choices is None and self.deadline_alpha_probs is None:
            return

        if self.deadline_alpha_choices is None or self.deadline_alpha_probs is None:
            raise ValueError("deadline_alpha_choices 和 deadline_alpha_probs 必须同时提供，或同时不提供")

        if len(self.deadline_alpha_choices) == 0:
            raise ValueError("deadline_alpha_choices 不能为空")

        if len(self.deadline_alpha_choices) != len(self.deadline_alpha_probs):
            raise ValueError("deadline_alpha_choices 和 deadline_alpha_probs 长度必须一致")

        probs = np.asarray(self.deadline_alpha_probs, dtype=np.float64)
        if np.any(probs < 0):
            raise ValueError("deadline_alpha_probs 不能包含负数")

        s = float(probs.sum())
        if s <= 0.0:
            raise ValueError("deadline_alpha_probs 之和必须大于 0")

        probs = probs / s
        self.deadline_alpha_probs = probs.tolist()

    def _sample_deadline_alpha(self, wf_id: int) -> float:
        """
        若提供离散分布，则按该分布对每个 workflow 独立采样；
        但不能使用 self.rng，否则会污染 DAX 模板抽样序列，导致与 deadline cache 错位。
        """
        if self.deadline_alpha_choices is None:
            return float(self.deadline_alpha)

        local_seed = (
                             int(self.random_seed) * 1000003
                             + int(wf_id) * 9176
                             + int(self.deadline_alpha_seed_offset)
                     ) & 0xFFFFFFFF

        rs = np.random.RandomState(local_seed)
        idx = int(rs.choice(len(self.deadline_alpha_choices), p=self.deadline_alpha_probs))
        return float(self.deadline_alpha_choices[idx])

    # ---------------- cache IO ----------------
    @staticmethod
    def _load_deadline_cache(path: str) -> Dict[int, Dict[str, Any]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"deadline cache not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        cache: Dict[int, Dict[str, Any]] = {}
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            for rec in obj["data"]:
                cache[int(rec["seed"])] = rec
        elif isinstance(obj, dict) and "seed_to_record" in obj:
            for k, rec in obj["seed_to_record"].items():
                cache[int(k)] = rec
        else:
            raise ValueError("deadline cache json 格式不正确：必须包含 data(list) 或 seed_to_record(dict)")
        return cache

    def _cache_lookup_makespan(self, episode_seed: int, wf_id: int, dax_name: Optional[str] = None) -> float:
        assert self._deadline_cache is not None
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

    # ---------------- incremental energy helpers ----------------
    @staticmethod
    def _merge_segments(segs: List[Tuple[float, float, float]], eps: float = 1e-12) -> List[Tuple[float, float, float]]:
        if not segs:
            return segs
        segs_sorted = sorted(segs, key=lambda x: (x[0], x[1]))
        out = [segs_sorted[0]]
        for s, e, l in segs_sorted[1:]:
            ps, pe, pl = out[-1]
            if (abs(l - pl) <= 1e-12) and (s <= pe + eps):
                out[-1] = (ps, max(pe, e), pl)
            else:
                out.append((s, e, l))
        cleaned = []
        for s, e, l in out:
            if e > s + eps:
                cleaned.append((s, e, l))
        return cleaned

    def _seg_energy(self, host_id: int, load_pc: float, dur: float) -> float:
        host = self.hosts[int(host_id)]
        ratio = float(load_pc) / max(float(host.total_pc), 1e-12)
        ratio = max(0.0, min(1.0, ratio))
        return float(host.power(ratio) * float(dur))

    def _add_interval_load(self, host_id: int, start: float, end: float, delta_pc: float) -> float:
        start = float(start)
        end = float(end)
        delta_pc = float(delta_pc)
        if end <= start + 1e-12:
            return 0.0

        segs = self._host_segments[int(host_id)]
        new_segs: List[Tuple[float, float, float]] = []

        i = 0
        while i < len(segs) and segs[i][1] <= start + 1e-12:
            new_segs.append(segs[i])
            i += 1

        cur = start
        dE = 0.0

        while cur < end - 1e-12:
            if i >= len(segs) or segs[i][0] >= end - 1e-12:
                gap_end = end if i >= len(segs) else min(end, segs[i][0])
                load_new = 0.0 + delta_pc
                new_segs.append((cur, gap_end, load_new))
                dE += self._seg_energy(host_id, load_new, gap_end - cur)
                cur = gap_end
                continue

            s0, e0, l0 = segs[i]

            if s0 > cur + 1e-12:
                gap_end = min(end, s0)
                load_new = 0.0 + delta_pc
                new_segs.append((cur, gap_end, load_new))
                dE += self._seg_energy(host_id, load_new, gap_end - cur)
                cur = gap_end
                continue

            if s0 < cur - 1e-12:
                new_segs.append((s0, cur, l0))

            overlap_end = min(end, e0)
            l_new = l0 + delta_pc

            dE += self._seg_energy(host_id, l_new, overlap_end - cur) - self._seg_energy(host_id, l0, overlap_end - cur)
            new_segs.append((cur, overlap_end, l_new))

            cur = overlap_end

            if overlap_end < e0 - 1e-12:
                new_segs.append((overlap_end, e0, l0))

            i += 1

        while i < len(segs):
            new_segs.append(segs[i])
            i += 1

        new_segs = self._merge_segments(new_segs)
        self._host_segments[int(host_id)] = new_segs

        self.total_energy = float(self.total_energy + dE)
        return float(dE)

    # ---------------- reset ----------------
    def reset(self, *, seed=None, options=None):
        self.rng = np.random.RandomState(int(self.random_seed))

        self.current_time = 0.0
        self.total_energy = 0.0
        self.done_flag = False

        try:
            self.arrival_times = poisson_arrival_times(
                self.arrival_lambda, self.horizon, seed=int(self.random_seed), n_max=int(self.workflows_per_episode)
            )
        except TypeError:
            self.arrival_times = poisson_arrival_times(self.arrival_lambda, self.horizon, seed=int(self.random_seed))

        if self.workflows_per_episode is not None and len(self.arrival_times) > self.workflows_per_episode:
            self.arrival_times = self.arrival_times[: self.workflows_per_episode]

        self.next_arrival_idx = 0
        self.completed_workflows = 0

        self.workflows = []
        self.wf_remaining_tasks = {}
        self.wf_task_base = {}

        self.task_meta = []
        self.task_state = []
        self.task_parents = []
        self.task_children = []
        self.task_mi = []
        self.task_in_bits = []
        self.task_out_bits = []
        self.task_ready_time = []
        self.task_end_time = []
        self.task_baseline_finish = []

        self.task_assigned_vm_id = []
        self.task_assigned_host_id = []

        self.ready_task_ids = []

        self.vm_available_at[:] = 0.0
        self.records = []
        self._cur_task_gid = None

        self._host_segments = {int(hid): [] for hid in self.hosts.keys()}

        t_first_arr = float(self.arrival_times[0]) if len(self.arrival_times) > 0 else 0.0

        self.energy_window_started = False
        self.energy_window_start_time = t_first_arr

        self.current_time = t_first_arr
        self.total_energy = 0.0

        self.records = []
        self._host_segments = {int(hid): [] for hid in self.hosts.keys()}

        self._observe_until_decision()
        obs = self._build_obs()
        return obs, {}

    # ---------------- core loop helpers ----------------
    def _observe_until_decision(self):
        while True:
            self._process_completions()
            self._add_workflow_if_arrived()

            if self._has_decision():
                self._cur_task_gid = _pick_fcfs_task(self.ready_task_ids, self.task_ready_time)
                return

            if self._check_done():
                self.done_flag = True
                self._cur_task_gid = None
                return

            next_t = self._next_event_time()
            self._advance_time(next_t)

    def _has_decision(self) -> bool:
        if len(self.ready_task_ids) == 0:
            return False
        return bool(np.any(self.vm_available_at <= self.current_time + 1e-12))

    def _next_event_time(self) -> float:
        t_candidates = []
        if self.next_arrival_idx < len(self.arrival_times):
            t_candidates.append(float(self.arrival_times[self.next_arrival_idx]))
        t_candidates.extend(
            float(self.task_end_time[gid])
            for gid, state in enumerate(self.task_state) if state == "Running"
        )
        t_candidates = [t for t in t_candidates if t > self.current_time]
        if len(t_candidates) == 0:
            raise RuntimeError("Unfinished tasks remain but no arrival or completion event is pending")
        return float(min(t_candidates))

    def _advance_time(self, new_t: float):
        new_t = float(new_t)
        if new_t <= self.current_time:
            return

        if not self.energy_window_started:
            self.current_time = new_t
            return

        for hid in self.hosts.keys():
            hid = int(hid)
            self.records.append(LoadRecord(start_time=self.current_time, end_time=new_t, server_id=hid, vm_pc=0.0))
            self._add_interval_load(hid, self.current_time, new_t, 0.0)

        self.current_time = new_t

    def _check_done(self) -> bool:
        return (
            self.next_arrival_idx >= len(self.arrival_times)
            and not self.ready_task_ids
            and all(state == "Finished" for state in self.task_state)
        )

    def _process_completions(self):
        for gid, state in enumerate(self.task_state):
            if state != "Running" or self.task_end_time[gid] > self.current_time:
                continue
            self.task_state[gid] = "Finished"
            self._unlock_children(gid)
            wf_id, _ = self.task_meta[gid]
            self.wf_remaining_tasks[int(wf_id)] -= 1
            if self.wf_remaining_tasks[int(wf_id)] == 0:
                self.completed_workflows += 1

    # ---------------- add workflow with cache deadline ----------------
    def _add_workflow_if_arrived(self):
        while self.next_arrival_idx < len(self.arrival_times) and self.arrival_times[self.next_arrival_idx] <= self.current_time + 1e-12:
            arr_t = float(self.arrival_times[self.next_arrival_idx])

            if not self.energy_window_started:
                self.energy_window_started = True
                self.energy_window_start_time = float(arr_t)
                if self.current_time < self.energy_window_start_time - 1e-12:
                    self.current_time = self.energy_window_start_time

            idx = int(self.rng.randint(len(self.dax_paths)))
            path = self.dax_paths[idx]
            dax_name = os.path.basename(str(path))

            G, tasks = load_workflow_from_dax(
                path,
                seed=int(self.random_seed + self.next_arrival_idx),
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

            pc_mean = float(np.mean(self._vm_pc))
            bw_mean = float(np.mean([self.vms[v].bw for v in self.vm_ids]))
            wf.compute_upward_ranks(pc_mean, bw_mean)
            wf.compute_downward_ranks(pc_mean, bw_mean)

            self.wf_remaining_tasks[wf.workflow_id] = len(tasks)

            if self.deadline_mode == "cache_fcfs":
                makespan_ref = self._cache_lookup_makespan(
                    episode_seed=int(self.random_seed),
                    wf_id=int(wf.workflow_id),
                    dax_name=dax_name,
                )
                wf.ref_makespan = float(makespan_ref)

                alpha = float(self._sample_deadline_alpha(wf.workflow_id))
                wf.deadline_alpha_used = alpha
                wf.deadline = float(arr_t + alpha * float(makespan_ref))

            elif self.deadline_mode == "none":
                wf.ref_makespan = None
                wf.deadline_alpha_used = None
                wf.deadline = float(arr_t + max(self.horizon, 1.0))
            else:
                raise ValueError(f"Unsupported deadline_mode: {self.deadline_mode}")

            n_local = len(tasks)
            topo, succ = _topo_sort_local(tasks)
            avg_dur = _avg_task_dur_over_all_vms(tasks, self._vm_pc, self._vm_bw_bps)
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

            base = len(self.task_meta)
            self.wf_task_base[int(wf.workflow_id)] = int(base)

            for local_id in range(n_local):
                self.task_baseline_finish.append(float(arr_t + LF[local_id]))

            for t in tasks:
                self.task_meta.append((wf.workflow_id, int(t.task_id)))
                self.task_state.append("unReady")
                self.task_parents.append(list(getattr(t, "parents", [])))
                self.task_children.append(list(getattr(t, "children", [])))
                self.task_mi.append(float(t.workload_mi))
                in_bits_total = float((t.ext_in_bits or 0.0) + (t.from_parents_bits or 0.0))
                self.task_in_bits.append(in_bits_total)
                self.task_out_bits.append(float(t.out_file_size_sum_bits))
                self.task_ready_time.append(0.0)
                self.task_end_time.append(0.0)

                self.task_assigned_vm_id.append(-1)
                self.task_assigned_host_id.append(-1)

            for local_id, t in enumerate(tasks):
                if len(getattr(t, "parents", [])) == 0:
                    gid = base + local_id
                    self.task_state[gid] = "Ready"
                    self.task_ready_time[gid] = float(self.current_time)
                    self.ready_task_ids.append(int(gid))

            self.workflows.append(wf)
            self.next_arrival_idx += 1

    # ---------------- observation / mask ----------------
    def _build_obs(self) -> np.ndarray:
        now = float(self.current_time)
        qsize = float(len(self.ready_task_ids))

        if self._cur_task_gid is None:
            task_feat = np.zeros((4,), dtype=np.float32)
        else:
            gid = int(self._cur_task_gid)
            mi = float(self.task_mi[gid])
            inb = float(self.task_in_bits[gid])
            outb = float(self.task_out_bits[gid])
            ddl = float(self.task_baseline_finish[gid]) if gid < len(self.task_baseline_finish) else (now + 1e9)
            slack = float(ddl - now)
            task_feat = np.array([mi, inb, outb, slack], dtype=np.float32)

        av_in = np.maximum(self.vm_available_at - now, 0.0).astype(np.float32)
        pc = self._vm_pc.astype(np.float32)
        bw = (self._vm_bw_bps / 1e6).astype(np.float32)
        vm_feat = np.stack([av_in, pc, bw], axis=1).reshape(-1).astype(np.float32)

        obs = np.concatenate([np.array([now, qsize], dtype=np.float32), task_feat, vm_feat], axis=0)
        if self.normalize:
            obs = obs / (np.linalg.norm(obs) + 1e-6)
        return obs.astype(np.float32)

    def get_action_mask(self) -> np.ndarray:
        if self._cur_task_gid is None:
            return np.zeros((self.num_vms,), dtype=np.float32)
        idle = (self.vm_available_at <= self.current_time + 1e-12)
        return idle.astype(np.float32)

    # ---------------- helpers for parents_loc ----------------
    def _get_task_obj_by_gid(self, gid: int):
        wf_id, local_id = self.task_meta[int(gid)]
        wf = self.workflows[int(wf_id)]
        return wf.tasks[int(local_id)]

    def _build_parents_loc(self, gid: int) -> Dict[int, Tuple[int, int]]:
        wf_id, local_id = self.task_meta[int(gid)]
        wf_id = int(wf_id)
        local_id = int(local_id)
        base = int(self.wf_task_base.get(wf_id, 0))
        wf = self.workflows[wf_id]

        parents_loc: Dict[int, Tuple[int, int]] = {}
        for p_local in getattr(wf.tasks[local_id], "parents", []):
            p_local = int(p_local)
            p_gid = base + p_local
            if p_gid < 0 or p_gid >= len(self.task_state):
                continue
            if self.task_state[p_gid] != "Finished":
                continue
            p_vm = int(self.task_assigned_vm_id[p_gid])
            p_host = int(self.task_assigned_host_id[p_gid])
            if p_vm >= 0 and p_host >= 0:
                parents_loc[p_local] = (p_vm, p_host)
        return parents_loc

    def _unlock_children(self, gid: int):
        wf_id, local_id = self.task_meta[int(gid)]
        wf_id = int(wf_id)
        local_id = int(local_id)
        wf = self.workflows[wf_id]
        base = int(self.wf_task_base.get(wf_id, 0))

        children_local = getattr(wf.tasks[local_id], "children", None)
        if not children_local:
            return

        for c_local in children_local:
            c_local = int(c_local)
            c_gid = int(base + c_local)
            if self.task_state[c_gid] != "unReady":
                continue

            ok = True
            for p_local in self.task_parents[c_gid]:
                p_gid = int(base + int(p_local))
                if self.task_state[p_gid] != "Finished":
                    ok = False
                    break
            if ok:
                if self.task_state[c_gid] != "Ready":
                    self.task_state[c_gid] = "Ready"
                    self.task_ready_time[c_gid] = float(self.current_time)
                    if c_gid not in self.ready_task_ids:
                        self.ready_task_ids.append(c_gid)

    # ---------------- step ----------------
    def step(self, action: int):
        if self.done_flag or self._cur_task_gid is None:
            self.done_flag = True
            return self._build_obs(), 0.0, True, False, {}

        gid = int(self._cur_task_gid)
        mask = self.get_action_mask()

        if int(action) < 0 or int(action) >= self.num_vms or mask[int(action)] <= 0.5:
            valid = np.flatnonzero(mask > 0.5)
            if valid.size == 0:
                self._observe_until_decision()
                return self._build_obs(), 0.0, self.done_flag, False, {}
            action = int(valid[0])

        energy_before = float(self.total_energy)
        vm_idx = int(action)
        vm_id = int(self.vm_ids[vm_idx])
        vm = self.vms[vm_id]
        host_id = int(vm.host_id)

        parents_loc = self._build_parents_loc(gid)

        tau_in, tau_comp, tau_out, tau_exec = exec_time_components(
            task=self._get_task_obj_by_gid(gid),
            vm_pc_mi_s=float(vm.pc),
            vm_bw_Mb_s=float(vm.bw),
            parents_loc=parents_loc,
            child_vm_id=None,
            child_host_id=None,
            zero_if_same_vm=True,
            zero_if_same_host=True,
        )

        wf_id, _ = self.task_meta[gid]
        base = self.wf_task_base[int(wf_id)]
        parents_end = max(
            (self.task_end_time[base + int(p)] for p in self.task_parents[gid]),
            default=0.0,
        )
        start_t = float(max(self.current_time, self.vm_available_at[vm_idx], parents_end))
        end_t = float(start_t + tau_exec)

        self.records.append(LoadRecord(start_time=start_t, end_time=end_t, server_id=int(host_id), vm_pc=float(vm.pc)))
        self.vm_available_at[vm_idx] = end_t

        self.task_state[gid] = "Running"
        self.task_end_time[gid] = end_t

        self.task_assigned_vm_id[gid] = int(vm_id)
        self.task_assigned_host_id[gid] = int(host_id)

        if gid in self.ready_task_ids:
            self.ready_task_ids.remove(gid)

        # Preserve the selected-host assignment increment for diagnostics.
        dE_host = self._add_interval_load(host_id, start_t, end_t, float(vm.pc))

        ddl = float(self.task_baseline_finish[gid]) if gid < len(self.task_baseline_finish) else (end_t + 1e9)
        tardiness = float(max(0.0, end_t - ddl))

        # 对齐 HRL 下层 delay reward
        delay_norm = _clip01(tardiness / max(self.task_baseline_norm, 1e-9))
        r_delay = -float(delay_norm)

        self._observe_until_decision()
        dE_total = float(self.total_energy - energy_before)
        r_energy_total = -dE_total * float(self.energy_reward_scale)
        r_energy_host = -float(dE_host) * float(self.energy_reward_scale)
        reward = (1.0 - self.alpha_delay_vm) * r_energy_total + self.alpha_delay_vm * r_delay

        info = {
            "time": float(self.current_time),
            "energy": float(self.total_energy),
            "delta_energy": float(dE_host),
            "delta_energy_total": float(dE_total),
            "task_gid": int(gid),
            "task_end": float(end_t),
            "task_deadline": float(ddl),
            "task_tardiness": float(tardiness),
            "completed_workflows": int(self.completed_workflows),

            "delay_norm": float(delay_norm),
            "r_delay": float(r_delay),
            "r_energy_host": float(r_energy_host),
            "r_energy_total": float(r_energy_total),
            "alpha_delay_vm": float(self.alpha_delay_vm),
            "reward_mode": "full_cluster_energy",
        }
        return self._build_obs(), float(reward), bool(self.done_flag), False, info

    # ---------------- metrics ----------------
    def get_episode_metrics(self) -> Dict[str, float]:
        makespan = float(self.current_time)
        total_energy = float(self.total_energy)

        task_total_late = 0.0
        task_total_tardy = 0.0
        task_done_cnt = 0
        task_tardy_cnt = 0

        n_tasks = len(self.task_state)
        for gid in range(n_tasks):
            if self.task_state[gid] != "Finished":
                continue
            if gid >= len(self.task_baseline_finish):
                continue

            finish_t = float(self.task_end_time[gid])
            ddl_t = float(self.task_baseline_finish[gid])

            late = float(finish_t - ddl_t)
            tard = float(max(0.0, late))

            task_total_late += late
            task_total_tardy += tard
            task_done_cnt += 1
            if tard > 1e-12:
                task_tardy_cnt += 1

        task_avg_late = task_total_late / max(task_done_cnt, 1)
        task_avg_tardy = task_total_tardy / max(task_done_cnt, 1)
        task_pct_tardy = task_tardy_cnt / max(task_done_cnt, 1)
        task_success_rate = 1.0 - task_pct_tardy

        wf_total_late = 0.0
        wf_total_tardy = 0.0
        wf_done_cnt = 0
        wf_tardy_cnt = 0

        for wf in self.workflows:
            wf_id = int(wf.workflow_id)
            base = int(self.wf_task_base[wf_id])
            n_local = len(wf.tasks)

            all_finished = True
            wf_finish_t = 0.0

            for local_id in range(n_local):
                gid = base + local_id
                if self.task_state[gid] != "Finished":
                    all_finished = False
                    break
                wf_finish_t = max(wf_finish_t, float(self.task_end_time[gid]))

            if not all_finished:
                continue

            wf_deadline = float(wf.deadline)
            wf_late = float(wf_finish_t - wf_deadline)
            wf_tard = float(max(0.0, wf_late))

            wf_total_late += wf_late
            wf_total_tardy += wf_tard
            wf_done_cnt += 1
            if wf_tard > 1e-12:
                wf_tardy_cnt += 1

        wf_avg_late = wf_total_late / max(wf_done_cnt, 1)
        wf_avg_tardy = wf_total_tardy / max(wf_done_cnt, 1)
        wf_pct_tardy = wf_tardy_cnt / max(wf_done_cnt, 1)
        wf_success_rate = 1.0 - wf_pct_tardy

        return {
            "makespan": float(makespan),
            "total_energy": float(total_energy),

            "num_tasks": float(task_done_cnt),
            "num_tardy_tasks": float(task_tardy_cnt),
            "task_total_lateness": float(task_total_late),
            "task_total_tardiness": float(task_total_tardy),
            "task_avg_lateness": float(task_avg_late),
            "task_avg_tardiness": float(task_avg_tardy),
            "pct_tardy": float(task_pct_tardy),
            "task_pct_tardy": float(task_pct_tardy),
            "task_success_rate": float(task_success_rate),

            "num_workflows": float(wf_done_cnt),
            "num_tardy_workflows": float(wf_tardy_cnt),
            "wf_lateness_sum": float(wf_total_late),
            "wf_tardiness_sum": float(wf_total_tardy),
            "wf_avg_lateness": float(wf_avg_late),
            "wf_avg_tardiness": float(wf_avg_tardy),
            "wf_pct_tardy": float(wf_pct_tardy),
            "wf_success_rate": float(wf_success_rate),
        }