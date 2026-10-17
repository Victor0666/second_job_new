# -*- coding: utf-8 -*-
"""
algorithms.irws/env_irws.py

IRWS 环境（Route-A）——对齐 HRL 公平口径版本（6点全对齐）：

(1) deadline cache_fcfs + 子deadline Latest-Finish 完全一致：
    wf.deadline = arrival + alpha * makespan_ref  (ref 来自离线 FCFS+VM-first cache JSON)
    task_baseline_finish 使用 avg_dur(所有VM平均时长) + Latest-Finish 反推得到绝对deadline

(2) RNG/arrival/payload 规则一致：
    - reset() 绑定 self.rng = np.random.RandomState(self.random_seed)
    - 模板抽样仅用 self.rng
    - payload seed = random_seed + arrival_idx
    - arrival_times 使用 poisson_arrival_times(lmbda, horizon, seed, n_max=workflows_per_episode)

(3) 能耗记账一致（含 idle）：
    - 任务执行段写 LoadRecord(start,end,host_id,vm_pc)
    - 时间推进段对每个 host 写 0-load LoadRecord（idle energy 计入）
    - episode end 可做 EnergyCheck：env.total_energy == energy_from_records(records,hosts)

(4) max_ready_tasks 候选集一致：
    - 支持 max_ready_tasks="auto"（按 workflows_per_episode * max_task_count 推断）
    - ready candidates 先按 (ready_time, gid) 排序，再截断到 max_ready_tasks

(5) 训练预算按 assignment：
    - 每次成功分配一个 task 到某个 VM 记为 1 个 assignment step（训练脚本对齐）

(6) 评测 deterministic + 同 seeds + 同 metrics：
    - env.get_episode_metrics() 输出：total_energy, makespan, task_avg_lateness/tardiness, pct_tardy, wf_avg_tardy 等

新增：
- 支持“每个 workflow 独立采样 ddlAlpha”
- 使用独立的 deadline_rng，避免打乱 self.rng 的 DAX 模板抽样序列

本版关键修改：
- 环境 reward 改为与“修改后的 PD3QN reward”相同（浅对齐版）：

    r_delay       = - clip( max(0, finish_time - task_deadline) / task_baseline_norm , 0, 1 )
    r_energy_host = - ΔE_host * energy_reward_scale
    reward        = (1-alpha_delay_vm) * r_energy_host + alpha_delay_vm * r_delay

其中：
- ΔE_host 为本次 assignment 在所选 host 上新增的真实增量能耗
- 不再使用旧版 e_proxy / MI / energy_norm_per_mi_ref
"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

try:
    import gymnasium as gym
except Exception:
    import gym

from common.read_xml_opt_Tsize import load_workflow_from_dax, poisson_arrival_times
from common.workflow_opt import Workflow, exec_time_components, LoadRecord, energy_from_records
from common.resource_opt import create_cluster


# ---------------- helpers ----------------

def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _avg_task_dur_over_all_vms(tasks, vm_pc_mi_s: np.ndarray, vm_bw_Mb_s: np.ndarray) -> np.ndarray:
    """
    与 HRL cache_fcfs env 同口径：
        dur(i,p)=in/bw + mi/pc + out/bw
    其中 bw 换算为 bps=bw*1e6
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

    pc = vm_pc_mi_s.reshape(1, -1)                   # [1,P]
    bw_bps = (vm_bw_Mb_s * 1e6).reshape(1, -1)       # [1,P]

    dur = (
        in_bits.reshape(-1, 1) / np.maximum(bw_bps, 1e-12)
        + mi.reshape(-1, 1) / np.maximum(pc, 1e-12)
        + out_bits.reshape(-1, 1) / np.maximum(bw_bps, 1e-12)
    )
    return dur.mean(axis=1)


def _infer_task_count_from_dax_filename(dax_path: str) -> int:
    """
    支持你常见命名：CyberShake_30.xml / Montage_25.xml 等
    返回下划线后面的数字，否则返回 0
    """
    base = os.path.basename(str(dax_path))
    name = os.path.splitext(base)[0]
    if "_" not in name:
        return 0
    try:
        return int(name.split("_")[-1])
    except Exception:
        return 0


# ---------------- Environment ----------------

@dataclass
class TaskInfo:
    wf_id: int
    local_id: int


class IRWSEnvRouteA(gym.Env):
    """
    IRWS 环境：
      - 决策点：存在至少一个 Ready task 且至少一个 idle VM
      - 每一步：先在外部选 task（GNN），再在 env.step(vm_id) 执行分配
      - 事件驱动推进 current_time（到达/完成），推进段记录 idle energy（0-load）
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dax_paths: List[str],
        horizon: float = 1e9,
        arrival_lambda: float = 0.03,
        random_seed: int = 1,
        workflows_per_episode: int = 50,
        dax_probs: Optional[List[float]] = None,

        # resources
        num_hosts: int = 3,
        vms_per_host: Tuple[int, ...] = (9, 8, 8),
        pc_tiers: Tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 8.0),
        bw_tiers: Tuple[float, ...] = (1000.0, 2000.0, 4000.0, 6000.0, 8000.0),

        # fairness knobs
        max_ready_tasks: Any = "auto",
        normalize: bool = True,

        # deadline cache
        deadline_mode: str = "cache_fcfs",
        deadline_cache_path: Optional[str] = None,

        # fixed alpha fallback
        deadline_alpha: float = 1.5,

        # new: per-workflow alpha mixture
        deadline_alpha_choices: Optional[Tuple[float, ...]] = None,
        deadline_alpha_probs: Optional[Tuple[float, ...]] = None,

        deadline_cache_strict: bool = True,

        # new reward params (aligned to modified PD3QN)
        alpha_delay_vm: float = 0.75,
        task_baseline_norm: float = 300.0,
        energy_reward_scale: float = 1e-3,

        # compatibility with old param name (optional)
        alpha_delay: Optional[float] = None,
        energy_norm_per_mi_ref: float = 20.0,   # 兼容保留，本版 reward 中不再使用

        # bookkeeping
        energy_check_on_done: bool = True,
    ):
        super().__init__()

        self.dax_paths = list(dax_paths)
        self.dax_probs = None if dax_probs is None else np.array(dax_probs, dtype=np.float64)
        self.horizon = float(horizon)
        self.arrival_lambda = float(arrival_lambda)
        self.random_seed = int(random_seed)
        self.workflows_per_episode = int(workflows_per_episode)

        self.num_hosts = int(num_hosts)
        self.vms_per_host = tuple(int(x) for x in vms_per_host)
        self.pc_tiers = tuple(float(x) for x in pc_tiers)
        self.bw_tiers = tuple(float(x) for x in bw_tiers)

        self.normalize = bool(normalize)
        self.max_ready_tasks = max_ready_tasks

        # deadline cache
        self.deadline_mode = str(deadline_mode)
        self.deadline_cache_path = deadline_cache_path
        self.deadline_alpha = float(deadline_alpha)
        self.deadline_cache_strict = bool(deadline_cache_strict)

        # new: deadline alpha mixture
        self.deadline_alpha_choices, self.deadline_alpha_probs = self._resolve_deadline_alpha_mixture(
            deadline_alpha_choices, deadline_alpha_probs
        )

        self._deadline_cache = None
        if self.deadline_mode == "cache_fcfs":
            if not self.deadline_cache_path:
                raise ValueError("deadline_mode='cache_fcfs' 但未提供 deadline_cache_path")
            self._deadline_cache = self._load_deadline_cache(self.deadline_cache_path)

        # reward params
        self.alpha_delay_vm = float(alpha_delay_vm)
        if alpha_delay is not None:
            self.alpha_delay_vm = float(alpha_delay)

        self.task_baseline_norm = float(task_baseline_norm)
        self.energy_reward_scale = float(energy_reward_scale)

        # compatibility placeholders
        self.alpha_delay = float(self.alpha_delay_vm)
        self.energy_norm_per_mi_ref = float(energy_norm_per_mi_ref)

        self.energy_check_on_done = bool(energy_check_on_done)

        # cluster
        self.hosts, self.vms = create_cluster(
            num_hosts=self.num_hosts,
            vms_per_host=self.vms_per_host,
            pc_tiers=self.pc_tiers,
            bw_tiers=self.bw_tiers,
        )
        self.num_vms = len(self.vms)

        # for avg_dur calc
        self._vm_pc = np.array([float(self.vms[i].pc) for i in range(self.num_vms)], dtype=np.float64)   # MI/s
        self._vm_bw = np.array([float(self.vms[i].bw) for i in range(self.num_vms)], dtype=np.float64)   # Mb/s

        # infer auto max_ready_tasks
        self._max_ready_tasks_int = self._resolve_max_ready_tasks()

        # runtime state
        self.rng: Optional[np.random.RandomState] = None               # 只用于模板抽样
        self.deadline_rng: Optional[np.random.RandomState] = None      # 只用于ddlAlpha采样
        self.current_time: float = 0.0
        self.arrival_times: np.ndarray = np.zeros((0,), dtype=np.float64)
        self.next_arrival_idx: int = 0
        self.first_arrival_time: Optional[float] = None

        self.workflows: List[Workflow] = []
        self.completed_workflows: int = 0

        # flattened tasks
        self.task_meta: List[TaskInfo] = []
        self.task_state: List[str] = []             # "unReady" | "Ready" | "Running" | "Finished"
        self.task_ready_time: List[float] = []
        self.task_start_time: List[float] = []
        self.task_end_time: List[float] = []
        self.task_mi: List[float] = []
        self.task_in_bits: List[float] = []
        self.task_out_bits: List[float] = []
        self.task_up_rank: List[float] = []
        self.task_down_rank: List[float] = []

        # Workflow-local dependency IDs; runtime arrays and placements use global IDs.
        self.task_parents: List[List[int]] = []
        self.task_children: List[List[int]] = []
        self.ready_task_ids: List[int] = []

        # baseline deadlines per task (absolute)
        self.task_baseline_finish: List[float] = []

        # placements for parents_loc
        self.task_placement: Dict[int, Tuple[int, int]] = {}

        # VM availability times
        self.vm_available_at: np.ndarray = np.zeros((self.num_vms,), dtype=np.float64)

        # energy records
        self.records: List[LoadRecord] = []
        self.total_energy: float = 0.0

        # incremental host segments: host_id -> List[(start, end, load_pc_sum)]
        self._host_segments: Dict[int, List[Tuple[float, float, float]]] = {
            int(hid): [] for hid in range(self.num_hosts)
        }

        # current chosen task
        self.current_task_gid: Optional[int] = None

        # done flag
        self.done_flag: bool = False

        # Gym spaces
        self.action_space = gym.spaces.Discrete(self.num_vms)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    # ---------------- cache IO ----------------

    @staticmethod
    def _load_deadline_cache(path: str) -> Dict[int, dict]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"deadline cache not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        cache: Dict[int, dict] = {}
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            for rec in obj["data"]:
                cache[int(rec["seed"])] = rec
        elif isinstance(obj, dict) and "seed_to_record" in obj and isinstance(obj["seed_to_record"], dict):
            for k, rec in obj["seed_to_record"].items():
                cache[int(k)] = rec
        else:
            raise ValueError("deadline cache json 格式不正确：必须包含 data(list) 或 seed_to_record(dict)")
        return cache

    def _cache_lookup_makespan(self, episode_seed: int, wf_id: int, dax_name: Optional[str] = None) -> float:
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
            if isinstance(names, list) and wf_id < len(names) and str(names[wf_id]) != str(dax_name):
                print(
                    f"WARNING: [deadline-cache] seed={episode_seed} wf_id={wf_id} dax mismatch: "
                    f"runtime={dax_name} cache={names[wf_id]}"
                )
        return float(ms_list[wf_id])

    # ---------------- new: deadline alpha helpers ----------------

    @staticmethod
    def _resolve_deadline_alpha_mixture(
        choices: Optional[Tuple[float, ...]],
        probs: Optional[Tuple[float, ...]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        若 choices/probs 为空，则表示使用固定 deadline_alpha。
        若提供，则使用每个 workflow 独立采样。
        """
        if choices is None and probs is None:
            return None, None

        if choices is None or probs is None:
            raise ValueError("deadline_alpha_choices 和 deadline_alpha_probs 必须同时提供，或同时不提供")

        choices_arr = np.array(choices, dtype=np.float64).reshape(-1)
        probs_arr = np.array(probs, dtype=np.float64).reshape(-1)

        if choices_arr.ndim != 1 or probs_arr.ndim != 1:
            raise ValueError("deadline_alpha_choices / deadline_alpha_probs 必须是一维序列")
        if len(choices_arr) == 0:
            raise ValueError("deadline_alpha_choices 不能为空")
        if len(choices_arr) != len(probs_arr):
            raise ValueError("deadline_alpha_choices 与 deadline_alpha_probs 长度必须一致")
        if np.any(probs_arr < 0):
            raise ValueError("deadline_alpha_probs 不能含负数")

        s = float(probs_arr.sum())
        if s <= 0:
            raise ValueError("deadline_alpha_probs 之和必须 > 0")

        probs_arr = probs_arr / s
        return choices_arr, probs_arr

    def _sample_deadline_alpha(self) -> float:
        """
        每个 workflow 独立采样 ddlAlpha。
        注意：必须用独立的 deadline_rng，不能占用 self.rng，
        否则会破坏 DAX 模板抽样序列，影响 cache 对齐。
        """
        if self.deadline_alpha_choices is None or self.deadline_alpha_probs is None:
            return float(self.deadline_alpha)

        if self.deadline_rng is None:
            raise RuntimeError("deadline_rng 尚未初始化，请先 reset()")

        idx = int(self.deadline_rng.choice(len(self.deadline_alpha_choices), p=self.deadline_alpha_probs))
        return float(self.deadline_alpha_choices[idx])

    # ---------------- config helpers ----------------

    def _resolve_max_ready_tasks(self) -> int:
        if isinstance(self.max_ready_tasks, (int, np.integer)) and int(self.max_ready_tasks) > 0:
            return int(self.max_ready_tasks)
        if isinstance(self.max_ready_tasks, str) and self.max_ready_tasks.lower() == "auto":
            max_n = 0
            for p in self.dax_paths:
                max_n = max(max_n, _infer_task_count_from_dax_filename(str(p)))
            if max_n <= 0:
                max_n = 2000
            return int(self.workflows_per_episode * max_n)
        return int(self.workflows_per_episode * 2000)

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
        """
        与修改后的 PD3QN 一致：
        在 host 的 [start,end] 区间新增 delta_pc 负载，返回该次新增的真实增量能耗 ΔE_host。
        """
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

    # ---------------- core API ----------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # self.rng 只用于模板抽样
        self.rng = np.random.RandomState(int(self.random_seed))

        # 独立 deadline RNG，避免打乱模板序列
        self.deadline_rng = np.random.RandomState(int(self.random_seed) + 100003)

        self.current_time = 0.0
        self.next_arrival_idx = 0
        self.first_arrival_time = None

        self.workflows = []
        self.completed_workflows = 0

        self.task_meta = []
        self.task_state = []
        self.task_ready_time = []
        self.task_start_time = []
        self.task_end_time = []
        self.task_mi = []
        self.task_in_bits = []
        self.task_out_bits = []
        self.task_up_rank = []
        self.task_down_rank = []
        self.task_parents = []
        self.task_children = []
        self.ready_task_ids = []
        self.task_baseline_finish = []
        self.task_placement = {}
        self.vm_available_at = np.zeros((self.num_vms,), dtype=np.float64)

        self.records = []
        self.total_energy = 0.0
        self._host_segments = {int(hid): [] for hid in range(self.num_hosts)}

        self.current_task_gid = None
        self.done_flag = False

        self.arrival_times = np.array(
            poisson_arrival_times(
                self.arrival_lambda,
                self.horizon,
                seed=int(self.random_seed),
                n_max=int(self.workflows_per_episode),
            ),
            dtype=np.float64,
        )
        if len(self.arrival_times) == 0:
            self.done_flag = True
            return np.zeros((1,), dtype=np.float32), {}

        self.current_time = float(self.arrival_times[0])
        self.first_arrival_time = float(self.arrival_times[0])

        self._add_workflows_up_to_current_time()
        self._advance_until_decision()

        return np.zeros((1,), dtype=np.float32), {}

    # ---------------- decision helpers ----------------

    def get_ready_task_candidates(self) -> Dict[str, Any]:
        gids = list(self.ready_task_ids)
        gids.sort(key=lambda gid: (self.task_ready_time[gid], gid))
        if len(gids) > self._max_ready_tasks_int:
            gids = gids[: self._max_ready_tasks_int]

        feats = self._build_task_features(gids)
        mask = np.ones((len(gids),), dtype=np.float32)
        return {"gids": gids, "feats": feats, "mask": mask}

    def select_task(self, task_gid: int):
        if self.done_flag:
            raise RuntimeError("env already done")
        if task_gid not in self.ready_task_ids:
            raise ValueError(f"task_gid={task_gid} not in ready_task_ids")
        self.current_task_gid = int(task_gid)

    def get_vm_obs_and_mask(self) -> Dict[str, Any]:
        if self.current_task_gid is None:
            raise RuntimeError("call select_task() before get_vm_obs_and_mask()")

        gid = int(self.current_task_gid)
        mask = (self.vm_available_at <= self.current_time + 1e-12).astype(np.float32)
        if float(mask.sum()) <= 0.0:
            mask[-1] = 1.0

        obs = self._build_vm_obs_for_task(gid)
        return {"obs": obs, "mask": mask}

    def step(self, action: int):
        if self.done_flag:
            return np.zeros((1,), dtype=np.float32), 0.0, True, False, {}

        if self.current_task_gid is None:
            raise RuntimeError("select_task() must be called before step()")

        vm_id = int(action)
        gid = int(self.current_task_gid)
        self.current_task_gid = None

        if self.vm_available_at[vm_id] > self.current_time + 1e-12:
            reward = -10.0
            info = {"illegal": True}
            return np.zeros((1,), dtype=np.float32), float(reward), False, False, info

        self.ready_task_ids.remove(gid)
        reward, info = self._assign_task_to_vm(gid, vm_id)

        self._advance_until_decision()

        terminated = bool(self.done_flag)
        truncated = False
        return np.zeros((1,), dtype=np.float32), float(reward), terminated, truncated, info

    # ---------------- internals: workflow arrival ----------------

    def _add_workflows_up_to_current_time(self):
        while self.next_arrival_idx < len(self.arrival_times) and self.arrival_times[self.next_arrival_idx] <= self.current_time + 1e-12:
            arr_t = float(self.arrival_times[self.next_arrival_idx])

            # DAX template choice: strictly use self.rng only
            if self.dax_probs is None:
                idx = int(self.rng.randint(len(self.dax_paths)))
            else:
                idx = int(self.rng.choice(len(self.dax_paths), p=self.dax_probs))
            path = str(self.dax_paths[idx])
            dax_name = os.path.basename(path)

            # payload seed aligned to HRL
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
            wf.dax_path = path

            # ranks
            pc_mean = float(np.mean(self._vm_pc))
            bw_mean = float(np.mean(self._vm_bw))
            wf.compute_upward_ranks(pc_mean, bw_mean)
            wf.compute_downward_ranks(pc_mean, bw_mean)

            # deadline setting
            if self.deadline_mode == "cache_fcfs":
                makespan_ref = self._cache_lookup_makespan(
                    int(self.random_seed),
                    int(wf.workflow_id),
                    dax_name=dax_name,
                )
                alpha_used = self._sample_deadline_alpha()

                wf.ref_makespan = float(makespan_ref)
                wf.deadline_alpha_used = float(alpha_used)
                wf.deadline = float(arr_t + alpha_used * float(makespan_ref))

            elif self.deadline_mode == "none":
                wf.ref_makespan = None
                wf.deadline_alpha_used = float(self._sample_deadline_alpha())
                wf.deadline = float(arr_t + max(self.horizon, 1.0))
            else:
                raise ValueError(f"Unsupported deadline_mode: {self.deadline_mode}")

            # Latest-Finish reverse deadline
            avg_dur = _avg_task_dur_over_all_vms(tasks, self._vm_pc, self._vm_bw)
            D_wf_rel = float(wf.deadline - arr_t)

            n_local = len(tasks)
            indeg = np.zeros((n_local,), dtype=np.int32)
            succ: List[List[int]] = [[] for _ in range(n_local)]
            for i, t in enumerate(tasks):
                for c in list(getattr(t, "children", [])):
                    succ[i].append(int(c))
                indeg[i] = len(list(getattr(t, "parents", [])))

            topo = []
            q = [i for i in range(n_local) if indeg[i] == 0]
            q.sort()
            while q:
                u = q.pop(0)
                topo.append(u)
                for v in succ[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q.append(v)
                        q.sort()

            is_sink = np.array([len(succ[i]) == 0 for i in range(n_local)], dtype=bool)
            LF = np.zeros((n_local,), dtype=np.float64)
            LS = np.zeros((n_local,), dtype=np.float64)

            for i in range(n_local):
                if is_sink[i]:
                    LF[i] = D_wf_rel
                    LS[i] = LF[i] - avg_dur[i]

            for u in reversed(topo):
                if is_sink[u]:
                    continue
                if len(succ[u]) > 0:
                    min_ls_child = float(np.min(LS[np.array(succ[u], dtype=np.int32)]))
                    LF[u] = min_ls_child
                else:
                    LF[u] = D_wf_rel
                LS[u] = LF[u] - avg_dur[u]

            # flatten tasks
            base = len(self.task_meta)
            for local_id, t in enumerate(tasks):
                self.task_meta.append(TaskInfo(wf_id=wf.workflow_id, local_id=int(local_id)))
                self.task_state.append("unReady")
                self.task_ready_time.append(0.0)
                self.task_start_time.append(0.0)
                self.task_end_time.append(0.0)

                self.task_mi.append(float(t.workload_mi))
                in_bits = float((getattr(t, "ext_in_bits", 0.0) or 0.0) + (getattr(t, "from_parents_bits", 0.0) or 0.0))
                self.task_in_bits.append(in_bits)
                self.task_out_bits.append(float(getattr(t, "out_file_size_sum_bits", 0.0) or 0.0))
                self.task_up_rank.append(float(getattr(t, "upward_rank", 0.0) or 0.0))
                self.task_down_rank.append(float(getattr(t, "downward_rank", 0.0) or 0.0))

                self.task_parents.append([int(p) for p in getattr(t, "parents", [])])
                self.task_children.append([int(c) for c in getattr(t, "children", [])])

                abs_deadline = float(arr_t + LF[int(local_id)])
                self.task_baseline_finish.append(abs_deadline)

            for local_id, t in enumerate(tasks):
                if len(list(getattr(t, "parents", []))) == 0:
                    gid = base + int(local_id)
                    self.task_state[gid] = "Ready"
                    self.task_ready_time[gid] = float(self.current_time)
                    self.ready_task_ids.append(gid)

            self.workflows.append(wf)
            self.next_arrival_idx += 1

    # ---------------- internals: scheduling & time advance ----------------

    def _assign_task_to_vm(self, gid: int, vm_id: int) -> Tuple[float, dict]:
        vm = self.vms[vm_id]
        host_id = int(vm.host_id)

        t0 = float(self.current_time)
        t0 = max(t0, float(self.vm_available_at[vm_id]))
        t0 = max(t0, float(self.task_ready_time[gid]))

        base = gid - int(self.task_meta[gid].local_id)
        parents = self.task_parents[gid]
        if parents:
            pe = [float(self.task_end_time[base + p]) for p in parents]
            t0 = max(t0, float(max(pe)))

        parents_loc = self._build_parents_loc(gid)

        tau_in, tau_comp, tau_out, tau_exec = exec_time_components(
            self._task_obj_from_gid(gid),
            vm_pc_mi_s=float(vm.pc),
            vm_bw_Mb_s=float(vm.bw),
            parents_loc=parents_loc,
            child_vm_id=int(vm_id),
            child_host_id=int(host_id),
            zero_if_same_vm=True,
            zero_if_same_host=True,
        )
        dur = float(tau_exec)
        t1 = float(t0 + dur)

        self.task_state[gid] = "Running"
        self.task_start_time[gid] = float(t0)
        self.task_end_time[gid] = float(t1)
        self.task_placement[gid] = (int(vm_id), int(host_id))
        self.vm_available_at[vm_id] = float(t1)

        self.records.append(LoadRecord(start_time=t0, end_time=t1, server_id=int(host_id), vm_pc=float(vm.pc)))

        # 与修改后的 PD3QN 一致：使用真实 ΔE_host
        dE_host = self._add_interval_load(host_id, t0, t1, float(vm.pc))

        ddl = float(self.task_baseline_finish[gid])
        lateness = float(t1 - ddl)
        tardiness = float(max(0.0, lateness))

        delay_norm = _clip01(float(tardiness / max(self.task_baseline_norm, 1e-9)))
        r_delay = -float(delay_norm)

        r_energy_host = -float(dE_host) * float(self.energy_reward_scale)

        alpha = float(self.alpha_delay_vm)
        reward = (1.0 - alpha) * float(r_energy_host) + alpha * float(r_delay)

        info = {
            "gid": int(gid),
            "vm_id": int(vm_id),
            "host_id": int(host_id),
            "start": float(t0),
            "end": float(t1),
            "dur": float(dur),
            "ddl_task": float(ddl),
            "lateness": float(lateness),
            "tardiness": float(tardiness),
            "delta_energy": float(dE_host),
            "delay_norm": float(delay_norm),
            "r_delay": float(r_delay),
            "r_energy_host": float(r_energy_host),
            "alpha_delay_vm": float(alpha),
            "reward_mode": "align_modified_pd3qn",
        }
        return float(reward), info

    def _build_parents_loc(self, gid: int) -> Dict[int, Tuple[int, int]]:
        base = gid - int(self.task_meta[gid].local_id)
        return {
            p: self.task_placement[base + p]
            for p in self.task_parents[gid] if base + p in self.task_placement
        }

    def _task_obj_from_gid(self, gid: int):
        ti = self.task_meta[gid]
        wf = self.workflows[int(ti.wf_id)]
        return wf.tasks[int(ti.local_id)]

    def _advance_until_decision(self):
        self._process_completions_at_time(self.current_time)

        while True:
            if self._check_done():
                self._finalize_energy_if_done()
                self.done_flag = True
                return

            has_ready = (len(self.ready_task_ids) > 0)
            has_idle_vm = bool(np.any(self.vm_available_at <= self.current_time + 1e-12))
            if has_ready and has_idle_vm:
                return

            t_next = float("inf")
            if self.next_arrival_idx < len(self.arrival_times):
                t_next = min(t_next, float(self.arrival_times[self.next_arrival_idx]))

            t_comp = self._next_completion_time()
            if t_comp is not None:
                t_next = min(t_next, float(t_comp))

            if not math.isfinite(t_next):
                self._finalize_energy_if_done()
                self.done_flag = True
                return

            t0 = float(self.current_time)
            t1 = float(t_next)
            if t1 < t0 + 1e-12:
                t1 = t0

            if t1 > t0 + 1e-12:
                for hid in range(self.num_hosts):
                    self.records.append(LoadRecord(start_time=t0, end_time=t1, server_id=int(hid), vm_pc=0.0))
                    self._add_interval_load(int(hid), t0, t1, 0.0)

            self.current_time = float(t1)

            self._add_workflows_up_to_current_time()
            self._process_completions_at_time(self.current_time)

    def _next_completion_time(self) -> Optional[float]:
        tmin = None
        for gid, st in enumerate(self.task_state):
            if st != "Running":
                continue
            te = float(self.task_end_time[gid])
            if te <= self.current_time + 1e-12:
                continue
            if tmin is None or te < tmin:
                tmin = te
        return tmin

    def _process_completions_at_time(self, t: float):
        t = float(t)
        finished_now = []
        for gid, st in enumerate(self.task_state):
            if st != "Running":
                continue
            if float(self.task_end_time[gid]) <= t + 1e-12:
                finished_now.append(gid)

        for gid in finished_now:
            self.task_state[gid] = "Finished"
            base = gid - int(self.task_meta[gid].local_id)
            for c_local in self.task_children[gid]:
                c = base + c_local
                if self.task_state[c] != "unReady":
                    continue
                ok = True
                for p in self.task_parents[c]:
                    if self.task_state[base + p] != "Finished":
                        ok = False
                        break
                if ok:
                    self.task_state[c] = "Ready"
                    self.task_ready_time[c] = float(t)
                    self.ready_task_ids.append(c)

        self._update_completed_workflows()

    def _update_completed_workflows(self):
        completed = 0
        for wf_id in range(len(self.workflows)):
            ok = True
            for gid, ti in enumerate(self.task_meta):
                if int(ti.wf_id) != wf_id:
                    continue
                if self.task_state[gid] != "Finished":
                    ok = False
                    break
            if ok:
                completed += 1
        self.completed_workflows = int(completed)

    def _check_done(self) -> bool:
        if len(self.workflows) < self.workflows_per_episode:
            return False
        if self.completed_workflows < self.workflows_per_episode:
            return False
        for st in self.task_state:
            if st == "Running":
                return False
        return True

    def _finalize_energy_if_done(self):
        e_prev = float(self.total_energy)
        e_true = float(energy_from_records(self.records, self.hosts))
        diff = abs(e_prev - e_true)

        if self.energy_check_on_done and diff > 1e-6:
            print(f"[EnergyCheck] mismatch: E_inc={e_prev:.6f} E_true={e_true:.6f} diff={diff:.6f}")

        self.total_energy = float(e_true)

    # ---------------- metrics ----------------

    def get_episode_metrics(self) -> Dict[str, float]:
        if not self.done_flag:
            E = float(energy_from_records(self.records, self.hosts))
        else:
            E = float(self.total_energy)

        if self.first_arrival_time is None:
            makespan = 0.0
        else:
            last_finish = 0.0
            for gid, st in enumerate(self.task_state):
                if st == "Finished":
                    last_finish = max(last_finish, float(self.task_end_time[gid]))
            makespan = float(last_finish - float(self.first_arrival_time))

        task_lateness_list = []
        task_tardiness_list = []
        task_tardy_count = 0
        wf_task_tardy_sum: Dict[int, float] = {}

        for gid, st in enumerate(self.task_state):
            if st != "Finished":
                continue

            ddl = float(self.task_baseline_finish[gid])
            ft = float(self.task_end_time[gid])

            lateness = float(ft - ddl)
            tardiness = float(max(0.0, lateness))

            task_lateness_list.append(lateness)
            task_tardiness_list.append(tardiness)

            if tardiness > 1e-12:
                task_tardy_count += 1

            wf_id = int(self.task_meta[gid].wf_id)
            wf_task_tardy_sum[wf_id] = wf_task_tardy_sum.get(wf_id, 0.0) + tardiness

        task_count = int(len(task_lateness_list))
        task_total_lateness = float(np.sum(task_lateness_list)) if task_count > 0 else 0.0
        task_total_tardiness = float(np.sum(task_tardiness_list)) if task_count > 0 else 0.0
        task_avg_lateness = float(np.mean(task_lateness_list)) if task_count > 0 else 0.0
        task_avg_tardiness = float(np.mean(task_tardiness_list)) if task_count > 0 else 0.0
        pct_tardy = float(task_tardy_count / task_count) if task_count > 0 else 0.0
        task_success_rate = float(1.0 - pct_tardy) if task_count > 0 else 0.0

        wf_finish_time: Dict[int, float] = {}
        wf_total_tasks: Dict[int, int] = {}
        wf_finished_tasks: Dict[int, int] = {}

        for gid, ti in enumerate(self.task_meta):
            wf_id = int(ti.wf_id)
            wf_total_tasks[wf_id] = wf_total_tasks.get(wf_id, 0) + 1

            if self.task_state[gid] == "Finished":
                wf_finished_tasks[wf_id] = wf_finished_tasks.get(wf_id, 0) + 1
                ft = float(self.task_end_time[gid])
                if wf_id not in wf_finish_time or ft > wf_finish_time[wf_id]:
                    wf_finish_time[wf_id] = ft

        wf_lateness_list = []
        wf_tardiness_list = []
        wf_tardy_count = 0

        for wf_id, wf in enumerate(self.workflows):
            total_n = wf_total_tasks.get(wf_id, 0)
            finished_n = wf_finished_tasks.get(wf_id, 0)

            if total_n <= 0 or finished_n < total_n:
                continue

            finish_t = float(wf_finish_time[wf_id])
            wf_deadline = float(wf.deadline)

            wf_lateness = float(finish_t - wf_deadline)
            wf_tardiness = float(max(0.0, wf_lateness))

            wf_lateness_list.append(wf_lateness)
            wf_tardiness_list.append(wf_tardiness)

            if wf_tardiness > 1e-12:
                wf_tardy_count += 1

        wf_count = int(len(wf_lateness_list))
        wf_total_lateness = float(np.sum(wf_lateness_list)) if wf_count > 0 else 0.0
        wf_total_tardiness = float(np.sum(wf_tardiness_list)) if wf_count > 0 else 0.0
        wf_avg_lateness = float(np.mean(wf_lateness_list)) if wf_count > 0 else 0.0
        wf_avg_tardiness = float(np.mean(wf_tardiness_list)) if wf_count > 0 else 0.0
        wf_pct_tardy = float(wf_tardy_count / wf_count) if wf_count > 0 else 0.0
        wf_success_rate = float(1.0 - wf_pct_tardy) if wf_count > 0 else 0.0

        if wf_count > 0:
            wf_avg_tardy = float(
                sum(wf_task_tardy_sum.get(wf_id, 0.0) for wf_id in range(len(self.workflows))) / wf_count
            )
        else:
            wf_avg_tardy = 0.0

        return {
            "total_energy": float(E),
            "makespan": float(makespan),

            "task_count": float(task_count),
            "task_tardy_count": float(task_tardy_count),
            "task_total_lateness": float(task_total_lateness),
            "task_total_tardiness": float(task_total_tardiness),
            "task_avg_lateness": float(task_avg_lateness),
            "task_avg_tardiness": float(task_avg_tardiness),
            "pct_tardy": float(pct_tardy),
            "task_success_rate": float(task_success_rate),

            "wf_count": float(wf_count),
            "wf_tardy_count": float(wf_tardy_count),
            "wf_total_lateness": float(wf_total_lateness),
            "wf_total_tardiness": float(wf_total_tardiness),
            "wf_avg_lateness": float(wf_avg_lateness),
            "wf_avg_tardiness": float(wf_avg_tardiness),
            "wf_pct_tardy": float(wf_pct_tardy),
            "wf_success_rate": float(wf_success_rate),

            "wf_avg_tardy": float(wf_avg_tardy),
        }

    # ---------------- feature builders ----------------

    def _build_task_features(self, gids: List[int]) -> np.ndarray:
        if len(gids) == 0:
            return np.zeros((0, 7), dtype=np.float32)

        t = float(self.current_time)
        feats = []
        for gid in gids:
            mi = float(self.task_mi[gid])
            ib = float(self.task_in_bits[gid])
            ob = float(self.task_out_bits[gid])
            ur = float(self.task_up_rank[gid])
            dr = float(self.task_down_rank[gid])
            ddl = float(self.task_baseline_finish[gid])
            slack = float(ddl - t)
            wait = float(max(0.0, t - float(self.task_ready_time[gid])))

            if self.normalize:
                mi = mi / 3000.0
                ib = ib / (1024 * 1024 * 8 * 1024)
                ob = ob / (1024 * 1024 * 8 * 1024)
                ur = ur / 1000.0
                dr = dr / 1000.0
                slack = slack / 1000.0
                wait = wait / 1000.0

            feats.append([mi, ib, ob, ur, dr, slack, wait])

        return np.array(feats, dtype=np.float32)

    def _build_vm_obs_for_task(self, gid: int) -> np.ndarray:
        t = float(self.current_time)
        task_feats = self._build_task_features([gid])[0]
        ddl = float(self.task_baseline_finish[gid])

        parents_loc = self._build_parents_loc(gid)
        vm_feats = []
        for vm_id in range(self.num_vms):
            vm = self.vms[vm_id]
            idle = 1.0 if self.vm_available_at[vm_id] <= t + 1e-12 else 0.0
            avail_dt = float(max(0.0, self.vm_available_at[vm_id] - t))

            _ti, _tc, _to, tau_exec = exec_time_components(
                self._task_obj_from_gid(gid),
                vm_pc_mi_s=float(vm.pc),
                vm_bw_Mb_s=float(vm.bw),
                parents_loc=parents_loc,
                child_vm_id=int(vm_id),
                child_host_id=int(vm.host_id),
                zero_if_same_vm=True,
                zero_if_same_host=True,
            )
            est_exec = float(tau_exec)
            est_finish = float(max(t, float(self.vm_available_at[vm_id])) + est_exec)
            est_slack = float(ddl - est_finish)

            pc = float(vm.pc)
            bw = float(vm.bw)

            if self.normalize:
                avail_dt = avail_dt / 1000.0
                pc = pc / 10.0
                bw = bw / 10000.0
                est_exec = est_exec / 1000.0
                est_slack = est_slack / 1000.0

            vm_feats.append([idle, avail_dt, pc, bw, est_exec, est_slack])

        vm_feats = np.array(vm_feats, dtype=np.float32).reshape(-1)
        obs = np.concatenate([task_feats.astype(np.float32), vm_feats], axis=0).astype(np.float32)
        return obs