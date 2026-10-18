# -*- coding: utf-8 -*-
"""
env_heft.py

三层智能体环境（阶段式 HRL + HostAgent + VMAgent）：

L1 Manager（阶段级）【动态权重增量控制】：
- 维护 5 维 raw 权重 w_raw（FCFS/SJF/MCF/HUR/EDF）
- 每个 phase 开始时，Manager 选择一个离散动作 a ∈ {0..242}
  对应 Δw ∈ {-0.1, 0, +0.1}^5（共 3^5=243 种组合）
- 更新：w_raw ← clip(w_raw + Δw, min=0)；若全 0 回退为全 1
- combo_weights = normalize(w_raw) 用于 ready tasks 打分排序
- 提供 get_manager_action_mask()：若某维 w_raw==0，则禁止该维再减 0.1 的动作

Deadline 逻辑（✅按你的新要求修改）：
1) 工作流级 deadline:
   wf.deadline = wf.arrival_time + 1.5 * HEFT_makespan
   - HEFT_makespan：在“当前集群的全部 VM 上”跑一遍简化 HEFT（含任务上传/计算/下载时长，任务依赖约束）
2) 任务级 deadline（task_baseline_finish）：
   根据 wf_deadline 做倒推（Latest-Finish 反推）得到每个任务的子 deadline（绝对时间）
   - 仍用于 Host/VM reward 的 lateness 计算（finish_time - task_deadline）
3) 工作流级 deadline 用途：
   - EDF 特征（在启发式特征中）
   - Manager state 的 slack 统计（min/avg slack）
   - 判定工作流是否超时（wf_finish_time > wf_deadline）
   - 统计每个 episode 的 workflow lateness 之和：sum(max(0, wf_finish_time - wf_deadline))

中层/下层 reward：路线A（可归因能耗代理 + 延迟归一化加和）负 cost
- delay_norm = clip( max(0, finish_time - task_deadline) / task_baseline_norm, 0, 1 )
- energy_norm = clip( (E_hat/MI) / energy_norm_per_mi_ref, 0, 1 )
- r = -[(1-alpha)*energy_norm + alpha*delay_norm]
其中 E_hat 为“边际功率差 × 任务持续时间”的代理：
- P_before = host.power(load_before)
- P_after  = host.power(load_after)
- E_hat = max(0, P_after - P_before) * duration

上层能耗仍按真实积分在时间推进时结算，保证与原HRL口径一致。
"""

import heapq
import numpy as np

import os
import re
from functools import lru_cache

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym
    from gym import spaces

# === 统一能耗与工作流 ===
from common.workflow_opt import LoadRecord, energy_from_records, Workflow, exec_time_components
from common.read_xml_opt_Tsize import load_workflow_from_dax, poisson_arrival_times
from common.resource_opt import create_cluster


def _safe_div(a, b, eps=1e-9):
    return float(a) / float(b if abs(b) > eps else (eps if b >= 0 else -eps))


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


# ===================== Manager 动作表（3^5=243） =====================
MANAGER_STEP = 0.1


def _build_manager_action_table(step: float = 0.1) -> np.ndarray:
    """
    固定顺序生成 3^5 个 Δw 组合，保证可复现：
    每维取值 [-step, 0, +step]，按 lexicographic 顺序展开 (d0,d1,d2,d3,d4)
    """
    vals = (-step, 0.0, step)
    table = []
    for d0 in vals:
        for d1 in vals:
            for d2 in vals:
                for d3 in vals:
                    for d4 in vals:
                        table.append([d0, d1, d2, d3, d4])
    return np.asarray(table, dtype=np.float32)


MANAGER_ACTION_TABLE = _build_manager_action_table(MANAGER_STEP)  # shape=(243,5)

# ===================== max_ready_tasks 自动推导（方案1变量化） =====================
@lru_cache(maxsize=None)
def _infer_task_count_from_dax_path(dax_path: str) -> int:
    """
    优先从文件名推断任务数：如 CyberShake_30.xml -> 30
    若推断失败，则读取DAX解析一次，返回 len(tasks)
    """
    base = os.path.basename(dax_path)
    m = re.search(r"_(\d+)\.xml$", base)
    if m:
        return int(m.group(1))

    # fallback：读一次 DAX（只用于少量模板，且有 lru_cache）
    try:
        G, tasks = load_workflow_from_dax(dax_path, seed=0, randomize_payloads=False)
        return int(len(tasks))
    except Exception:
        return 0


def _auto_max_ready_tasks(dax_paths, workflows_per_episode: int) -> int:
    """
    max_ready_tasks = W * Nmax
    - W: workflows_per_episode
    - Nmax: dax_paths 中最大任务规模
    """
    nmax = 0
    for p in dax_paths:
        nmax = max(nmax, _infer_task_count_from_dax_path(str(p)))
    if nmax <= 0:
        nmax = 100  # 兜底：仅影响归一化尺度，不影响调度正确性

    if workflows_per_episode is None:
        return int(nmax)

    return int(workflows_per_episode * nmax)


# ===================== HEFT（简化版）用于工作流 makespan 估计 =====================
def _topo_sort_local(tasks):
    """Kahn topo sort on local task graph (task_id assumed 0..n-1)."""
    n = len(tasks)
    indeg = [0] * n
    succ = [[] for _ in range(n)]
    for t in tasks:
        i = int(t.task_id)
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
        # fallback（理论上不应发生）
        topo = list(range(n))
    return topo, succ


def _heft_makespan_and_finish_times(tasks, vm_pc, vm_bw_bps):
    """
    在给定 VM 集合上执行简化 HEFT（无显式父子通信项；上传/下载/计算都计入任务自身 duration）。
    返回：
      - makespan (float, relative)
      - finish_times (np.ndarray shape=(n,))
      - avg_dur (np.ndarray shape=(n,))  # 平均 task duration，用于倒推 deadline
    """
    n = len(tasks)
    if n == 0:
        return 0.0, np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    topo, succ = _topo_sort_local(tasks)

    mi = np.array([float(t.workload_mi) for t in tasks], dtype=np.float64)
    in_bits = np.array([float((getattr(t, "ext_in_bits", 0.0) or 0.0) + (getattr(t, "from_parents_bits", 0.0) or 0.0)) for t in tasks], dtype=np.float64)
    out_bits = np.array([float(getattr(t, "out_file_size_sum_bits", 0.0) or 0.0) for t in tasks], dtype=np.float64)

    # duration matrix: [n, P]
    # dur(i, p) = in/bw + mi/pc + out/bw
    bw = vm_bw_bps.reshape(1, -1)  # [1,P]
    pc = vm_pc.reshape(1, -1)      # [1,P]
    dur = (in_bits.reshape(-1, 1) / np.maximum(bw, 1e-12)
           + mi.reshape(-1, 1) / np.maximum(pc, 1e-12)
           + out_bits.reshape(-1, 1) / np.maximum(bw, 1e-12))

    avg_dur = dur.mean(axis=1)  # [n]

    # upward rank (comm=0)
    rank_u = np.zeros((n,), dtype=np.float64)
    for u in reversed(topo):
        if len(succ[u]) == 0:
            rank_u[u] = avg_dur[u]
        else:
            rank_u[u] = avg_dur[u] + float(np.max(rank_u[succ[u]]))

    # HEFT list scheduling:
    indeg = np.array([len(getattr(tasks[i], "parents", [])) for i in range(n)], dtype=np.int32)
    ready = []
    for i in range(n):
        if indeg[i] == 0:
            # max-heap by rank => min-heap on (-rank, id)
            heapq.heappush(ready, (-rank_u[i], i))

    Pn = vm_pc.size
    vm_ready = np.zeros((Pn,), dtype=np.float64)
    finish = np.zeros((n,), dtype=np.float64)
    start = np.zeros((n,), dtype=np.float64)

    # parents list for quick access
    parents = [list(getattr(tasks[i], "parents", [])) for i in range(n)]

    placement = {}
    scheduled = 0
    while ready:
        _neg_r, u = heapq.heappop(ready)

        if len(parents[u]) == 0:
            ready_t = 0.0
        else:
            ready_t = float(np.max(finish[np.array(parents[u], dtype=np.int32)]))

        # compute EFT on all VMs
        st = np.maximum(vm_ready, ready_t)
        candidate_durations = np.array([
            exec_time_components(
                tasks[u], float(vm_pc[j]), float(vm_bw_bps[j]) / 1e6,
                parents_loc=placement, child_vm_id=j, zero_if_same_host=False,
            )[-1]
            for j in range(Pn)
        ])
        ft = st + candidate_durations
        p_star = int(np.argmin(ft))
        start[u] = float(st[p_star])
        finish[u] = float(ft[p_star])
        vm_ready[p_star] = finish[u]
        placement[u] = (p_star, -1)

        scheduled += 1
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(ready, (-rank_u[v], v))

    if scheduled != n:
        # fallback：如果出现异常，使用关键路径近似
        ms = float(np.max(avg_dur)) if n > 0 else 0.0
        return ms, finish, avg_dur

    makespan = float(np.max(finish))
    return makespan, finish, avg_dur


class CloudWorkflowEnv_VMAgents(gym.Env):
    metadata = {"render.modes": ["human"]}

    # ---------------- 初始化 ----------------
    def __init__(
        self,
        dax_paths,
        horizon=36000.0,
        arrival_lambda=0.1,
        random_seed=0,
        # 观测/任务槽
        max_ready_tasks=32,               # 仅用于比例归一的上限；本环境不Top-K截断
        normalize=True,
        # 资源
        num_hosts=3,
        vms_per_host=(9, 8, 8),
        pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),                # MI/s
        bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0), # Mb/s
        # DAX 模板选择
        dax_probs=None,
        # 状态包含角度（保留参数，实际三层将分别构造obs）
        state_include_vm=True,
        state_include_host=True,
        state_include_queue=True,
        # 模板组合权重（5 维：FCFS/SJF/MCF/HUR/EDF）【作为初值保留，但 reset 时会覆盖】
        combo_weights=None,
        # 每个 episode 完成的工作流个数
        workflows_per_episode=8,
    ):
        super().__init__()

        # 基础参数
        self.rng = np.random.RandomState(random_seed)
        self.horizon = float(horizon)
        self.arrival_lambda = float(arrival_lambda)
        self.dax_paths = list(dax_paths)
        self.random_seed = int(random_seed)
        self.normalize = bool(normalize)

        # === reward尺度与三层奖励参数 ===
        # 上层：真实能耗差分
        self.energy_reward_scale = 1e-3

        # 路线A：延迟归一化标尺（秒）
        self.task_baseline_norm = 300.0

        # 路线A：能耗强度归一化标尺（J/MI）
        self.energy_norm_per_mi_ref = 20.0

        # 路线A：delay权重（不同层可不同）
        self.alpha_delay_host = 0.35
        self.alpha_delay_vm = 0.60

        # 结束条件
        self.workflows_per_episode = int(workflows_per_episode) if workflows_per_episode is not None else None

        # DAX 模板概率
        self.dax_probs = None
        if dax_probs is not None:
            probs = np.asarray(dax_probs, dtype=float).copy()
            assert len(probs) == len(self.dax_paths), "dax_probs 长度需与 dax_paths 一致"
            probs = np.clip(probs, 0.0, None)
            if probs.sum() <= 0:
                probs[:] = 1.0
            self.dax_probs = probs / probs.sum()

        # 集群
        self.hosts, self.vms = create_cluster(
            num_hosts=num_hosts,
            vms_per_host=vms_per_host,
            pc_tiers=pc_tiers,
            bw_tiers=bw_tiers,
        )
        self.host_ids = sorted(self.hosts.keys())
        self.vm_ids = sorted(self.vms.keys())
        self.num_hosts = len(self.hosts)
        self.num_vms = len(self.vm_ids)
        self.vm_host = np.array([self.vms[vid].host_id for vid in self.vm_ids], dtype=np.int32)

        # ✅ HEFT 使用的 VM 参数（预先缓存）
        self._heft_vm_pc = np.array([float(self.vms[vid].pc) for vid in self.vm_ids], dtype=np.float64)
        self._heft_vm_bw_bps = np.array([float(self.vms[vid].bw) * 1e6 for vid in self.vm_ids], dtype=np.float64)

        # === host->vm索引映射 & padding ===
        self.host_to_vm_indices = {h: [] for h in self.host_ids}  # host_id -> List[global_vm_index]
        for j, vid in enumerate(self.vm_ids):
            h = self.vms[vid].host_id
            self.host_to_vm_indices[h].append(j)
        self.max_vms_per_host = max(len(vs) for vs in self.host_to_vm_indices.values()) if self.num_hosts > 0 else 0

        # 评分槽与启发式
        if (
            (max_ready_tasks is None)
            or (isinstance(max_ready_tasks, (int, float)) and max_ready_tasks <= 0)
            or (isinstance(max_ready_tasks, str) and max_ready_tasks.lower() == "auto")
        ):
            # 方案1：自动推导 max_ready_tasks = W * Nmax
            self.max_ready = _auto_max_ready_tasks(self.dax_paths, self.workflows_per_episode)
        else:
            self.max_ready = int(max_ready_tasks)

        self.max_ready = max(1, int(self.max_ready))
        self.task_heur_dim = 5  # [FCFS, SJF, MCF, HUR, EDF]

        # 状态包含（保留参数）
        self.state_include_vm = bool(state_include_vm)
        self.state_include_host = bool(state_include_host)
        self.state_include_queue = bool(state_include_queue)

        # ====== Manager 动态权重（raw + normalized）======
        self.manager_raw_w = np.ones(self.task_heur_dim, dtype=np.float32)

        if combo_weights is not None:
            w = np.asarray(combo_weights, dtype=float).reshape(-1)
            assert w.size == self.task_heur_dim, "combo_weights 维度需等于 5（FCFS/SJF/MCF/HUR/EDF）"
            w = np.clip(w, 0.0, None)
            if np.allclose(w.sum(), 0.0):
                w = np.ones_like(w)
            self.manager_raw_w = w.astype(np.float32)

        self.combo_weights = (self.manager_raw_w / max(float(self.manager_raw_w.sum()), 1e-9)).astype(np.float32)

        # ====== 三层观测维度定义 ======
        self.block1_dim = 12
        self.block2_dim = 10
        self.vm_feat_dim = 5
        self.host_feat_dim = 5

        # 中层 HostAgent：Block1 + Block2 + HostBlock(5*num_hosts)
        self.host_obs_dim = self.block1_dim + self.block2_dim + self.host_feat_dim * self.num_hosts
        self.host_act_dim = self.num_hosts

        # 下层 VMAgent：Block1 + Block2 + HostFeat(5) + VMBlockLocal(5*max_vms_per_host)
        self.vm_obs_dim = self.block1_dim + self.block2_dim + self.host_feat_dim + self.vm_feat_dim * self.max_vms_per_host
        self.vm_act_dim = self.max_vms_per_host

        # 保留gym空间（不影响三层接口；便于打印）
        self.observation_space = spaces.Dict({
            "host_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.host_obs_dim,), dtype=np.float32),
            "host_mask": spaces.Box(low=0.0, high=1.0, shape=(self.host_act_dim,), dtype=np.float32),
            "vm_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.vm_obs_dim,), dtype=np.float32),
            "vm_mask": spaces.Box(low=0.0, high=1.0, shape=(self.vm_act_dim,), dtype=np.float32),
        })

        # 能耗统计（与原HRL一致）
        self.total_energy = 0.0
        self.lifetime_energy = 0.0
        self._records = []
        self.parents_loc_by_wf = {}
        self._energy_cache = 0.0

        # 保险丝阈值
        self._fuse_zero_assign_limit = 50
        self._fuse_no_event_limit = 10

        # 运行态缓冲
        self._reset_internal_buffers()

    # ---------------- L1 HRL 接口（改造） ----------------
    def _normalize_manager_raw(self):
        s = float(np.sum(self.manager_raw_w))
        if s <= 1e-12:
            self.manager_raw_w[:] = 1.0
            s = float(np.sum(self.manager_raw_w))
        self.combo_weights = (self.manager_raw_w / max(s, 1e-9)).astype(np.float32)

    def set_manager_combo(self, weights):
        """
        兼容旧接口：直接设置权重（会写入 raw_w，并归一化到 combo_weights）
        """
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        assert w.size == self.task_heur_dim, "combo 权重长度应为 5"
        w = np.clip(w, 0.0, None)
        if np.allclose(w.sum(), 0.0):
            w = np.ones_like(w)
        self.manager_raw_w = w.astype(np.float32)
        self._normalize_manager_raw()

        self._phase_started = False
        self._phase_tasks = []

    def apply_manager_delta(self, delta_vec):
        """
        ✅ 新接口：对 5 维 raw 权重做增量更新（±0.1/0），并归一化到 combo_weights
        - w_raw ← max(0, w_raw + delta)
        - 若全 0：回退全 1
        """
        d = np.asarray(delta_vec, dtype=np.float32).reshape(-1)
        assert d.size == self.task_heur_dim, "delta_vec 长度必须为 5"
        self.manager_raw_w = np.maximum(0.0, self.manager_raw_w + d).astype(np.float32)
        self._normalize_manager_raw()

        self._phase_started = False
        self._phase_tasks = []

    def get_manager_action_mask(self) -> np.ndarray:
        """
        ✅ Manager 动作 mask（长度 243）
        - 若某维 w_raw == 0，则禁止该维再减（delta<0）的动作
        """
        w = self.manager_raw_w.reshape(1, -1)  # (1,5)
        delta = MANAGER_ACTION_TABLE           # (243,5)
        invalid = (w <= 1e-12) & (delta < 0.0) # (243,5)
        mask = (~invalid.any(axis=1)).astype(np.float32)  # (243,)
        return mask

    def get_manager_state(self):
        """
        ✅ state：原 10 维系统状态 + 当前权重（归一化后的 5 维）
        """
        now = getattr(self, "current_time", 0.0)

        ready_cnt   = float(len(self.ready_task_ids))
        waiting_cnt = float(sum(1 for s in self.task_state if s == "unReady"))
        running_cnt = float(sum(1 for s in self.task_state if s == "Running"))

        waits = [max(0.0, now - self.task_ready_time[tid]) for tid in self.ready_task_ids]
        avg_wait = float(np.mean(waits)) if len(waits) > 0 else 0.0
        max_wait = float(np.max(waits)) if len(waits) > 0 else 0.0

        ready_ratio   = ready_cnt   / max(1.0, self.max_ready)
        waiting_ratio = waiting_cnt / max(1.0, len(self.task_state) if len(self.task_state) > 0 else 1.0)
        running_ratio = running_cnt / max(1.0, self.num_vms)

        idle_vms = float(np.sum(self._vm_idle_mask()))
        idle_vm_ratio = idle_vms / max(1.0, self.num_vms)

        host_load_list, host_running_ratio_list = [], []
        for h in self.host_ids:
            active_pc = 0.0
            running = 0
            cnt = 0
            soonest = None
            for j, vid in enumerate(self.vm_ids):
                if self.vms[vid].host_id == h:
                    cnt += 1
                    if self.vm_available_at[j] > now + 1e-12:
                        active_pc += self.vms[vid].pc
                        running += 1
                    if soonest is None or self.vm_available_at[j] < soonest:
                        soonest = self.vm_available_at[j]
            host_load_list.append(_safe_div(active_pc, max(self.hosts[h].total_pc, 1e-9)))
            host_running_ratio_list.append(_safe_div(running, max(cnt, 1)))

        host_load_mean = float(np.mean(host_load_list)) if host_load_list else 0.0
        host_running_ratio_mean = float(np.mean(host_running_ratio_list)) if host_running_ratio_list else 0.0

        pc_mean = float(np.mean([self.vms[v].pc for v in self.vms])) if len(self.vms) > 0 else 1.0
        bw_mean = float(np.mean([self.vms[v].bw for v in self.vms])) if len(self.vms) > 0 else 1.0
        bw_bps  = bw_mean * 1e6

        # slack 统计用工作流级 deadline（wf.deadline）
        slacks = []
        for tid in self.ready_task_ids:
            exp_t = (
                self.task_in_bits[tid]  / max(bw_bps, 1e-9)
                + self.task_mi[tid]     / max(pc_mean, 1.0)
                + self.task_out_bits[tid]/ max(bw_bps, 1e-9)
            )
            wf_idx, _ = self.task_meta[tid]
            wf = self.workflows[wf_idx]
            dl = getattr(wf, "deadline", None)
            if dl is None:
                continue
            slacks.append(dl - (now + exp_t))

        if len(slacks) > 0:
            min_slack = float(np.min(slacks))
            avg_slack = float(np.mean(slacks))
            min_slack_norm = min_slack / max(self.horizon, 1.0)
            avg_slack_norm = avg_slack / max(self.horizon, 1.0)
        else:
            min_slack_norm = 0.0
            avg_slack_norm = 0.0

        vec = np.array([
            ready_ratio, waiting_ratio, running_ratio,
            avg_wait / max(self.horizon, 1.0),
            max_wait / max(self.horizon, 1.0),
            idle_vm_ratio, host_load_mean, host_running_ratio_mean,
            min_slack_norm, avg_slack_norm
        ], dtype=np.float32)

        w_norm = self.combo_weights.astype(np.float32)  # (5,)
        return np.concatenate([vec, w_norm], axis=0).astype(np.float32)

    # ---------------- L2/L3 三层接口 ----------------
    def get_host_state_for_next_assignment(self):
        """
        返回：({"obs": host_obs, "mask": host_mask}, has_next)

        - 若本阶段未开始：按combo_weights构建全量ready任务顺序
        - 若无“可立即分配”：has_next=False
        - 内部会弹出队首任务，缓存为 self._cur_tid，等待 host_select()
        """
        if self.done_flag:
            return {"obs": np.zeros(self.host_obs_dim, np.float32),
                    "mask": np.zeros(self.host_act_dim, np.float32)}, False

        if not self._phase_started:
            self._phase_prepare_tasks()
            self._phase_started = True
            self._phase_assign_cnt = 0
            self._phase_assigned_tids = []
            self._phase_size_sum_mi = 0.0

        if not self._has_decision_point():
            return {"obs": np.zeros(self.host_obs_dim, np.float32),
                    "mask": np.zeros(self.host_act_dim, np.float32)}, False

        if len(self._phase_tasks) == 0:
            return {"obs": np.zeros(self.host_obs_dim, np.float32),
                    "mask": np.zeros(self.host_act_dim, np.float32)}, False

        self._cur_tid = int(self._phase_tasks.pop(0))
        self._cur_host_id = None

        obs, mask = self._build_host_obs_for_task(self._cur_tid)
        if np.sum(mask) <= 0.0:
            self._phase_tasks.insert(0, self._cur_tid)
            self._cur_tid = None
            return {"obs": np.zeros(self.host_obs_dim, np.float32),
                    "mask": np.zeros(self.host_act_dim, np.float32)}, False

        return {"obs": obs.astype(np.float32), "mask": mask.astype(np.float32)}, True

    def host_select(self, host_index: int):
        assert hasattr(self, "_cur_tid") and self._cur_tid is not None, "请先调用 get_host_state_for_next_assignment()"
        hi = int(host_index)
        if hi < 0 or hi >= self.num_hosts:
            raise ValueError(f"host_index out of range: {hi}")
        host_id = self.host_ids[hi]

        has_idle = False
        now = self.current_time
        for j in self.host_to_vm_indices[host_id]:
            if self.vm_available_at[j] <= now + 1e-9:
                has_idle = True
                break
        if not has_idle:
            raise ValueError(f"Selected host {host_id} has no idle VM at time {now:.6f}")

        self._cur_host_id = host_id

    def get_vm_state_for_current_task(self):
        if self.done_flag:
            return {"obs": np.zeros(self.vm_obs_dim, np.float32),
                    "mask": np.zeros(self.vm_act_dim, np.float32)}, False

        if not hasattr(self, "_cur_tid") or self._cur_tid is None:
            return {"obs": np.zeros(self.vm_obs_dim, np.float32),
                    "mask": np.zeros(self.vm_act_dim, np.float32)}, False

        if self._cur_host_id is None:
            return {"obs": np.zeros(self.vm_obs_dim, np.float32),
                    "mask": np.zeros(self.vm_act_dim, np.float32)}, False

        obs, mask = self._build_vm_obs_for_task_host(self._cur_tid, self._cur_host_id)
        if np.sum(mask) <= 0.0:
            return {"obs": np.zeros(self.vm_obs_dim, np.float32),
                    "mask": np.zeros(self.vm_act_dim, np.float32)}, False

        return {"obs": obs.astype(np.float32), "mask": mask.astype(np.float32)}, True

    def vm_assign(self, vm_slot_index: int):
        assert hasattr(self, "_cur_tid") and self._cur_tid is not None, "请先调用 get_host_state_for_next_assignment()"
        assert self._cur_host_id is not None, "请先调用 host_select()"

        tid = int(self._cur_tid)
        host_id = int(self._cur_host_id)

        slot = int(vm_slot_index)
        if slot < 0 or slot >= self.max_vms_per_host:
            r_host = -1.0
            r_vm = -1.0
            info = {"invalid": 1, "reason": "vm_slot_out_of_range"}
            self._cur_tid = None
            self._cur_host_id = None
            return float(r_host), float(r_vm), info

        vm_list = self.host_to_vm_indices[host_id]
        if slot >= len(vm_list):
            r_host = -1.0
            r_vm = -1.0
            info = {"invalid": 1, "reason": "vm_slot_padding"}
            self._cur_tid = None
            self._cur_host_id = None
            return float(r_host), float(r_vm), info

        vm_global_idx = int(vm_list[slot])

        now = self.current_time
        if self.vm_available_at[vm_global_idx] > now + 1e-9:
            r_host = -1.0
            r_vm = -1.0
            info = {"invalid": 1, "reason": "vm_not_idle"}
            self._cur_tid = None
            self._cur_host_id = None
            return float(r_host), float(r_vm), info

        vm_id = self.vm_ids[vm_global_idx]
        pc_mi_s = self.vms[vm_id].pc

        mi       = self.task_mi[tid]

        duration = self._task_duration(tid, vm_global_idx)

        start_time = max(now, float(self.vm_available_at[vm_global_idx]))
        finish_time = start_time + duration

        E_hat = self._estimate_marginal_energy_proxy(
            host_id=host_id,
            start_time=start_time,
            duration=duration,
            added_pc=pc_mi_s,
        )
        energy_per_mi = E_hat / max(mi, 1e-9)
        energy_norm = _clip01(energy_per_mi / max(self.energy_norm_per_mi_ref, 1e-9))

        # ✅ 任务 deadline：仍使用 task_baseline_finish（子 deadline）
        if tid < len(self.task_baseline_finish):
            deadline = float(self.task_baseline_finish[tid])
        else:
            deadline = float(finish_time)
        lateness = max(0.0, float(finish_time - deadline))
        delay_norm = _clip01(lateness / max(self.task_baseline_norm, 1e-9))

        r_host = -((1.0 - float(self.alpha_delay_host)) * energy_norm + float(self.alpha_delay_host) * delay_norm)
        r_vm   = -((1.0 - float(self.alpha_delay_vm))   * energy_norm + float(self.alpha_delay_vm)   * delay_norm)

        self._assign_task_to_specific_vm(tid, vm_global_idx)

        self._phase_assign_cnt += 1
        self._phase_assigned_tids.append(tid)
        self._phase_size_sum_mi += float(mi)

        info = {
            "invalid": 0,
            "task_id": tid,
            "host_id": host_id,
            "vm_global_idx": vm_global_idx,
            "start_time": float(start_time),
            "finish_time": float(finish_time),
            "deadline": float(deadline),
            "lateness": float(lateness),
            "delay_norm": float(delay_norm),
            "E_hat": float(E_hat),
            "energy_per_mi": float(energy_per_mi),
            "energy_norm": float(energy_norm),
        }

        self._cur_tid = None
        self._cur_host_id = None

        return float(r_host), float(r_vm), info

    def finish_phase_and_advance(self):
        if self._phase_assign_cnt == 0 and self._has_decision_point():
            self._fuse_zero_assign_streak += 1
            if self._fuse_zero_assign_streak >= self._fuse_zero_assign_limit:
                self._fuse_zero_assign_streak = 0
                if len(self.ready_task_ids) > 0 and np.any(self._vm_idle_mask()):
                    forced_tid = int(self.ready_task_ids[0])
                    forced_vm = int(self._vm_serial_order()[0])
                    if self._vm_idle_mask()[forced_vm]:
                        self._assign_task_to_specific_vm(forced_tid, forced_vm)
                        self._phase_assign_cnt += 1
                        self._phase_assigned_tids.append(forced_tid)
                        self._phase_size_sum_mi += float(self.task_mi[forced_tid])
        else:
            self._fuse_zero_assign_streak = 0

        r_energy_phase = self._advance_until_decision_energy_only()

        if self._phase_assign_cnt > 0 and self._phase_size_sum_mi > 0:
            r_manager_raw = r_energy_phase / float(self._phase_size_sum_mi)
        else:
            r_manager_raw = 0.0

        info = {
            "current_time": self.current_time,
            "assign_cnt": int(self._phase_assign_cnt),
            "phase_size_sum_mi": float(self._phase_size_sum_mi),
            "r_energy_phase": float(r_energy_phase),
            "phase_assigned_tids": list(self._phase_assigned_tids),
        }

        self._phase_started = False
        self._phase_tasks = []
        self._phase_assigned_tids = []
        self._phase_assign_cnt = 0
        self._phase_size_sum_mi = 0.0
        self._cur_tid = None
        self._cur_host_id = None

        return float(r_manager_raw), info

    # ---------------- Gym reset/render ----------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.random_seed = int(seed)
        super().reset(seed=seed)
        self._reset_internal_buffers()

        # 上层初始权重：固定 seed=12345 的随机初值（可复现）
        rs = np.random.RandomState(12345)
        w0 = rs.uniform(low=0.0, high=1.0, size=self.task_heur_dim).astype(np.float32)
        if float(w0.sum()) <= 1e-12:
            w0[:] = 1.0
        self.manager_raw_w = w0
        self._normalize_manager_raw()

        self.arrival_times = poisson_arrival_times(
            self.arrival_lambda,
            n_max=self.workflows_per_episode,
            seed=self.random_seed,
        )

        if self.workflows_per_episode is not None:
            n = int(self.workflows_per_episode)
            if n > 0 and len(self.arrival_times) > n:
                self.arrival_times = self.arrival_times[:n]

        self.first_arrival_time = float(self.arrival_times[0]) if self.arrival_times else 0.0
        self.next_arrival_idx = 0

        _ = self._advance_until_decision_energy_only()

        self._phase_started = False
        self._phase_tasks = []
        self._phase_assigned_tids = []
        self._phase_assign_cnt = 0
        self._phase_size_sum_mi = 0.0
        self._cur_tid = None
        self._cur_host_id = None

        self._fuse_zero_assign_streak = 0
        self._fuse_no_event_streak = 0

        info = {
            "current_time": getattr(self, "current_time", 0.0),
            "first_arrival_time": self.first_arrival_time,
            "wf_target": self.workflows_per_episode if self.workflows_per_episode is not None else None,
        }
        obs = {
            "host_obs": np.zeros(self.host_obs_dim, np.float32),
            "host_mask": np.zeros(self.host_act_dim, np.float32),
            "vm_obs": np.zeros(self.vm_obs_dim, np.float32),
            "vm_mask": np.zeros(self.vm_act_dim, np.float32),
        }
        return obs, info

    def render(self):
        idle_vms = int(np.sum(self._vm_idle_mask()))
        print(f"[t={self.current_time:.2f}] ready={len(self.ready_task_ids)} idleVM={idle_vms} done={self.done_flag}")

    # ---------------- 运行态缓冲 ----------------
    def _reset_internal_buffers(self):
        self.current_time = 0.0
        self.done_flag = False

        self.workflows = []
        self.arrival_times = []
        self.next_arrival_idx = 0

        self.task_meta = []
        self.task_state = []
        self.task_parents = []
        self.task_children = []
        self.task_mi = []
        self.task_in_bits = []
        self.task_out_bits = []
        self.task_up_rank = []
        self.task_down_rank = []
        self.task_ready_time = []
        self.task_end_time = []

        self.ready_task_ids = []
        self.event_heap = []
        self.vm_available_at = np.zeros(self.num_vms, dtype=np.float64)

        self.completed_workflows = 0
        self.wf_remaining_tasks = {}

        # ✅ 任务级 deadline（用于 reward）
        self.task_baseline_finish = []

        # ✅ 工作流级完成/超时统计
        self.wf_finish_time = {}     # wf_id -> abs finish_time
        self.wf_lateness = {}        # wf_id -> max(0, finish - wf_deadline)
        self.wf_timeout = {}         # wf_id -> bool
        self.ep_wf_lateness_sum = 0.0

        self.total_energy = 0.0
        self._records = []
        self.parents_loc_by_wf = {}
        self._energy_cache = 0.0

        self._cur_tid = None
        self._cur_host_id = None

    # ---------------- 调度逻辑与停止条件 ----------------
    def _no_more_arrivals(self):
        return self.next_arrival_idx >= len(self.arrival_times)

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

            pc_mean = float(np.mean([self.vms[v].pc for v in self.vms]))
            bw_mean = float(np.mean([self.vms[v].bw for v in self.vms]))
            wf.compute_upward_ranks(pc_mean, bw_mean)
            wf.compute_downward_ranks(pc_mean, bw_mean)

            self.wf_remaining_tasks[wf.workflow_id] = len(tasks)

            # ========== ✅ 1) 工作流级 ddl：arrival + 1.5 * HEFT_makespan ==========
            heft_ms, heft_finish_rel, avg_dur = _heft_makespan_and_finish_times(
                tasks, self._heft_vm_pc, self._heft_vm_bw_bps
            )
            alpha = 1.5
            wf_deadline_abs = float(arr_t + alpha * float(heft_ms))
            wf.deadline = wf_deadline_abs
            wf.heft_makespan = float(heft_ms)  # 可选记录

            # ========== ✅ 2) 任务级 ddl：由 wf ddl 倒推 ==========
            n_local = len(tasks)
            topo, succ = _topo_sort_local(tasks)

            # 以平均 duration 做倒推（与 rank/HEFT 一致口径）
            P = avg_dur.reshape(-1).astype(np.float64)  # [n]
            D_wf_rel = float(alpha * float(heft_ms))

            LF = np.zeros((n_local,), dtype=np.float64)
            LS = np.zeros((n_local,), dtype=np.float64)

            is_sink = np.array([len(succ[i]) == 0 for i in range(n_local)], dtype=bool)

            # 先给 sink 初始化
            for i in range(n_local):
                if is_sink[i]:
                    LF[i] = D_wf_rel
                    LS[i] = LF[i] - P[i]

            # 反向拓扑倒推
            for u in reversed(topo):
                if is_sink[u]:
                    continue
                if len(succ[u]) > 0:
                    min_LS_child = float(np.min(LS[np.array(succ[u], dtype=np.int32)]))
                    LF[u] = min_LS_child
                else:
                    LF[u] = D_wf_rel
                # 保证 deadline 不早于 HEFT 下该任务的完成时刻（相对）
                LF[u] = max(float(LF[u]), float(heft_finish_rel[u]))
                LS[u] = LF[u] - P[u]

            # sink 也做一次 clamp（避免极端数值）
            for u in range(n_local):
                if is_sink[u]:
                    LF[u] = max(float(LF[u]), float(heft_finish_rel[u]))
                    LS[u] = LF[u] - P[u]

            # 将任务绝对 deadline 追加到全局 task_baseline_finish
            base = len(self.task_meta)

            for local_id in range(n_local):
                abs_deadline = float(arr_t + LF[local_id])
                self.task_baseline_finish.append(abs_deadline)

            # ========== 任务扁平化写入 env arrays ==========
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

    def _vm_idle_mask(self):
        return self.vm_available_at <= self.current_time + 1e-9

    def _has_decision_point(self):
        return (len(self.ready_task_ids) > 0) and np.any(self._vm_idle_mask())

    def _episode_should_end(self):
        if self.workflows_per_episode is None:
            return (
                self.current_time >= self.horizon
                and len(self.event_heap) == 0
                and self._no_more_arrivals()
            )
        else:
            all_arrived = self._no_more_arrivals()
            all_done = self.completed_workflows >= self.workflows_per_episode
            no_running = len(self.event_heap) == 0
            no_ready = len(self.ready_task_ids) == 0
            return all_arrived and all_done and no_running and no_ready

    # ---------------- 时间推进（仅真实能耗差分） ----------------
    def _energy_reward_to_current_time(self) -> float:
        started = len(self.workflows) > 0
        if not started:
            return 0.0
        t = self.current_time
        clipped = []
        for r in self._records:
            if r.end_time <= 0.0:
                continue
            if r.start_time >= t:
                continue
            end = min(r.end_time, t)
            if end <= r.start_time:
                continue
            clipped.append(LoadRecord(r.start_time, end, r.server_id, r.vm_pc))

        E = energy_from_records(
            clipped, self.hosts, start_time=self.first_arrival_time, end_time=t
        )
        dE = max(0.0, E - self._energy_cache)
        self._energy_cache = E
        self.total_energy += dE
        self.lifetime_energy += dE
        return -(dE * self.energy_reward_scale)

    def _advance_until_decision_energy_only(self) -> float:
        r_energy = 0.0
        while True:
            self._add_workflow_if_arrived()
            if self._has_decision_point():
                self._fuse_no_event_streak = 0
                break

            next_times = []
            if len(self.event_heap) > 0:
                next_times.append(self.event_heap[0][0])
            if not self._no_more_arrivals():
                next_times.append(self.arrival_times[self.next_arrival_idx])

            if len(next_times) == 0:
                self._fuse_no_event_streak += 1
                if self._episode_should_end() or self._fuse_no_event_streak >= self._fuse_no_event_limit:
                    self.done_flag = True
                    break
                self.current_time += 1e-9
                r_energy += self._energy_reward_to_current_time()
                continue

            next_t = min(next_times)
            if next_t <= self.current_time + 1e-12:
                next_t = self.current_time + 1e-9

            self.current_time = next_t
            r_energy += self._energy_reward_to_current_time()
            self._process_finish_events_at_current_time()

            if self._has_decision_point():
                self._fuse_no_event_streak = 0
                break
            if self._episode_should_end() or self.current_time >= self.horizon:
                self.done_flag = True
                break
        return r_energy

    def _process_finish_events_at_current_time(self):
        while len(self.event_heap) > 0 and self.event_heap[0][0] <= self.current_time + 1e-12:
            _, etype, task_id, vm_idx = heapq.heappop(self.event_heap)
            if etype != "finish":
                continue
            self.task_state[task_id] = "Finished"
            self.task_end_time[task_id] = self.current_time

            wf_id, _ = self.task_meta[task_id]
            if wf_id in self.wf_remaining_tasks:
                self.wf_remaining_tasks[wf_id] -= 1
                if self.wf_remaining_tasks[wf_id] == 0:
                    self.completed_workflows += 1

                    # ✅ 工作流完成：记录 workflow lateness（sum max(0, finish - wf_deadline)）
                    wf = self.workflows[wf_id]
                    finish_t = float(self.current_time)
                    self.wf_finish_time[wf_id] = finish_t
                    wf_finish_late = max(0.0, finish_t - float(getattr(wf, "deadline", finish_t)))
                    self.wf_lateness[wf_id] = float(wf_finish_late)
                    self.wf_timeout[wf_id] = bool(finish_t > float(getattr(wf, "deadline", finish_t)))
                    self.ep_wf_lateness_sum += float(wf_finish_late)

            wf_idx, _ = self.task_meta[task_id]
            for child_local in self.task_children[task_id]:
                child_gid = None
                for gid, meta in enumerate(self.task_meta):
                    if meta[0] == wf_idx and meta[1] == child_local:
                        child_gid = gid
                        break
                if child_gid is None:
                    continue
                if self.task_state[child_gid] == "unReady":
                    all_ok = True
                    for p_local in self.task_parents[child_gid]:
                        p_gid = None
                        for gid, meta in enumerate(self.task_meta):
                            if meta[0] == wf_idx and meta[1] == p_local:
                                p_gid = gid
                                break
                        if p_gid is None or self.task_state[p_gid] != "Finished":
                            all_ok = False
                            break
                    if all_ok:
                        self.task_state[child_gid] = "Ready"
                        self.task_ready_time[child_gid] = self.current_time
                        self.ready_task_ids.append(child_gid)

        self.ready_task_ids = [tid for tid in self.ready_task_ids if self.task_state[tid] == "Ready"]

    # ---------------- 阶段构建（排序模板） ----------------
    def _compute_task_heuristics_for_ready(self):
        now = self.current_time
        sel_ids = list(self.ready_task_ids)
        if len(sel_ids) == 0:
            return [], np.zeros((0, self.task_heur_dim), dtype=np.float32)

        pc_mean = float(np.mean([self.vms[v].pc for v in self.vms]))
        bw_mean = float(np.mean([self.vms[v].bw for v in self.vms]))
        bw_mean_bps = bw_mean * 1e6

        time_scale = 1.0 + np.mean(np.array(self.task_mi, dtype=np.float64) / max(pc_mean, 1.0))

        feats = np.zeros((len(sel_ids), self.task_heur_dim), dtype=np.float32)
        for i, tid in enumerate(sel_ids):
            exp_t = (
                self.task_in_bits[tid]  / max(bw_mean_bps, 1e-9)
                + self.task_mi[tid]     / max(pc_mean, 1.0)
                + self.task_out_bits[tid]/ max(bw_mean_bps, 1e-9)
            )
            fcfs = max(0.0, now - self.task_ready_time[tid])
            sjf  = 1.0 / (exp_t + 1.0)
            mcf  = float(len(self.task_children[tid]))
            hur  = float(self.task_up_rank[tid])

            # ✅ EDF：使用工作流级 deadline（wf.deadline）
            wf_idx, _ = self.task_meta[tid]
            wf = self.workflows[wf_idx]
            if getattr(wf, "deadline", None) is not None:
                slack = max(0.0, float(wf.deadline) - (now + exp_t))
                edf = 1.0 / (slack + 1.0)
            else:
                edf = 0.0

            if self.normalize:
                fcfs = fcfs / max(self.horizon, 1.0)
                hur  = hur  / max(time_scale, 1.0)
                mcf  = mcf  / max(1.0, np.mean([len(ch) for ch in self.task_children]) + 1e-6)

            feats[i] = np.array([fcfs, sjf, mcf, hur, edf], dtype=np.float32)
        return sel_ids, feats

    def _phase_prepare_tasks(self):
        self._add_workflow_if_arrived()
        sel_ids, feats = self._compute_task_heuristics_for_ready()
        if feats.shape[0] == 0:
            self._phase_tasks = []
            return
        scores = feats @ self.combo_weights.reshape(-1, 1)
        order = np.argsort(-scores.squeeze(1))
        self._phase_tasks = [int(sel_ids[i]) for i in order.tolist()]

    # ---------------- 三层观测构造 ----------------
    # （以下内容保持不变）
    def _compute_block2(self):
        now = self.current_time
        ready_cnt   = float(len(self.ready_task_ids))
        waiting_cnt = float(sum(1 for s in self.task_state if s == "unReady"))
        running_cnt = float(sum(1 for s in self.task_state if s == "Running"))

        waits = [max(0.0, now - self.task_ready_time[x]) for x in self.ready_task_ids]
        avg_wait = float(np.mean(waits)) if len(waits) > 0 else 0.0
        max_wait = float(np.max(waits)) if len(waits) > 0 else 0.0

        sum_mi = float(np.sum([self.task_mi[x] for x in self.ready_task_ids])) if len(self.ready_task_ids) > 0 else 0.0
        urgent_cnt = 0


        slacks = []
        for x in self.ready_task_ids:
            exp_t = float(np.mean([self._task_duration(x, j) for j in range(self.num_vms)]))
            wf_idx, _ = self.task_meta[x]
            wf = self.workflows[wf_idx]
            dl = getattr(wf, "deadline", None)
            if dl is not None:
                s = float(dl) - (now + exp_t)
                slacks.append(s)
                if s < 0:
                    urgent_cnt += 1

        if len(slacks) > 0:
            min_slack = float(np.min(slacks))
            avg_slack = float(np.mean(slacks))
        else:
            min_slack = 0.0
            avg_slack = 0.0

        block2 = np.array([
            ready_cnt / max(1.0, self.max_ready),
            waiting_cnt / max(1.0, len(self.task_state) if len(self.task_state) > 0 else 1.0),
            running_cnt / max(1.0, self.num_vms),
            avg_wait / max(self.horizon, 1.0),
            max_wait / max(self.horizon, 1.0),
            sum_mi / max(1.0, np.mean(self.task_mi) if len(self.task_mi) > 0 else 1.0),
            float(urgent_cnt) / max(1.0, len(self.ready_task_ids)),
            min_slack / max(self.horizon, 1.0),
            avg_slack / max(self.horizon, 1.0),
            float(len(set([self.task_meta[x][0] for x in self.ready_task_ids]))) / max(
                1.0, len(self.workflows) if len(self.workflows) > 0 else 1.0
            ),
        ], dtype=np.float32)
        return block2

    def _compute_block1(self, tid: int):
        now = self.current_time
        in_bits  = float(self.task_in_bits[tid])
        out_bits = float(self.task_out_bits[tid])
        mi       = float(self.task_mi[tid])

        io_ratio = _safe_div(in_bits + out_bits, max(mi, 1.0))
        wait_norm = (now - self.task_ready_time[tid]) / max(self.horizon, 1.0)
        children_cnt_n = float(len(self.task_children[tid])) / max(
            1.0, np.mean([len(ch) for ch in self.task_children]) + 1e-6
        )
        up_rank_n = self.task_up_rank[tid] / max(
            1.0, np.mean(self.task_up_rank) if len(self.task_up_rank) > 0 else 1.0
        )
        down_rank_n = self.task_down_rank[tid] / max(
            1.0, np.mean(self.task_down_rank) if len(self.task_down_rank) > 0 else 1.0
        )

        vm_idle = self._vm_idle_mask()
        idle_indices = np.where(vm_idle)[0].tolist()
        if len(idle_indices) == 0:
            est_best = est_mean = 0.0
        else:
            durations = []
            for j in idle_indices:
                durations.append(self._task_duration(tid, j))
            est_best = float(np.min(durations))
            est_mean = float(np.mean(durations))

        wf_idx, _ = self.task_meta[tid]
        wf = self.workflows[wf_idx]
        if getattr(wf, "deadline", None) is not None:
            slack_on_best = float(wf.deadline) - (now + est_best)
        else:
            slack_on_best = 0.0

        same_wf_ready_frac = 0.0
        if len(self.ready_task_ids) > 0:
            same = [x for x in self.ready_task_ids if self.task_meta[x][0] == wf_idx]
            same_wf_ready_frac = float(len(same)) / float(max(1, len(self.ready_task_ids)))

        block1 = np.array([
            mi / max(1.0, np.mean(self.task_mi) if len(self.task_mi) > 0 else 1.0),
            in_bits / max(1.0, np.mean(self.task_in_bits) if len(self.task_in_bits) > 0 else 1.0),
            out_bits / max(1.0, np.mean(self.task_out_bits) if len(self.task_out_bits) > 0 else 1.0),
            io_ratio,
            wait_norm,
            children_cnt_n,
            up_rank_n,
            down_rank_n,
            est_best / max(self.horizon, 1.0),
            est_mean / max(self.horizon, 1.0),
            slack_on_best / max(self.horizon, 1.0),
            same_wf_ready_frac,
        ], dtype=np.float32)
        return block1, in_bits, out_bits, mi

    def _compute_host_block(self):
        now = self.current_time
        host_stats = {
            "active_pc": {h: 0.0 for h in self.host_ids},
            "idle_vm":   {h: 0 for h in self.host_ids},
            "soonest":   {h: None for h in self.host_ids},
            "running":   {h: 0 for h in self.host_ids},
            "vm_cnt":    {h: 0 for h in self.host_ids},
        }
        for j, vid in enumerate(self.vm_ids):
            h = self.vms[vid].host_id
            host_stats["vm_cnt"][h] += 1
            if self.vm_available_at[j] > now + 1e-12:
                host_stats["active_pc"][h] += self.vms[vid].pc
                host_stats["running"][h]   += 1
            else:
                host_stats["idle_vm"][h]   += 1
            if host_stats["soonest"][h] is None or self.vm_available_at[j] < host_stats["soonest"][h]:
                host_stats["soonest"][h] = self.vm_available_at[j]

        mean_total_pc = max(1.0, float(np.mean([self.hosts[x].total_pc for x in self.host_ids]))) if self.num_hosts > 0 else 1.0
        host_block = np.zeros((self.num_hosts, self.host_feat_dim), dtype=np.float32)
        for hi, h in enumerate(self.host_ids):
            vm_cnt = max(1, host_stats["vm_cnt"][h])
            total_pc_n = self.hosts[h].total_pc / mean_total_pc
            host_load = _safe_div(host_stats["active_pc"][h], max(self.hosts[h].total_pc, 1e-9))
            host_idle_vm = float(host_stats["idle_vm"][h]) / vm_cnt
            soonest = host_stats["soonest"][h] if host_stats["soonest"][h] is not None else now
            host_soonest_delay = max(0.0, float(soonest - now))
            host_running_ratio = _safe_div(host_stats["running"][h], vm_cnt)
            host_block[hi] = np.array([
                total_pc_n,
                host_load,
                host_idle_vm,
                host_soonest_delay / max(self.horizon, 1e-9),
                host_running_ratio,
            ], dtype=np.float32)
        return host_block

    def _build_host_obs_for_task(self, tid: int):
        block1, _, _, _ = self._compute_block1(tid)
        block2 = self._compute_block2()
        host_block = self._compute_host_block()

        obs = np.concatenate([block1, block2, host_block.reshape(-1)], axis=0)

        mask = np.zeros((self.host_act_dim,), dtype=np.float32)
        now = self.current_time
        for hi, h in enumerate(self.host_ids):
            ok = False
            for j in self.host_to_vm_indices[h]:
                if self.vm_available_at[j] <= now + 1e-9:
                    ok = True
                    break
            if ok:
                mask[hi] = 1.0
        return obs, mask

    def _build_vm_obs_for_task_host(self, tid: int, host_id: int):
        block1, in_bits, out_bits, mi = self._compute_block1(tid)
        block2 = self._compute_block2()
        host_block_all = self._compute_host_block()
        hi = self.host_ids.index(host_id)
        host_feat = host_block_all[hi].astype(np.float32)

        mean_vm_pc = max(1.0, float(np.mean([self.vms[v].pc for v in self.vms])))
        mean_vm_bw = max(1.0, float(np.mean([self.vms[v].bw for v in self.vms])))

        vm_block = np.zeros((self.max_vms_per_host, self.vm_feat_dim), dtype=np.float32)
        mask = np.zeros((self.vm_act_dim,), dtype=np.float32)

        now = self.current_time
        vm_list = self.host_to_vm_indices[host_id]
        for slot in range(self.max_vms_per_host):
            if slot >= len(vm_list):
                continue
            j = vm_list[slot]
            vid = self.vm_ids[j]
            vm = self.vms[vid]
            idle_flag = 1.0 if self.vm_available_at[j] <= now + 1e-12 else 0.0
            avail_delay = max(0.0, float(self.vm_available_at[j] - now))
            pred_finish = max(now, float(self.vm_available_at[j])) + self._task_duration(tid, j)

            vm_block[slot] = np.array([
                idle_flag,
                avail_delay / max(self.horizon, 1e-9),
                vm.pc / mean_vm_pc,
                vm.bw / mean_vm_bw,
                pred_finish / max(self.horizon, 1e-9),
            ], dtype=np.float32)

            if idle_flag > 0.5:
                mask[slot] = 1.0

        obs = np.concatenate([block1, block2, host_feat, vm_block.reshape(-1)], axis=0)
        return obs, mask

    # ---------------- 能耗代理（路线A关键） ----------------
    def _estimate_marginal_energy_proxy(self, host_id: int, start_time: float, duration: float, added_pc: float) -> float:
        if duration <= 0.0:
            return 0.0
        host = self.hosts[host_id]
        total_pc = max(host.total_pc, 1e-9)

        active_pc_before = 0.0
        for j in self.host_to_vm_indices[host_id]:
            if self.vm_available_at[j] > start_time + 1e-12:
                vid = self.vm_ids[j]
                active_pc_before += float(self.vms[vid].pc)

        load_before = _safe_div(active_pc_before, total_pc)
        load_after  = _safe_div(active_pc_before + float(added_pc), total_pc)

        load_before = float(np.clip(load_before, 0.0, 1.0))
        load_after  = float(np.clip(load_after,  0.0, 1.0))

        P_before = float(host.power(load_before))
        P_after  = float(host.power(load_after))
        dP = max(0.0, P_after - P_before)
        return float(dP * duration)

    # ---------------- 分配/辅助 ----------------
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

    def _assign_task_to_specific_vm(self, task_id, vm_index):
        vm_id = self.vm_ids[vm_index]
        now = self.current_time
        pc_mi_s = self.vms[vm_id].pc


        start_time = max(now, float(self.vm_available_at[vm_index]))
        end_time = start_time + self._task_duration(task_id, vm_index)

        self.task_state[task_id] = "Running"
        self.task_end_time[task_id] = end_time
        self.vm_available_at[vm_index] = end_time
        if task_id in self.ready_task_ids:
            self.ready_task_ids.remove(task_id)
        heapq.heappush(self.event_heap, (end_time, "finish", task_id, vm_index))

        host_id = self.vms[vm_id].host_id
        self._records.append(LoadRecord(start_time, end_time, host_id, pc_mi_s))
        wf_id, local_id = self.task_meta[task_id]
        self.parents_loc_by_wf.setdefault(int(wf_id), {})[int(local_id)] = (int(vm_id), int(host_id))

    def _vm_serial_order(self):
        avail = np.round(self.vm_available_at, 12)
        neg_pc = -np.array([self.vms[vid].pc for vid in self.vm_ids], dtype=np.float64)
        idx = np.arange(self.num_vms, dtype=np.int64)
        order = np.lexsort((idx, neg_pc, avail))
        return order
