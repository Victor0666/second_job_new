# -*- coding: utf-8 -*-
"""
env_nohost.py

两层版本（Manager + Global-VMAgent）：

- 保留 DDLCACHE_FCFS 中的：
  1) workflow deadline cache_fcfs + 每个 workflow 独立采样 ddlAlpha
  2) task 子deadline 的 Latest-Finish 倒推
  3) Manager 的 phase reward:
       r_mgr = - [ alpha * phase_delay_cost + (1-alpha) * phase_energy_cost ]

- 去掉 HostAgent：
  Manager 完成 phase 内 ready tasks 排序后，
  直接由 VMAgent 在“全局 idle VM”上选择一个 VM。

设计原则：
- 尽量不改你原来的 reward / deadline / energy accounting / phase 推进逻辑
- 因此：
  * Manager reward 完全沿用父类
  * lower reward 仍沿用原 VMAgent 口径：
        r_vm = (1-alpha_delay_vm) * r_energy_host + alpha_delay_vm * r_delay
    只是动作从“host 内 slot”改成“全局 vm index”
"""

import numpy as np

from environments.mahdrl_fcfs import (
    CloudWorkflowEnv_VMAgents as _BaseEnv,
    MANAGER_ACTION_TABLE,
)


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


class CloudWorkflowEnv_VMAgents(_BaseEnv):
    """
    继承你当前的 DDLCACHE_FCFS 环境，只增加：
    1) get_vm_state_for_next_assignment()
    2) vm_assign_global(vm_global_idx)

    其余 workflow deadline / manager reward / energy accounting / phase 逻辑都不动。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.host_id_to_pos = {h: i for i, h in enumerate(self.host_ids)}

        # 每个全局VM的特征维度：
        # [idle_flag, avail_delay, vm_pc_norm, vm_bw_norm, pred_finish_norm,
        #  host_load, host_idle_vm_ratio, host_running_ratio]
        self.global_vm_feat_dim = 8
        self.global_vm_obs_dim = self.block1_dim + self.block2_dim + self.global_vm_feat_dim * self.num_vms
        self.global_vm_act_dim = self.num_vms

    # ------------------------------------------------------------------
    # 构造“当前任务 -> 全局所有VM”的状态
    # ------------------------------------------------------------------
    def _build_global_vm_obs_for_task(self, tid: int):
        block1, in_bits, out_bits, mi = self._compute_block1(tid)
        block2 = self._compute_block2()
        host_block = self._compute_host_block()   # [num_hosts, 5]

        mean_vm_pc = max(1.0, float(np.mean([self.vms[v].pc for v in self.vms])))
        mean_vm_bw = max(1.0, float(np.mean([self.vms[v].bw for v in self.vms])))

        vm_block = np.zeros((self.num_vms, self.global_vm_feat_dim), dtype=np.float32)
        mask = np.zeros((self.global_vm_act_dim,), dtype=np.float32)

        now = self.current_time
        for j, vid in enumerate(self.vm_ids):
            vm = self.vms[vid]
            hid = vm.host_id
            hi = self.host_id_to_pos[hid]

            idle_flag = 1.0 if self.vm_available_at[j] <= now + 1e-12 else 0.0
            avail_delay = max(0.0, float(self.vm_available_at[j] - now))

            pred_finish = max(now, float(self.vm_available_at[j])) + self._task_duration(tid, j)

            # host_block 每行含义：
            # [total_pc_n, host_load, host_idle_vm, host_soonest_delay, host_running_ratio]
            vm_block[j] = np.array([
                idle_flag,
                avail_delay / max(self.horizon, 1e-9),
                vm.pc / mean_vm_pc,
                vm.bw / mean_vm_bw,
                pred_finish / max(self.horizon, 1e-9),
                float(host_block[hi, 1]),  # host_load
                float(host_block[hi, 2]),  # host_idle_vm ratio
                float(host_block[hi, 4]),  # host_running ratio
            ], dtype=np.float32)

            if idle_flag > 0.5:
                mask[j] = 1.0

        obs = np.concatenate([block1, block2, vm_block.reshape(-1)], axis=0).astype(np.float32)
        return obs, mask.astype(np.float32)

    # ------------------------------------------------------------------
    # 直接给“下一次分配”的全局VM状态
    # ------------------------------------------------------------------
    def get_vm_state_for_next_assignment(self):
        if self.done_flag:
            return {
                "obs": np.zeros(self.global_vm_obs_dim, np.float32),
                "mask": np.zeros(self.global_vm_act_dim, np.float32),
            }, False

        if not self._phase_started:
            self._phase_prepare_tasks()
            self._phase_started = True
            self._phase_assign_cnt = 0
            self._phase_assigned_tids = []
            self._phase_size_sum_mi = 0.0

            self._phase_preadvanced = False
            self._phase_preadvance_r_energy_total = 0.0
            self._phase_preadvance_r_energy_by_host = {int(h): 0.0 for h in self.host_ids}

        if not self._has_decision_point():
            return {
                "obs": np.zeros(self.global_vm_obs_dim, np.float32),
                "mask": np.zeros(self.global_vm_act_dim, np.float32),
            }, False

        if len(self._phase_tasks) == 0:
            return {
                "obs": np.zeros(self.global_vm_obs_dim, np.float32),
                "mask": np.zeros(self.global_vm_act_dim, np.float32),
            }, False

        self._cur_tid = int(self._phase_tasks.pop(0))
        self._cur_host_id = None

        obs, mask = self._build_global_vm_obs_for_task(self._cur_tid)
        if np.sum(mask) <= 0.0:
            self._phase_tasks.insert(0, self._cur_tid)
            self._cur_tid = None
            return {
                "obs": np.zeros(self.global_vm_obs_dim, np.float32),
                "mask": np.zeros(self.global_vm_act_dim, np.float32),
            }, False

        return {"obs": obs.astype(np.float32), "mask": mask.astype(np.float32)}, True

    # ------------------------------------------------------------------
    # 直接按“全局 VM 索引”完成一次 assignment
    # ------------------------------------------------------------------
    def vm_assign_global(self, vm_global_idx: int):
        assert hasattr(self, "_cur_tid") and self._cur_tid is not None, "请先调用 get_vm_state_for_next_assignment()"

        j = int(vm_global_idx)
        if j < 0 or j >= self.num_vms:
            info = {"invalid": 1, "reason": "vm_global_idx_out_of_range"}
            self._cur_tid = None
            self._cur_host_id = None
            return -1.0, info

        now = self.current_time
        if self.vm_available_at[j] > now + 1e-9:
            info = {"invalid": 1, "reason": "vm_not_idle"}
            self._cur_tid = None
            self._cur_host_id = None
            return -1.0, info

        tid = int(self._cur_tid)
        vm_id = self.vm_ids[j]
        host_id = int(self.vms[vm_id].host_id)
        self._cur_host_id = host_id  # 仅为兼容内部逻辑/信息记录


        mi = self.task_mi[tid]

        duration = self._task_duration(tid, j)

        start_time = max(now, float(self.vm_available_at[j]))
        finish_time = start_time + duration

        if tid < len(self.task_baseline_finish):
            deadline = float(self.task_baseline_finish[tid])
        else:
            deadline = float(finish_time)

        lateness = max(0.0, float(finish_time - deadline))
        delay_norm = _clip01(lateness / max(self.task_baseline_norm, 1e-9))
        r_delay = -float(delay_norm)

        # 真实分配
        self._assign_task_to_specific_vm(tid, j)

        self._phase_assign_cnt += 1
        self._phase_assigned_tids.append(tid)
        self._phase_size_sum_mi += float(mi)

        if self._phase_has_immediate_next_assignment():
            r_energy_total = 0.0
            r_energy_host = 0.0
        else:
            r_energy_total, r_energy_by_host = self._advance_until_decision_energy_only(return_breakdown=True)
            r_energy_host = float(r_energy_by_host.get(host_id, 0.0))

            self._phase_preadvanced = True
            self._phase_preadvance_r_energy_total = float(r_energy_total)
            self._phase_preadvance_r_energy_by_host = {
                int(h): float(r_energy_by_host.get(h, 0.0)) for h in self.host_ids
            }

        # 为了尽量保持“下层口径不变”，仍沿用原始 VMAgent reward
        r_vm = (1.0 - float(self.alpha_delay_vm)) * float(r_energy_host) + float(self.alpha_delay_vm) * float(r_delay)

        info = {
            "invalid": 0,
            "task_id": tid,
            "host_id": host_id,
            "vm_global_idx": j,
            "start_time": float(start_time),
            "finish_time": float(finish_time),
            "deadline": float(deadline),
            "lateness": float(lateness),
            "delay_norm": float(delay_norm),
            "r_delay": float(r_delay),
            "r_energy_total": float(r_energy_total),
            "r_energy_host": float(r_energy_host),
            "lower_reward_mode": "nohost_global_vm",
        }

        self._cur_tid = None
        self._cur_host_id = None
        return float(r_vm), info