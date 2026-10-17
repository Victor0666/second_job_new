# -*- coding: utf-8 -*-
"""
workflow_opt.py
改动要点：
1) Workflow 的秩计算改用真实的数据量字段：
   - 输入：ext_in_bits + from_parents_bits
   - 输出：out_file_size_sum_bits
   - 计算：workload_mi / avg_proc_speed
2) 新增：执行时间组件估计函数 exec_time_components()
   - 根据 VM 的 pc（MI/s）与 bw（Mb/s）计算 (tau_in, tau_comp, tau_out, tau_exec)
   - 支持“与父任务同 VM / 同 Host 时通信视为 0”这样的近似策略
"""

import networkx as nx
from dataclasses import dataclass

BITS_PER_MB = 1024.0 * 1024.0 * 8.0

@dataclass
class LoadRecord:
    start_time: float
    end_time: float
    server_id: int
    vm_pc: float  # MI/s

class Workflow:
    def __init__(self, workflow_id, graph, tasks, arrival_time, source_device_id, deadline=None):
        self.workflow_id = workflow_id
        self.graph = graph
        self.tasks = tasks
        self.arrival_time = arrival_time
        self.source_device_id = source_device_id
        self.deadline = deadline
        self.finish_time = None
        self.is_arrived = False

    def _all_parents_finished(self, task):
        for pred in self.graph.predecessors(task.task_id):
            if self.tasks[pred].state != "Finished":
                return False
        return True

    def get_ready_tasks(self):
        ready = []
        for t in self.tasks:
            if t.state == "unReady" and self._all_parents_finished(t):
                t.state = "Ready"
            if t.state == "Ready":
                ready.append(t)
        return ready

    def compute_upward_ranks(self, avg_proc_speed_mi_s: float, avg_tx_speed_Mb_s: float):
        """向上秩：上传 + 计算 + 下载 + 后继最大秩"""
        for t in self.tasks:
            t.upward_rank = 0.0

        topo = list(nx.topological_sort(self.graph))
        avg_tx_bps = max(avg_tx_speed_Mb_s, 1e-9) * 1e6
        for tid in reversed(topo):
            t = self.tasks[tid]
            upload = (t.ext_in_bits + t.from_parents_bits) / avg_tx_bps
            download = t.out_file_size_sum_bits / avg_tx_bps
            comp = t.workload_mi / max(avg_proc_speed_mi_s, 1e-9)
            self_time = upload + comp + download

            succ_rank = 0.0
            for s in self.graph.successors(tid):
                succ_rank = max(succ_rank, self.tasks[s].upward_rank or 0.0)
            t.upward_rank = self_time + succ_rank

    def compute_downward_ranks(self, avg_proc_speed_mi_s: float, avg_tx_speed_Mb_s: float):
        """向下秩：上传 + 计算 + 下载 + 前驱最大秩"""
        for t in self.tasks:
            t.downward_rank = 0.0

        topo = list(nx.topological_sort(self.graph))
        avg_tx_bps = max(avg_tx_speed_Mb_s, 1e-9) * 1e6
        for tid in topo:
            t = self.tasks[tid]
            upload = (t.ext_in_bits + t.from_parents_bits) / avg_tx_bps
            download = t.out_file_size_sum_bits / avg_tx_bps
            comp = t.workload_mi / max(avg_proc_speed_mi_s, 1e-9)
            self_time = upload + comp + download

            pred_rank = 0.0
            for p in self.graph.predecessors(tid):
                pred_rank = max(pred_rank, self.tasks[p].downward_rank or 0.0)
            t.downward_rank = self_time + pred_rank


COMMUNICATION_MODEL = "same_vm_only_v1"


def validate_deadline_cache(obj, path):
    """Reject reference makespans computed under a different transfer model."""
    if obj.get("meta", {}).get("communication_model") != COMMUNICATION_MODEL:
        import re
        match = re.search(r"task([SML])_res([SML])", str(path))
        scenario = f" --task {match[1]} --resource {match[2]}" if match else " --task S --resource S"
        raise ValueError(
            f"Deadline cache uses an outdated communication model: {path}. "
            f"Regenerate with: .\\run.cmd common/precompute_deadlines.py{scenario}"
        )


def exec_time_components(
    task,
    vm_pc_mi_s: float,
    vm_bw_Mb_s: float,
    *,
    parents_loc: dict | None = None,      # {parent_task_id: (parent_vm_id, parent_host_id)}
    child_vm_id: int | None = None,
    child_host_id: int | None = None,
    zero_if_same_vm: bool = True,
    zero_if_same_host: bool = False,
):
    """
    计算 (tau_in, tau_comp, tau_out, tau_exec)
    - task.workload_mi：MI
    - 带宽换算：bits/s = vm_bw_Mb_s * 1e6
    - 若提供 parents_loc / child_{vm,host}，可在“同 VM（同主机不同 VM 不减免）”时将父→子传输记为 0。
      否则默认将 task.from_parents_bits 全部通过网络传输。
    """
    bps = max(vm_bw_Mb_s, 1e-9) * 1e6

    # 来自父任务的数据量（考虑同 VM 零传输的近似）
    from_parents_bits_eff = 0.0
    if parents_loc is None or (child_vm_id is None and child_host_id is None):
        from_parents_bits_eff = task.from_parents_bits
    else:
        for p, bits in task.parent_in_bits.items():
            vmh = parents_loc.get(p)
            if vmh is None:
                from_parents_bits_eff += bits
                continue
            p_vm, p_host = vmh
            same_vm = (child_vm_id is not None and p_vm == child_vm_id)
            same_host = (child_host_id is not None and p_host == child_host_id)
            if (same_vm and zero_if_same_vm) or (same_host and zero_if_same_host):
                # The experiment enables only the same-VM exemption.
                pass
            else:
                from_parents_bits_eff += bits

    tau_in = (task.ext_in_bits + from_parents_bits_eff) / bps
    tau_comp = task.workload_mi / max(vm_pc_mi_s, 1e-9)
    tau_out = task.out_file_size_sum_bits / bps
    tau_exec = tau_in + tau_comp + tau_out
    return tau_in, tau_comp, tau_out, tau_exec


def energy_from_records(records, hosts, *, start_time=None, end_time=None):
    """Integrate every host over one common window, including idle power.

    Explicit endpoints define the episode window. If omitted, use the common
    span of all records, rather than a different span for each host.
    """
    records = list(records)
    if start_time is None:
        start_time = min((r.start_time for r in records), default=0.0)
    if end_time is None:
        end_time = max((r.end_time for r in records), default=start_time)
    start_time, end_time = float(start_time), float(end_time)
    if end_time < start_time:
        raise ValueError("Energy window end must not precede its start")
    if end_time == start_time:
        return 0.0

    events = {hid: {start_time: 0.0, end_time: 0.0} for hid in hosts}
    for r in records:
        start = max(start_time, float(r.start_time))
        end = min(end_time, float(r.end_time))
        if end <= start:
            continue
        host_events = events[r.server_id]
        host_events[start] = host_events.get(start, 0.0) + float(r.vm_pc)
        host_events[end] = host_events.get(end, 0.0) - float(r.vm_pc)

    total_energy = 0.0
    for hid, host in hosts.items():
        load = 0.0
        previous = start_time
        for t, delta in sorted(events[hid].items()):
            ratio = max(0.0, min(1.0, load / max(float(host.total_pc), 1e-9)))
            total_energy += float(host.power(ratio)) * (t - previous)
            load += delta
            previous = t
    return float(total_energy)
