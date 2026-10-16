# -*- coding: utf-8 -*-
"""
resource_opt.py
说明：单位约定
- VM.pc  ：处理能力（MI/s）
- VM.bw  ：带宽（Mb/s）→ 需要换算：bits/s = bw * 1e6
"""

from dataclasses import dataclass, field

@dataclass
class VM:
    vm_id: int
    host_id: int
    pc: float      # 处理能力（MI/s）
    bw: float      # 带宽（Mb/s）

@dataclass
class Host:
    host_id: int
    vm_ids: list = field(default_factory=list)
    power_model: object = None
    total_pc: float = 0.0
    total_bw: float = 0.0

    def power(self, load_ratio: float) -> float:
        if load_ratio < 0.0:
            load_ratio = 0.0
        if load_ratio > 1.0:
            load_ratio = 1.0
        if self.power_model is None:
            return 0.0
        return self.power_model(load_ratio)

def _power_model_factory(host_id):
    mod = host_id % 3

    def nec(load):
        if load <= 0.1: return load * 44 + 15.9
        if load <= 0.2: return load * 21 + 18.2
        if load <= 0.3: return load * 20 + 18.4
        if load <= 0.4: return load * 28 + 16.0
        if load <= 0.5: return load * 26 + 16.8
        if load <= 0.6: return load * 32 + 13.8
        if load <= 0.7: return load * 38 + 10.2
        if load <= 0.8: return load * 27 + 17.9
        if load <= 0.9: return load * 31 + 14.7
        return load * 25 + 20.1

    def fujitsu(load):
        if load <= 0.1: return load * 44 + 13.1
        if load <= 0.2: return load * 30 + 14.5
        if load <= 0.3: return load * 25 + 15.5
        if load <= 0.4: return load * 24 + 15.8
        if load <= 0.5: return load * 28 + 14.2
        if load <= 0.6: return load * 37 + 9.7
        if load <= 0.7: return load * 54 - 0.5
        if load <= 0.8: return load * 72 - 13.1
        if load <= 0.9: return load * 70 - 11.5
        return load * 46 + 10.1

    def lenovo(load):
        if load <= 0.1: return load * 32 + 16.8
        if load <= 0.2: return load * 28 + 17.2
        if load <= 0.3: return load * 32 + 16.4
        if load <= 0.4: return load * 39 + 14.3
        if load <= 0.5: return load * 51 + 9.5
        if load <= 0.6: return load * 54 + 8.0
        if load <= 0.7: return load * 59 + 5.0
        if load <= 0.8: return load * 93 - 18.8
        if load <= 0.9: return load * 132 - 50.0
        return load * 184 - 96.8

    return nec if mod == 0 else (fujitsu if mod == 1 else lenovo)

def create_cluster(
    num_hosts=3,
    vms_per_host=(3, 3, 4),
    pc_tiers=(2.0, 4.0, 8.0),
    bw_tiers=(2000.0, 4000.0, 8000.0),
):
    hosts, vms = {}, {}
    vm_counter = 0
    for h in range(num_hosts):
        hosts[h] = Host(host_id=h, power_model=_power_model_factory(h))
        vm_count_this_host = vms_per_host[h % len(vms_per_host)]
        for i in range(vm_count_this_host):
            tier = i % len(pc_tiers)
            vm = VM(vm_counter, h, pc_tiers[tier], bw_tiers[tier])
            vms[vm_counter] = vm
            hosts[h].vm_ids.append(vm_counter)
            hosts[h].total_pc += vm.pc
            hosts[h].total_bw += vm.bw
            vm_counter += 1
    return hosts, vms
