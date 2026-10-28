"""Fast ablation scheduling with no host or VM APF and original OE accounting."""
import numpy as np
from environments.mahdrl_oe_ge_apf_score_fast_nha import CloudWorkflowEnv_VMAgents as AllIdleHostsEnv
from environments.mahdrl_oe_ge_apf_score import VariableArraySpace
from environments.mahdrl_heft import spaces
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from algorithms.mahdrl_oe_ge_apf_score_fast_tl.config import (
    ALGORITHM, VM_CONTEXT_DIM, VM_CANDIDATE_DIM, variant_metadata,
)


class CloudWorkflowEnv_VMAgents(AllIdleHostsEnv):
    algorithm = ALGORITHM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Local dimensions stay intact for the inherited physical assignment path.
        self.global_vm_act_dim = self.num_vms
        self.observation_space = spaces.Dict({
            "vm_obs": spaces.Dict({
                "context": spaces.Box(-np.inf, np.inf, (VM_CONTEXT_DIM,), np.float32),
                "candidates": VariableArraySpace((VM_CANDIDATE_DIM,)),
                "exists_mask": VariableArraySpace(binary=True)}),
            "vm_mask": VariableArraySpace(binary=True)})

    def variant_metadata(self):
        return dict(super().variant_metadata(), **variant_metadata())

    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        empty = self._empty(VM_CONTEXT_DIM, VM_CANDIDATE_DIM)
        return dict(vm_obs=empty["obs"], vm_mask=empty["mask"]), info

    def _build_global_vm_state_for_task(self, tid):
        self._ensure_decision(tid)
        if "global_vm_state" in self._fast_cache:
            return self._fast_cache["global_vm_state"]
        with self.profiler.stage("global_vm_observation"):
            candidates = np.zeros((self.num_vms, VM_CANDIDATE_DIM), np.float32)
            mask = np.zeros(self.num_vms, np.float32)
            context = None
            b = self.base_vm_obs_dim
            for hid in self.host_ids:
                x, local_mask = self._build_vm_obs_for_task_host(tid, hid)
                indices = self.host_to_vm_indices[hid]
                k = len(indices)
                if context is None:
                    context = np.concatenate((x[:22], x[b:b+1])).astype(np.float32)
                vm_features = np.concatenate((x[27:b].reshape(self.max_vms_per_host, 5)[:k],
                    x[b+3+self.num_hosts:].reshape(self.max_vms_per_host, 2)[:k]), axis=1)
                candidates[indices] = np.concatenate((vm_features,
                    np.broadcast_to(x[22:27], (k, 5)),
                    np.broadcast_to(x[b+1+self.num_hosts:b+3+self.num_hosts], (k, 2))), axis=1)
                mask[indices] = local_mask[:k]
            s = self._apf_snapshot(tid)
            exists = np.ones(self.num_vms, np.float32)
            times = np.array([s["rows"].get(j, (np.inf,))[0] for j in range(self.num_vms)])
            for value in (context, candidates, exists, mask, times):
                value.flags.writeable = False
            state = dict(obs=dict(context=context, candidates=candidates, exists_mask=exists),
                mask=mask, candidate_ids=list(self.vm_ids), performance_times=times)
            self._fast_cache["global_vm_state"] = state
            return state

    def get_vm_state_for_next_assignment(self):
        empty = lambda: (self._empty(VM_CONTEXT_DIM, VM_CANDIDATE_DIM), False)
        if self.done_flag:
            return empty()
        if not self._phase_started:
            self._phase_prepare_tasks()
            self._phase_started = True
            self._phase_assign_cnt = 0
            self._phase_assigned_tids = []
            self._phase_size_sum_mi = 0.0
            self._phase_preadvanced = False
            self._phase_preadvance_r_energy_total = 0.0
            self._phase_preadvance_r_energy_by_host = {int(h): 0.0 for h in self.host_ids}
        if not self._has_decision_point() or not self._phase_tasks:
            return empty()
        self._cur_tid = int(self._phase_tasks.pop(0))
        self._cur_host_id = None
        state = self._build_global_vm_state_for_task(self._cur_tid)
        if not np.any(state["mask"] > .5):
            self._phase_tasks.insert(0, self._cur_tid)
            self._cur_tid = None
            return empty()
        return state, True

    def host_select(self, host_index):
        raise RuntimeError("TL has no host selection; use vm_assign_global")

    def vm_assign_global(self, action_index):
        if self._cur_tid is None:
            raise RuntimeError("Call get_vm_state_for_next_assignment before assigning")
        state = self._build_global_vm_state_for_task(self._cur_tid)
        if (isinstance(action_index, (bool, np.bool_)) or not isinstance(action_index, (int, np.integer))
                or not 0 <= int(action_index) < self.num_vms or state["mask"][int(action_index)] <= .5):
            raise ValueError("Illegal or physically unavailable global VM action")
        j = int(action_index)
        hid = self.vms[self.vm_ids[j]].host_id
        slot = self.host_to_vm_indices[hid].index(j)
        legal_vm_count = int(state["mask"].sum())
        legal_host_count = len(self._apf_snapshot(self._cur_tid)["hosts"])
        self._cur_host_id = hid
        with self.profiler.stage("environment_assignment"):
            _, reward, info = super().vm_assign(slot)
        info.update(vm_id=int(self.vm_ids[j]), global_action_index=j,
                    legal_vm_count=legal_vm_count, legal_host_count=legal_host_count)
        self.apf_decisions[-1].update(global_action_index=j, vm_selection_scope="global",
            vm_apf_scope="disabled", global_retained_vm_count=legal_vm_count)
        return reward, info
