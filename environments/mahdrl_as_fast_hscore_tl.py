"""Global VM actions over HScore's unchanged per-host fixed-slot observations."""
import numpy as np
from environments.mahdrl_as_fast_hscore import CloudWorkflowEnv_VMAgents as HScoreEnv
from environments.mahdrl_fcfs import MANAGER_ACTION_TABLE
from environments.mahdrl_heft import spaces
from environments.mahdrl_oe_ge_apf_score import VariableArraySpace
from algorithms.mahdrl_oe_ge_apf_score_fast.packing import frozen_array
from algorithms.mahdrl_as_fast_hscore_tl.config import (
    ALGORITHM, VM_CAPACITY, VM_INPUT_DIM, variant_metadata,
)


class CloudWorkflowEnv_VMAgents(HScoreEnv):
    algorithm = ALGORITHM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_vm_act_dim = self.num_hosts*VM_CAPACITY
        self.observation_space = spaces.Dict({"vm_obs": VariableArraySpace((VM_INPUT_DIM,)),
            "vm_mask": VariableArraySpace(binary=True)})

    def variant_metadata(self):
        metadata = super().variant_metadata()
        for key in ("mechanism", "va_preview", "ha_epsilon_clock"):
            metadata.pop(key, None)
        metadata.update(variant_metadata())
        return metadata

    @staticmethod
    def _empty_grouped():
        return dict(obs=np.zeros((0, VM_INPUT_DIM), np.float32), mask=np.zeros(0, np.float32),
            performance_times=np.zeros(0, np.float64), candidate_ids=[], row_host_ids=())

    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        empty = self._empty_grouped()
        return dict(vm_obs=empty["obs"], vm_mask=empty["mask"]), info

    def _build_global_vm_state_for_task(self, tid):
        if self._cur_host_id is not None:
            raise RuntimeError("TL global observation requires no preselected host")
        self._ensure_decision(tid)
        if "grouped_global_state" in self._fast_cache:
            return self._fast_cache["grouped_global_state"]
        with self.profiler.stage("global_vm_observation"):
            local = self.preview_vm_states()
            obs = np.zeros((self.num_hosts, VM_INPUT_DIM), np.float32)
            mask = np.zeros((self.num_hosts, VM_CAPACITY), np.float32)
            times = np.zeros_like(mask, dtype=np.float64)
            ids = []
            for row, hid in enumerate(self.host_ids):
                indices = self.host_to_vm_indices[hid]
                ids.extend([self.vm_ids[j] for j in indices] + [None]*(VM_CAPACITY-len(indices)))
                if hid in local:
                    obs[row], mask[row], times[row] = local[hid]["obs"], local[hid]["mask"], local[hid]["performance_times"]
            state = dict(obs=frozen_array(obs), mask=frozen_array(mask.ravel()),
                performance_times=frozen_array(times.ravel(), dtype=np.float64),
                candidate_ids=tuple(ids), row_host_ids=tuple(self.host_ids))
            self._fast_cache["grouped_global_state"] = state
            return state

    def get_vm_state_for_next_assignment(self):
        _, ok = super().get_host_state_for_next_assignment()
        if not ok:
            return self._empty_grouped(), False
        return self._build_global_vm_state_for_task(self._cur_tid), True

    def host_select(self, host_index):
        raise RuntimeError("HScore TL has no HA action; use vm_assign_global")

    def vm_assign_global(self, action_index):
        if self._cur_tid is None:
            raise RuntimeError("Obtain a global VM decision before assigning")
        state = self._build_global_vm_state_for_task(self._cur_tid)
        if (isinstance(action_index, (bool, np.bool_)) or not isinstance(action_index, (int, np.integer))
                or not 0 <= int(action_index) < len(state["mask"]) or state["mask"][int(action_index)] <= .5):
            raise ValueError("Illegal, busy or padded global VM action")
        row, slot = divmod(int(action_index), VM_CAPACITY)
        hid = self.host_ids[row]
        j = self.host_to_vm_indices[hid][slot]
        legal_vm_count = int(state["mask"].sum())
        legal_host_count = int((state["mask"].reshape(-1, VM_CAPACITY) > .5).any(1).sum())
        self._cur_host_id = hid
        _, reward, info = super().vm_assign(slot)
        info.update(vm_id=int(self.vm_ids[j]), global_action_index=int(action_index),
            legal_vm_count=legal_vm_count, legal_host_count=legal_host_count)
        self.apf_decisions[-1].update(global_action_index=int(action_index),
            vm_selection_scope="global_grouped_fixed9", global_retained_vm_count=legal_vm_count)
        return reward, info
