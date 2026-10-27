"""IRWS mechanisms adapted to the common deadline-constrained cloud problem."""
from dataclasses import dataclass
import networkx as nx
import numpy as np
from common.workflow_opt import exec_time_components
from environments.irws import IRWSEnvRouteA
from .config import REWARD_PROTOCOL


@dataclass(frozen=True)
class GraphData:
    wf_id: int
    gids: tuple
    features: tuple
    parents: tuple
    upward: tuple
    downward: tuple
    reference_time: tuple


def paper_ranks(graph, tasks, mean_pc, mean_bw_mbps):
    """Literal Eq.(12)/(13), seconds. Empty predecessor maximum is zero."""
    order=list(nx.topological_sort(graph));up=np.zeros(len(tasks));down=up.copy()
    incoming=[sum((getattr(t,'parent_in_bits',{}) or {}).values()) for t in tasks]
    reference=np.array([t.workload_mi/mean_pc+incoming[i]/(mean_bw_mbps*1e6) for i,t in enumerate(tasks)])
    def edge(i,j):return float((tasks[j].parent_in_bits or {}).get(i,0.))/(mean_bw_mbps*1e6)
    for i in reversed(order):
        up[i]=tasks[i].workload_mi/mean_pc+max((edge(i,j)+up[j] for j in graph.successors(i)),default=0.)
    for j in order:
        down[j]=max((edge(i,j)+down[i]+tasks[i].workload_mi/mean_pc for i in graph.predecessors(j)),default=0.)
    return up,down,reference


def fusion_weight(reference_time, downward_rank):
    """Eq.(30) unchanged for positive rank; entry-only finite boundary convention."""
    if not np.isfinite([reference_time,downward_rank]).all() or min(reference_time,downward_rank)<0:
        raise ValueError('Invalid physical time/rank')
    if downward_rank==0:
        return (1. if reference_time>0 else 0.), True
    return float(reference_time/downward_rank), False


class PaperCloudEnv(IRWSEnvRouteA):
    """FIFO queues and periodic policy ticks on the unchanged physical calculators.

    Adapted settings: 10 waiting tasks per VM, 5-second policy interval.
    FIFO is nonpreemptive. Reservation energy is charged once at dispatch; queued
    tasks have load records only on their predicted, deterministic service interval.
    """
    def __init__(self, *args, queue_capacity=10, slot_duration=5., **kwargs):
        if isinstance(queue_capacity,bool) or int(queue_capacity)!=queue_capacity or queue_capacity<1:
            raise ValueError('queue_capacity must be a positive integer')
        if not np.isfinite(slot_duration) or slot_duration<=0:
            raise ValueError('slot_duration must be finite and positive')
        self.queue_capacity=int(queue_capacity);self.slot_duration=float(slot_duration)
        super().__init__(*args,**kwargs)

    def reset(self, **kwargs):
        self._invalidate_resources()
        self.vm_queues={i:[] for i in range(self.num_vms)}
        self.vm_running={i:None for i in range(self.num_vms)}
        self.task_dispatch_time={};self.interval_index=0;self._paper_graphs={}
        self._initializing=True
        try: result=super().reset(**kwargs)
        finally: self._initializing=False
        if not self.done_flag:
            self.interval_index=self._tick_at_or_after(self.current_time)
            self._advance_physics(self.interval_index*self.slot_duration)
        return result

    def _invalidate_resources(self):
        # Equal timestamps can still have different queues after a dispatch.
        self.state_revision=getattr(self,'state_revision',0)+1
        self._resource_cache={}

    def _tick_at_or_after(self, time):
        return int(np.ceil(float(time)/self.slot_duration-1e-10))

    def queue_mask(self):
        return np.asarray([len(self.vm_queues[i])<self.queue_capacity for i in range(self.num_vms)],dtype=np.float32)

    def can_dispatch(self):
        return not self.done_flag and bool(self.ready_task_ids) and bool(self.queue_mask().any())

    def _advance_until_decision(self):
        if not getattr(self,'_initializing',False) and not self.can_dispatch():
            self.advance_interval()

    def _next_completion_time(self):
        times=[self.task_end_time[g] for g in self.vm_running.values() if g is not None]
        return min(times,default=None)

    def _finish_and_start(self):
        # Multiple zero-duration virtual tasks, if present, finish at the same event.
        for _ in range(len(self.task_state)+1):
            finished=[vm for vm,g in self.vm_running.items() if g is not None and self.task_end_time[g]<=self.current_time+1e-12]
            if not finished: return
            super()._process_completions_at_time(self.current_time)
            for vm in finished:
                self.vm_running[vm]=None
                if self.vm_queues[vm]:
                    gid=self.vm_queues[vm].pop(0)
                    if abs(self.task_start_time[gid]-self.current_time)>1e-8:
                        raise RuntimeError('FIFO service start differs from deterministic reservation')
                    self.vm_running[vm]=gid;self.task_state[gid]='Running'
        raise RuntimeError('Queue completion loop did not settle')

    def _check_done(self):
        return (super()._check_done() and not any(self.vm_queues.values())
                and all(g is None for g in self.vm_running.values()))

    def _advance_physics(self, target):
        if target<self.current_time-1e-10: raise ValueError('Physical time cannot go backwards')
        self._invalidate_resources()
        while True:
            self._add_workflows_up_to_current_time()
            self._finish_and_start()
            if self._check_done():
                self._finalize_energy_if_done();self.done_flag=True;return
            if self.current_time>=target-1e-12: return
            candidates=[float(target)]
            if self.next_arrival_idx<len(self.arrival_times):
                candidates.append(float(self.arrival_times[self.next_arrival_idx]))
            completion=self._next_completion_time()
            if completion is not None: candidates.append(float(completion))
            nxt=min(candidates)
            if nxt<=self.current_time: raise RuntimeError('Unfinished queue has no future physical event')
            self.current_time=nxt

    def advance_interval(self):
        """Skip only ticks with no dispatch opportunity; keep exact physical event times."""
        while not self.done_flag:
            target_index=self.interval_index+1
            if not self.can_dispatch():
                events=[]
                if self.next_arrival_idx<len(self.arrival_times): events.append(float(self.arrival_times[self.next_arrival_idx]))
                completion=self._next_completion_time()
                if completion is not None: events.append(float(completion))
                if not events: raise RuntimeError('Unfinished episode has no arrival or completion')
                target_index=max(target_index,self._tick_at_or_after(min(events)))
            self._advance_physics(target_index*self.slot_duration)
            self.interval_index=target_index
            if self.can_dispatch(): return

    def finish_phase_and_advance(self):
        self.advance_interval()

    def get_ready_task_candidates(self):
        gids=sorted({i for i in self.ready_task_ids if self.task_state[i]=='Ready'},
                    key=lambda i:(self.task_ready_time[i],i))
        return dict(gids=gids,mask=np.ones(len(gids),dtype=np.float32))

    def select_task(self,task_gid):
        if not 0<=task_gid<len(self.task_state) or self.task_state[task_gid]!='Ready':raise ValueError('Task dependency is not satisfied')
        super().select_task(task_gid)

    def scale(self,value,key):
        return self.state_normalizer.scale(value,key) if self.state_normalizer else float(value)

    def graphs(self):
        mean_pc=float(np.mean(self._vm_pc));mean_bw=float(np.mean(self._vm_bw))
        for wf in self.workflows:
            if wf.workflow_id in self._paper_graphs:continue
            gids=tuple(i for i,ti in enumerate(self.task_meta) if ti.wf_id==wf.workflow_id)
            up,down,ref=paper_ranks(wf.graph,wf.tasks,mean_pc,mean_bw)
            # Paper ranks differ from F18's historical ranks: scale seconds by common T_ref.
            features=tuple((self.scale(u,'T_ref'),self.scale(d,'T_ref')) for u,d in zip(up,down))
            self._paper_graphs[wf.workflow_id]=GraphData(wf.workflow_id,gids,features,
                tuple(tuple(wf.graph.predecessors(i)) for i in range(len(wf.tasks))),tuple(up),tuple(down),tuple(ref))
        return tuple(self._paper_graphs[w.workflow_id] for w in self.workflows)

    def resource_features(self,gid):
        if gid in self._resource_cache:return self._resource_cache[gid]
        task=self._task_obj_from_gid(gid);parents=self._build_parents_loc(gid);now=float(self.current_time)
        result=[self.scale(task.workload_mi,'workload_mi_ref')]
        for vm_id in range(self.num_vms):
            vm=self.vms[vm_id]
            duration=exec_time_components(task,vm.pc,vm.bw,parents_loc=parents,
                child_vm_id=vm_id,child_host_id=vm.host_id,zero_if_same_vm=True,zero_if_same_host=False)[-1]
            transfer=float(task.ext_in_bits or 0.)+sum(float(bits) for p,bits in (task.parent_in_bits or {}).items()
                if p not in parents or parents[p][0]!=vm_id)
            result.extend([float(len(self.vm_queues[vm_id]))/self.queue_capacity,
                self.scale(max(0.,self.vm_available_at[vm_id]-now),'T_ref'),self.scale(duration,'T_ref'),
                self.scale(vm.pc,'vm_pc_ref'),self.scale(vm.bw,'vm_bw_ref'),self.scale(transfer,'input_bits_ref')])
        result=np.asarray(result,dtype=np.float32)
        if not np.isfinite(result).all():raise ValueError('Nonfinite paper resource observation')
        result.setflags(write=False)
        self._resource_cache[gid]=result
        return result

    def get_vm_obs_and_mask(self):
        if self.current_task_gid is None: raise RuntimeError('Select a Ready task first')
        return dict(obs=self.resource_features(self.current_task_gid),mask=self.queue_mask())

    def paper_step(self, action):
        if self.current_task_gid is None: raise RuntimeError('Select a Ready task first')
        if not 0<=action<self.num_vms or not self.queue_mask()[action]:
            raise ValueError('Illegal VM action or full waiting queue; no physical mutation')
        gid=int(self.current_task_gid)
        if self.task_state[gid]!='Ready': raise ValueError('Task is not Ready')
        self._invalidate_resources()
        ti=self.task_meta[gid]
        graph=next(g for g in self.graphs() if g.wf_id==ti.wf_id)
        before=float(self.total_energy);busy=self.vm_running[action] is not None
        self.ready_task_ids.remove(gid)
        _,info=self._assign_task_to_vm(gid,action)
        self.task_dispatch_time[gid]=float(self.current_time)
        if busy:
            self.task_state[gid]='Queued';self.vm_queues[action].append(gid)
        else: self.vm_running[action]=gid
        self.current_task_gid=None
        delta=float(self.total_energy-before)
        external=(1-self.alpha_delay_vm)*(-delta*self.energy_reward_scale)+self.alpha_delay_vm*info['r_delay']
        weight,boundary=fusion_weight(graph.reference_time[ti.local_id],graph.downward[ti.local_id])
        info.update(step_reward=float(external),outer_cost=-float(external),delta_energy_total=delta,
                    r_energy_total=-delta*self.energy_reward_scale,dispatch=float(self.current_time),
                    interval=self.interval_index,queued=busy,fusion_weight=weight,fusion_boundary=boundary,
                    reward_protocol=REWARD_PROTOCOL,reward_mode='host_task_window_v1')
        if not np.isfinite([external,weight]).all(): raise ValueError('Nonfinite restored reward')
        return float(external),False,info

    def step(self, action):
        reward,done,info=self.paper_step(action)
        return np.zeros(1,dtype=np.float32),reward,done,False,info
