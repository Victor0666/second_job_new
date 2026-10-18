"""Paper-only interfaces over the unchanged common IRWS physical simulator."""
from dataclasses import dataclass
import copy
import networkx as nx
import numpy as np
from common.workflow_opt import exec_time_components
from environments.irws import IRWSEnvRouteA


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
    if not np.isfinite([reference_time,downward_rank]).all() or min(reference_time,downward_rank)<0:
        raise ValueError('Invalid physical time/rank in Eq.(30)')
    if downward_rank==0:
        # Eq.(13) has zero at entry nodes: use intrinsic-only at the boundary.
        return (1. if reference_time>0 else 0.), True
    # Preserve the printed formula, including finite >1 values; never silently clip it.
    return float(reference_time/downward_rank), False


class PaperCloudEnv(IRWSEnvRouteA):
    def reset(self,**kwargs):
        self._paper_graphs={};self._paper_finish_highwater={};self._paper_previous_finish={}
        return super().reset(**kwargs)

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
        task=self._task_obj_from_gid(gid);parents=self._build_parents_loc(gid);now=float(self.current_time)
        result=[self.scale(task.workload_mi,'workload_mi_ref')]
        for vm_id in range(self.num_vms):
            vm=self.vms[vm_id]
            duration=exec_time_components(task,vm.pc,vm.bw,parents_loc=parents,
                child_vm_id=vm_id,child_host_id=vm.host_id,zero_if_same_vm=True,zero_if_same_host=False)[-1]
            transfer=float(task.ext_in_bits or 0.)+sum(float(bits) for p,bits in (task.parent_in_bits or {}).items()
                if p not in parents or parents[p][0]!=vm_id)
            result.extend([float(self.vm_available_at[vm_id]>now+1e-12),
                self.scale(max(0.,self.vm_available_at[vm_id]-now),'T_ref'),self.scale(duration,'T_ref'),
                self.scale(vm.pc,'vm_pc_ref'),self.scale(vm.bw,'vm_bw_ref'),self.scale(transfer,'input_bits_ref')])
        result=np.asarray(result,dtype=np.float32)
        if not np.isfinite(result).all():raise ValueError('Nonfinite paper resource observation')
        return result

    def get_vm_obs_and_mask(self):
        if self.current_task_gid is None:raise RuntimeError('Select a Ready task first')
        if not np.any(self.vm_available_at<=self.current_time+1e-12):self._advance_until_decision()
        if self.done_flag:return dict(obs=np.zeros(1+6*self.num_vms,dtype=np.float32),mask=np.zeros(self.num_vms,dtype=np.float32))
        return dict(obs=self.resource_features(self.current_task_gid),
                    mask=(self.vm_available_at<=self.current_time+1e-12).astype(np.float32))

    def reference_cost(self,gid):
        task=self._task_obj_from_gid(gid);pc=float(np.mean(self._vm_pc));bw=float(np.mean(self._vm_bw))
        duration=exec_time_components(task,pc,bw,parents_loc=self._build_parents_loc(gid),
            child_vm_id=-1,child_host_id=-1,zero_if_same_vm=True,zero_if_same_host=False)[-1]
        # Counterfactuals use the same host integration; copies never mutate physical state.
        energies=[]
        for host in self.hosts:
            shadow=copy.copy(self);shadow._host_segments={h:list(v) for h,v in self._host_segments.items()}
            energies.append(shadow._add_interval_load(host,self.current_time,self.current_time+duration,pc))
        energy=float(np.mean(energies))
        if duration<=0 or energy<=0 or not np.isfinite([duration,energy]).all():
            raise ValueError('Pseudo-average server has no positive reference cost')
        return float(self.current_time+duration),energy

    def paper_step(self,action,alpha=.5):
        if self.current_task_gid is None:raise RuntimeError('Select a task first')
        if not 0<=action<self.num_vms or self.vm_available_at[action]>self.current_time+1e-12:
            raise ValueError('Illegal VM action; no physical mutation')
        gid=int(self.current_task_gid);ti=self.task_meta[gid];wf=self.workflows[ti.wf_id]
        graph=next(g for g in self.graphs() if g.wf_id==ti.wf_id)
        ref_finish,ref_energy=self.reference_cost(gid)
        _,_,terminated,truncated,info=super().step(action)
        actual_energy=float(info['delta_energy']);finish=float(info['end'])
        step_reward=alpha*(ref_finish-finish)/ref_finish+(1-alpha)*(ref_energy-actual_energy)/ref_energy
        reference_time=graph.reference_time[ti.local_id]
        lam,boundary=fusion_weight(reference_time,graph.downward[ti.local_id])
        previous=self._paper_previous_finish.get(ti.wf_id,float(wf.arrival_time))
        realized_intrinsic=-(alpha*(reference_time-(finish-previous))/max(reference_time,1e-12)
            +(1-alpha)*(ref_energy-actual_energy)/ref_energy)
        self._paper_previous_finish[ti.wf_id]=finish
        highwater=self._paper_finish_highwater.get(ti.wf_id,float(wf.arrival_time))
        # Completion-envelope increments telescope to workflow makespan despite parallel completion order.
        increment=max(highwater,finish)-highwater;self._paper_finish_highwater[ti.wf_id]=max(highwater,finish)
        cost=(alpha*increment+(1-alpha)*float(info['delta_energy_total']))/len(wf.tasks)
        info.update(step_reward=float(step_reward),fusion_weight=lam,fusion_boundary=boundary,
            realized_intrinsic_diagnostic=float(realized_intrinsic),outer_cost=float(cost),
            reference_finish=ref_finish,reference_energy=ref_energy)
        if not np.isfinite([step_reward,lam,cost,realized_intrinsic]).all():raise ValueError('Nonfinite paper reward')
        return float(step_reward),bool(terminated or truncated),info