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
    """Bounded dynamic intrinsic weight; never negates or removes external reward."""
    if not np.isfinite([reference_time,downward_rank]).all() or min(reference_time,downward_rank)<0:
        raise ValueError('Invalid physical time/rank for intrinsic weight')
    scale=max(reference_time,downward_rank)
    weight=0. if scale==0 else (reference_time/scale)/(reference_time/scale+downward_rank/scale)
    return float(weight), bool(downward_rank==0)


class PaperCloudEnv(IRWSEnvRouteA):
    def reset(self,**kwargs):
        self._paper_graphs={}
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

    def task_context(self,gids):
        """Current task/workflow deadline remaining time and Ready wait, on shared T_ref."""
        rows=[]
        for gid in gids:
            ti=self.task_meta[gid];wf=self.workflows[ti.wf_id];now=float(self.current_time)
            # Unconstrained test environments have horizon placeholders, not real deadlines.
            task_remaining=workflow_remaining=0.
            if self.deadline_mode!='none':
                task_remaining=float(self.task_baseline_finish[gid])-now
                workflow_remaining=float(wf.deadline)-now
            wait=max(0.,now-float(self.task_ready_time[gid]))
            rows.append([self.scale(v,'T_ref') for v in (task_remaining,workflow_remaining,wait)])
        return np.asarray(rows,dtype=np.float64).reshape(-1,3)

    def resource_features(self,gid):
        task=self._task_obj_from_gid(gid);parents=self._build_parents_loc(gid);now=float(self.current_time)
        result=[self.scale(task.workload_mi,'workload_mi_ref'),*self.task_context([gid])[0]]
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
        if self.done_flag:return dict(obs=np.zeros(4+6*self.num_vms,dtype=np.float32),mask=np.zeros(self.num_vms,dtype=np.float32))
        return dict(obs=self.resource_features(self.current_task_gid),
                    mask=(self.vm_available_at<=self.current_time+1e-12).astype(np.float32))

    def paper_step(self,action):
        if self.current_task_gid is None:raise RuntimeError('Select a Ready task first')
        if not 0<=action<self.num_vms or self.vm_available_at[action]>self.current_time+1e-12:
            raise ValueError('Illegal VM action; no physical mutation')
        gid=int(self.current_task_gid);ti=self.task_meta[gid]
        graph=next(g for g in self.graphs() if g.wf_id==ti.wf_id)
        # The public simulator already settles global energy (including tail idle) and tardiness.
        _,external,terminated,truncated,info=super().step(action)
        weight,boundary=fusion_weight(graph.reference_time[ti.local_id],graph.downward[ti.local_id])
        info.update(step_reward=float(external),outer_cost=-float(external),
                    fusion_weight=weight,fusion_boundary=boundary,reward_protocol=REWARD_PROTOCOL)
        if not np.isfinite([external,weight]).all():raise ValueError('Nonfinite adapted IRWS reward')
        return float(external),bool(terminated or truncated),info
