"""Topology-dependent Eq.(14)-(19), separate Table III actor/critic, history RNN."""
import networkx as nx
import torch
from torch import nn


def masked_distribution(logits,mask):
    mask=torch.as_tensor(mask,dtype=torch.bool,device=logits.device)
    if logits.shape!=mask.shape or not mask.any() or not torch.isfinite(logits).all():
        raise ValueError('Invalid policy logits or empty legal action set')
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask,-torch.inf))


def mlp(input_dim,hidden,output_dim):
    layers=[]
    for width in hidden:layers.extend((nn.Linear(input_dim,width),nn.Tanh()));input_dim=width
    return nn.Sequential(*layers,nn.Linear(input_dim,output_dim))


class TopologicalLayer(nn.Module):
    def __init__(self,dim):
        super().__init__();self.self_weight=nn.Linear(dim,dim,bias=False);self.neighbor_weight=nn.Linear(dim,dim,bias=False)
        self.forward_gru=nn.GRUCell(dim,dim);self.reverse_gru=nn.GRUCell(dim,dim)

    def forward(self,previous,neighbors,order):
        output=[None]*len(neighbors)
        for v in order:
            # Eq.(16) leaves the reduction implicit: featurewise neighbor attention,
            # including the previous self-state as described in the adjoining text.
            states=torch.stack([previous[v]]+[output[u] for u in neighbors[v]])
            attention=torch.softmax(self.self_weight(previous[v])+self.neighbor_weight(states),dim=0)
            aggregate=(attention*states).sum(0)
            # Two-token Bi-GRU of (past self, aggregated neighbors), Eq.(17).
            zero=torch.zeros_like(previous[v])
            forward=self.forward_gru(aggregate,self.forward_gru(previous[v],zero))
            reverse=self.reverse_gru(previous[v],self.reverse_gru(aggregate,zero))
            output[v]=(forward+reverse)/2
        return torch.stack(output)


class WorkflowEmbedding(nn.Module):
    def __init__(self,dim=5,passes=1):
        super().__init__();self.initial=nn.Linear(2,dim)
        self.forward_layers=nn.ModuleList(TopologicalLayer(dim) for _ in range(passes))
        self.reverse_layers=nn.ModuleList(TopologicalLayer(dim) for _ in range(passes));self.pool=nn.Linear(dim,dim)

    def forward(self,graph):
        device=self.initial.weight.device
        x=torch.tensor(graph.features,dtype=self.initial.weight.dtype,device=device)
        dag=nx.DiGraph();dag.add_nodes_from(range(len(graph.gids)))
        dag.add_edges_from((u,v) for v,parents in enumerate(graph.parents) for u in parents)
        order=list(nx.topological_sort(dag));children=tuple(tuple(dag.successors(i)) for i in range(len(graph.gids)))
        forward=reverse=torch.tanh(self.initial(x))
        for fl,rl in zip(self.forward_layers,self.reverse_layers):
            forward=fl(forward,graph.parents,order)
            reverse=rl(reverse,children,list(reversed(order)))
        # Eq.(18) is indexed by v: pool direction representations of each task,
        # not all tasks into one identical selection score (Fig.3).
        return self.pool(torch.stack((forward,reverse)).mean(0))


class TaskPolicy(nn.Module):
    def __init__(self,config):
        super().__init__();self.embedding=WorkflowEmbedding(config.embedding_dim,config.message_passes)
        self.score=mlp(config.embedding_dim+3,(config.embedding_dim,),1)

    def embeddings(self,graphs):
        result={}
        for graph in graphs:
            tensor=self.embedding(graph)
            result.update({gid:tensor[i] for i,gid in enumerate(graph.gids)})
        return result

    def distribution(self,embeddings,candidates,context):
        if not candidates or len(set(candidates))!=len(candidates):raise ValueError('Ready candidates must be unique and nonempty')
        matrix=torch.stack([embeddings[g] for g in candidates])
        context=torch.as_tensor(context,dtype=matrix.dtype,device=matrix.device)
        if context.shape!=(len(candidates),3) or not torch.isfinite(context).all():
            raise ValueError("Invalid current task deadline context")
        return torch.distributions.Categorical(logits=self.score(torch.cat((matrix,context),dim=-1)).squeeze(-1))


class HistoryReward(nn.Module):
    def __init__(self,obs_dim,actions,hidden):
        super().__init__();self.obs_dim=obs_dim;self.actions=actions
        # (s, VM action, external reward, execution duration, energy, done, s_next)
        self.cell=nn.GRUCell(2*obs_dim+actions+4,hidden);self.head=nn.Linear(hidden,1)

    def forward(self,token,hidden):
        hidden=self.cell(token,hidden)
        return self.head(hidden).squeeze(-1),hidden