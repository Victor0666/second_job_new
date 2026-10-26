"""Topology-dependent Eq.(14)-(19), separate Table III actor/critic, history RNN."""
import networkx as nx
import torch
from torch import nn


def masked_distribution(logits,mask):
    mask=torch.as_tensor(mask,dtype=torch.bool,device=logits.device)
    if logits.shape!=mask.shape or not mask.any(dim=-1).all() or not torch.isfinite(logits).all():
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

    @staticmethod
    def plan(neighbors, order, device):
        levels={}
        groups={}
        for v in order:
            depth=max((levels[u]+1 for u in neighbors[v]),default=0)
            levels[v]=depth
            groups.setdefault(depth,[]).append(v)
        result=[]
        sentinel=len(neighbors)
        for nodes in groups.values():
            width=max(1,max(len(neighbors[v]) for v in nodes))
            parents=[list(neighbors[v])+[sentinel]*(width-len(neighbors[v])) for v in nodes]
            result.append((torch.tensor(nodes,device=device),torch.tensor(parents,device=device),
                           torch.tensor([max(1,len(neighbors[v])) for v in nodes],device=device).unsqueeze(-1)))
        return tuple(result)

    def forward(self,previous,neighbors=None,order=None,plan=None):
        if plan is None:
            plan=self.plan(neighbors,order,previous.device)
        # A zero sentinel handles empty/padded predecessor lists without self edges.
        output=previous.new_zeros((len(previous)+1,previous.shape[-1]))
        for nodes,parents,counts in plan:
            mean=output[parents].sum(1)/counts
            old=previous[nodes]
            aggregate=torch.softmax(self.self_weight(old)+self.neighbor_weight(mean),dim=-1)
            zero=torch.zeros_like(old)
            forward=self.forward_gru(aggregate,self.forward_gru(old,zero))
            reverse=self.reverse_gru(old,self.reverse_gru(aggregate,zero))
            output=output.index_copy(0,nodes,(forward+reverse)/2)
        return output[:-1]


class WorkflowEmbedding(nn.Module):
    def __init__(self,dim=5,passes=1):
        super().__init__();self.initial=nn.Linear(2,dim)
        self.forward_layers=nn.ModuleList(TopologicalLayer(dim) for _ in range(passes))
        self.reverse_layers=nn.ModuleList(TopologicalLayer(dim) for _ in range(passes));self.pool=nn.Linear(dim,dim)
        self.clear_cache()

    def clear_cache(self):
        self._graphs={}
        self._bundles={}

    def _apply(self,fn):
        result=super()._apply(fn)
        self.clear_cache()
        return result

    def _prepare_graph(self,graph):
        key=id(graph)
        if key not in self._graphs:
            dag=nx.DiGraph()
            dag.add_nodes_from(range(len(graph.gids)))
            dag.add_edges_from((u,v) for v,parents in enumerate(graph.parents) for u in parents)
            order=tuple(nx.topological_sort(dag))
            children=tuple(tuple(dag.successors(i)) for i in range(len(graph.gids)))
            features=torch.tensor(graph.features,dtype=self.initial.weight.dtype,device=self.initial.weight.device)
            self._graphs[key]=(graph,features,order,children)
        return self._graphs[key]

    def _prepare(self,graphs):
        key=tuple(id(g) for g in graphs)
        if key not in self._bundles:
            features=[];parents=[];children=[];order=[];reverse_order=[];sizes=[]
            offset=0
            for graph in graphs:
                _,x,topology,successors=self._prepare_graph(graph)
                features.append(x);sizes.append(len(graph.gids))
                parents.extend(tuple(u+offset for u in ps) for ps in graph.parents)
                children.extend(tuple(u+offset for u in cs) for cs in successors)
                order.extend(v+offset for v in topology)
                reverse_order.extend(v+offset for v in reversed(topology))
                offset+=len(graph.gids)
            device=self.initial.weight.device
            self._bundles[key]=(torch.cat(features),TopologicalLayer.plan(parents,order,device),
                                TopologicalLayer.plan(children,reverse_order,device),sizes)
        return self._bundles[key]

    def forward(self,graph):
        return self.forward_many((graph,))[0]

    def forward_many(self,graphs):
        if not graphs:return ()
        x,forward_plan,reverse_plan,sizes=self._prepare(graphs)
        forward=reverse=torch.tanh(self.initial(x))
        for fl,rl in zip(self.forward_layers,self.reverse_layers):
            forward=fl(forward,plan=forward_plan)
            reverse=rl(reverse,plan=reverse_plan)
        # Eq.(18) is indexed by v: pool direction representations of each task,
        # not all tasks into one identical selection score (Fig.3).
        return self.pool(torch.stack((forward,reverse)).mean(0)).split(sizes)


class TaskPolicy(nn.Module):
    def __init__(self,config):
        super().__init__();self.embedding=WorkflowEmbedding(config.embedding_dim,config.message_passes)
        self.score=mlp(config.embedding_dim,(config.embedding_dim,),1)
        # Eq.(19) gives no independent Q-head update; do not invent a joint PPO loss.
        self.score.requires_grad_(False)

    def embeddings(self,graphs):
        result={}
        for graph,tensor in zip(graphs,self.embedding.forward_many(graphs)):
            result.update({gid:tensor[i] for i,gid in enumerate(graph.gids)})
        return result

    def distribution(self,embeddings,candidates):
        if not candidates or len(set(candidates))!=len(candidates):raise ValueError('Ready candidates must be unique and nonempty')
        matrix=torch.stack([embeddings[g] for g in candidates])
        return torch.distributions.Categorical(logits=self.score(matrix).squeeze(-1))


class HistoryReward(nn.Module):
    def __init__(self,obs_dim,actions,hidden):
        super().__init__();self.obs_dim=obs_dim;self.actions=actions
        # Paper trajectory token: (s, VM action, external reward, done, s_next).
        self.cell=nn.GRUCell(2*obs_dim+actions+2,hidden);self.head=nn.Linear(hidden,1)

    def forward(self,token,hidden):
        hidden=self.cell(token,hidden)
        return self.head(hidden).squeeze(-1),hidden
