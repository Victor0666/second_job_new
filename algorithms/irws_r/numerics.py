"""The existing diagnostic fields, with bounded CUDA statistics copied at snapshot."""
import numpy as np
import torch
from common.f19_numerics import NumericalDiagnostics as BaseDiagnostics


class NumericalDiagnostics(BaseDiagnostics):
    def reset(self):
        super().reset()
        self._pending={}

    def observe(self,name,values,scalar=False):
        if not self.enabled:return
        if not torch.is_tensor(values) or values.device.type!='cuda':
            return super().observe(name,values,scalar)
        values=values.detach().double()
        if not values.numel():return
        rows=values.reshape(-1,1) if scalar else values.reshape(-1,values.shape[-1])
        finite=torch.isfinite(rows)
        counts=(~finite).sum(0)
        maxima=torch.where(finite,rows.abs(),0.).max(0).values
        samples=torch.where(finite,rows,torch.nan)[-self.capacity:].clone()
        if name in self._pending:
            old=self._pending[name]
            if rows.shape[1]!=len(old['max_abs']):raise ValueError('Diagnostic field count changed: '+name)
            counts=counts+old['nonfinite']
            maxima=torch.maximum(maxima,old['max_abs'])
            samples=torch.cat((old['sample'],samples))[-self.capacity:]
            count=old['count']+len(rows)
        else:count=len(rows)
        self._pending[name]=dict(count=count,nonfinite=counts,max_abs=maxima,sample=samples)

    def gradients(self,parameters,threshold=None,name='gradient'):
        parameters=list(parameters)
        grads=[p.grad.detach() for p in parameters if p.grad is not None]
        if not self.enabled:return None
        if not grads or grads[0].device.type!='cuda':
            return super().gradients(parameters,threshold,name)
        norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g).double() for g in grads]))
        if not torch.stack([torch.isfinite(g).all() for g in grads]+[torch.isfinite(norm)]).all():
            raise FloatingPointError('Nonfinite '+name)
        self.observe(name,norm,scalar=True)
        record=self.clipping.setdefault(name,dict(observations=0,clipped=0,threshold=threshold))
        record['observations']+=1
        record['clipped']+=int(threshold is not None and threshold>0 and norm>threshold)
        return norm

    def snapshot(self):
        if self._pending:
            # One bulk copy, preserving whole-episode counts and latest-row quantiles.
            packed=torch.cat([torch.cat((p['nonfinite'].double(),p['max_abs'],p['sample'].flatten()))
                              for p in self._pending.values()]).cpu().numpy()
            offset=0
            for name,p in self._pending.items():
                width=len(p['max_abs']);size=p['sample'].numel()
                self.profiles[name]=dict(count=p['count'],nonfinite=packed[offset:offset+width].astype(np.int64),
                    max_abs=packed[offset+width:offset+2*width].copy(),
                    sample=packed[offset+2*width:offset+2*width+size].reshape(-1,width).copy())
                offset+=2*width+size
        return super().snapshot()
