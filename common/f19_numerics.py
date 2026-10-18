"""Read-only bounded numerical diagnostics; no RNG, gradients or policy mutation."""
import os
import numpy as np
import torch


class NumericalDiagnostics:
    def __init__(self, capacity=256):
        if not isinstance(capacity,int) or capacity < 1:
            raise ValueError("Diagnostic capacity must be a positive integer")
        self.enabled = os.environ.get('F19_DIAGNOSTICS', os.environ.get('F11_DIAGNOSTICS','1')) != '0'
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.profiles = {}
        self.clipping = {}

    def observe(self, name, values, scalar=False):
        if not self.enabled: return
        if torch.is_tensor(values): values = values.detach().cpu().numpy()
        array = np.asarray(values, dtype=np.float64)
        if not array.size: return
        rows = array.reshape(-1, 1) if scalar else array.reshape(-1, array.shape[-1])
        key = name
        if key not in self.profiles:
            self.profiles[key] = dict(count=0, nonfinite=np.zeros(rows.shape[1],dtype=np.int64),
                                      max_abs=np.zeros(rows.shape[1]), sample=np.empty((0,rows.shape[1])))
        profile = self.profiles[key]
        if rows.shape[1] != len(profile['max_abs']):
            raise ValueError('Diagnostic field count changed: '+name)
        finite = np.isfinite(rows)
        profile['count'] += len(rows)
        profile['nonfinite'] += (~finite).sum(axis=0)
        profile['max_abs'] = np.maximum(profile['max_abs'], np.where(finite,np.abs(rows),0).max(axis=0))
        # shortcut: quantiles describe the latest 256 rows, not whole-run exact quantiles.
        profile['sample'] = np.concatenate((profile['sample'],np.where(finite,rows,np.nan)[-self.capacity:]))[-self.capacity:]

    def inputs(self, module, name='input'):
        def capture(_module, args):
            if args: self.observe(name,args[0])
            # A forward pre-hook must return None to preserve the original arguments.
        module.register_forward_pre_hook(capture)

    def gradients(self, parameters, threshold=None, name='gradient'):
        if not self.enabled: return None
        parameters = list(parameters)
        grads = [p.grad.detach() for p in parameters if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('Nonfinite '+name)
        norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g).double() for g in grads]))) if grads else 0.
        if not np.isfinite(norm): raise FloatingPointError('Nonfinite '+name+' norm')
        self.observe(name,[norm],scalar=True)
        counts = self.clipping.setdefault(name,dict(observations=0,clipped=0,threshold=threshold))
        counts['observations'] += 1
        counts['clipped'] += int(threshold is not None and threshold > 0 and norm > threshold)
        return norm

    def snapshot(self):
        if not self.enabled: return dict(enabled=False)
        profiles = {}
        for name, profile in self.profiles.items():
            quantiles=[]
            for column in profile['sample'].T:
                finite=column[np.isfinite(column)]
                quantiles.append(np.percentile(finite,[1,50,95,99]).tolist() if finite.size else [None]*4)
            columns=list(map(list,zip(*quantiles)))
            profiles[name]=dict(rows_seen=profile['count'],window_rows=len(profile['sample']),
                field_indices=list(range(len(quantiles))),nonfinite=profile['nonfinite'].tolist(),
                max_abs=profile['max_abs'].tolist(),**dict(zip(('P1','P50','P95','P99'),columns)))
        clipping={name:dict(**counts,clipped_ratio=counts['clipped']/max(1,counts['observations'])) for name,counts in self.clipping.items()}
        return dict(enabled=True,quantile_scope='latest_rows',capacity=self.capacity,profiles=profiles,clipping=clipping)
