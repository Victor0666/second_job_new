"""Detailed statistics are sampled; transition/target budgets and guards are exact."""
import numpy as np
import torch
from common.f11 import LearningDiagnostics
from common.f19_numerics import NumericalDiagnostics


class SampledNumerics(NumericalDiagnostics):
    def __init__(self, every):
        super().__init__()
        self.every, self.active = every, False

    def observe(self, name, values, scalar=False):
        if self.active:
            super().observe(name, values, scalar)

    def gradients(self, parameters, threshold=None, name="gradient"):
        grads = [p.grad.detach() for p in parameters if p.grad is not None]
        # A single device-to-host flag instead of one synchronization per parameter.
        if grads and not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all().item()):
            raise FloatingPointError("Nonfinite " + name)
        if not self.active or not self.enabled:
            return None
        norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g).double() for g in grads]))) if grads else 0.
        if not np.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient norm")
        self.observe(name, [norm], scalar=True)
        counts = self.clipping.setdefault(name, dict(observations=0, clipped=0, threshold=threshold))
        counts["observations"] += 1
        counts["clipped"] += int(threshold is not None and threshold > 0 and norm > threshold)
        return norm

    def snapshot(self):
        return dict(super().snapshot(), sample_every=self.every,
                    sample_scope="first_and_every_N_updates_or_policy_calls; validation_disabled")


class FastDiagnostics(LearningDiagnostics):
    def __init__(self, every=100):
        super().__init__()
        if every < 1:
            raise ValueError("diagnostics-every must be positive")
        self.numerics = SampledNumerics(every)
        self.pending_totals = None

    def transition(self, done):
        self.transitions += 1
        self.nonterminal += int(not done)

    def target(self, future, target):
        self.targets += target.numel()
        # Match original float32 reductions; defer only their host transfer.
        with torch.no_grad():
            totals = torch.stack(((future.abs() > 1e-12).sum().double(),
                                  future.abs().sum().double(), target.sum().double()))
            if self.pending_totals is None:
                self.pending_totals = totals
            else:
                self.pending_totals.add_(totals)

    def snapshot(self):
        if self.pending_totals is not None:
            nonzero, future, target = self.pending_totals.cpu().tolist()
            self.bootstrap_nonzero += int(nonzero)
            self.future_abs_sum += future
            self.target_sum += target
            self.pending_totals = None
        return dict(super().snapshot(), diagnostics_every=self.numerics.every,
                    count_scope="all_transitions_updates_and_sampled_targets")
