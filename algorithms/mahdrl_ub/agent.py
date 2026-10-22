"""TA-only future-value projection; baseline action selection and HA/VA stay intact."""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.d3qn_agent_EXDehid import D3QNAgent, _polyak_update
from common.f11 import LearningDiagnostics


TA_TARGET_RULE = dict(scheme="mahdrl_ub_ta_future_upper_v1", role="TA", q_upper_bound=0.0)


class _ProjectionDiagnostics(LearningDiagnostics):
    def __init__(self, numerics):
        super().__init__()
        # Preserve the input hooks already registered by the baseline constructor.
        self.numerics = numerics
        self.eligible_samples = self.projected_samples = 0
        self.last_batch_eligible = self.last_batch_projected = 0

    def projection(self, q_raw, raw_future, future, raw_target, eligible, projected):
        self.numerics.observe("Q_next_raw", q_raw[eligible], scalar=True)
        self.numerics.observe("Bootstrap_raw", raw_future, scalar=True)
        self.numerics.observe("Bootstrap_projected", future, scalar=True)
        self.numerics.observe("TD_target_raw", raw_target, scalar=True)
        if self.enabled:
            self.last_batch_eligible = int(eligible.sum().item())
            self.last_batch_projected = int(projected.sum().item())
            self.eligible_samples += self.last_batch_eligible
            self.projected_samples += self.last_batch_projected

    def snapshot(self):
        result = super().snapshot()
        result["ta_future_upper_bound"] = dict(
            **TA_TARGET_RULE,
            eligible_samples=self.eligible_samples,
            projected_samples=self.projected_samples,
            cumulative_projection_ratio=self.projected_samples / max(self.eligible_samples, 1),
            last_batch_eligible=self.last_batch_eligible,
            last_batch_projected=self.last_batch_projected,
            last_batch_projection_ratio=self.last_batch_projected / max(self.last_batch_eligible, 1),
        )
        return result


class TAUpperBoundAgent(D3QNAgent):
    """Use min(Q_target(s', a*), 0) only in the TA bootstrap, never in argmax."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f11 = _ProjectionDiagnostics(self.f11.numerics)

    def remember(self, state, mask, action, reward, next_state, next_mask, done):
        if reward > 0:
            raise ValueError("MAHDRL-UB TA requires nonpositive rewards for its zero future-value upper bound")
        return super().remember(state, mask, action, reward, next_state, next_mask, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)
        s, m, a, r, s2, m2, d = map(np.array, zip(*batch))
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        m = torch.tensor(m, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        m2 = torch.tensor(m2, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

        q_all = self.online(s)
        q_sa = q_all.gather(1, a.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            q2_online = self.online(s2).masked_fill(m2 <= 0.5, -1e9)
            a2 = q2_online.argmax(dim=1, keepdim=True)
            q2_target = self.target(s2).masked_fill(m2 <= 0.5, -1e9)
            q2_raw = q2_target.gather(1, a2).squeeze(1)
            has_valid = (m2 > 0.5).any(dim=1)
            q2_raw = torch.where(has_valid, q2_raw, torch.zeros_like(q2_raw))

            raw_future = (1.0 - d) * self.gamma * q2_raw
            # Current TA rewards are nonpositive, hence every true future return is <= 0.
            future = (1.0 - d) * self.gamma * q2_raw.clamp(max=0.0)
            target = r + future
            eligible = (d < 0.5) & has_valid
            self.f11.projection(q2_raw, raw_future, future, r + raw_future,
                                eligible, eligible & (q2_raw > 0.0))

        self.f11.target(future, target)
        loss = F.smooth_l1_loss(q_sa, target)
        self.f11.numerics.observe("Q_selected", q_sa, scalar=True)
        self.f11.numerics.observe("TD_target", target, scalar=True)
        self.f11.numerics.observe("TD_error", target - q_sa, scalar=True)
        loss_value = self.f11.loss(loss)
        self.optim.zero_grad()
        loss.backward()
        self.f11.numerics.gradients(self.online.parameters(), self.grad_clip)
        if self.grad_clip is not None and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optim.step()

        self._updates += 1
        self.f11.updates += 1
        self._eps_steps += 1
        if self.target_update_tau > 0.0:
            _polyak_update(self.target, self.online, self.target_update_tau)
        elif self._updates % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss_value

    def save(self, path):
        self.normalization_metadata = dict(
            getattr(self, "normalization_metadata", {"normalization_version": "legacy"}),
            ta_future_value_target=dict(TA_TARGET_RULE),
        )
        super().save(path)

    def load(self, path, strict=True, require_learning_model=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        metadata = checkpoint.get("normalization") if isinstance(checkpoint, dict) else None
        if not isinstance(metadata, dict) or metadata.get("ta_future_value_target") != TA_TARGET_RULE:
            raise ValueError("Checkpoint MAHDRL-UB TA target-rule mismatch; train MAHDRL-UB or select its matching checkpoint directory")
        del checkpoint
        super().load(path, strict=strict, require_learning_model=require_learning_model)
