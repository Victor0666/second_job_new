"""Analytic epsilon/GE expectations; previewing never samples an action."""
import numpy as np
import torch
from algorithms.mahdrl_oe_ge_apf_score_fast.packing import collate, frozen_observation
from .config import APF


def behavior_probabilities(q, mask, times, epsilon):
    q, mask, times = (np.asarray(x) for x in (q, mask, times))
    if q.ndim != 1 or mask.shape != q.shape or times.shape != q.shape:
        raise ValueError("Invalid preview action arrays")
    legal = np.flatnonzero(mask > .5)
    if (not len(legal) or not np.isin(mask, (0, 1)).all()
            or not np.isfinite(q).all() or not np.isfinite(times[legal]).all()
            or np.any(times[legal] < 0) or not np.isfinite(epsilon) or not 0 <= epsilon <= 1):
        raise ValueError("Invalid preview probabilities or legal candidates")
    greedy = legal[np.argmax(q[legal])]
    fastest = legal[np.isclose(times[legal], times[legal].min(), rtol=1e-9, atol=1e-9)]
    p = np.zeros(len(q), np.float64)
    p[legal] = epsilon * (1-APF.guided_probability) / len(legal)
    p[fastest] += epsilon * APF.guided_probability / len(fastest)
    p[greedy] += 1-epsilon
    return p


def prepare_decision(env, va, host, *, training):
    states = env.preview_vm_states()
    if set(states) != {env.host_ids[i] for i in np.flatnonzero(host["mask"] > .5)}:
        raise RuntimeError("Host legality and VM previews disagree")
    version, epsilon = va._updates, va.epsilon() if training else 0.
    with env.profiler.stage("va_preview"):
        batch = collate([s["obs"] for s in states.values()], [s["mask"] for s in states.values()], va.device)
        with torch.no_grad():
            values = va.online(*batch).detach().cpu().numpy()
    ref = float(env._obs_ref("T_ref", env.task_baseline_norm))
    if not np.isfinite(ref) or ref <= 0 or not 0 < va.gamma < 1:
        raise ValueError("Preview requires positive source time scale and gamma in (0,1)")
    reward_ref = (1-env.alpha_delay_vm)*env.energy_reward_scale*100*ref + env.alpha_delay_vm
    extra = np.zeros((env.num_hosts, 4), np.float32)
    for i, (hid, state) in enumerate(states.items()):
        q = values[i, :len(state["mask"])].copy()
        p = behavior_probabilities(q, state["mask"], state["performance_times"], epsilon)
        rows = state["rows"]
        features = [p @ rows[:, 0]/ref, p @ rows[:, 1]/(100*ref),
            p @ np.clip(rows[:, 2]/env.task_baseline_norm, 0, 1),
            (1-va.gamma)*(p @ q)/reward_ref]
        extra[env.host_ids.index(hid)] = features
        state.update(q=q, probabilities=p, va_version=version, epsilon=epsilon)
    if not np.isfinite(extra).all():
        raise ValueError("Non-finite VA-aware host features")
    obs = dict(host["obs"])
    if env.mechanism.va_preview:
        obs["candidates"] = np.concatenate((obs["candidates"], extra), axis=1)
    decision = dict(host, obs=frozen_observation(obs))
    return decision, states, extra
