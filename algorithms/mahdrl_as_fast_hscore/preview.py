"""Batch fixed93 VA observations and cache the exact behavior used after HA."""
import numpy as np
import torch
from algorithms.mahdrl_as_fast_vaaware.preview import behavior_probabilities
from algorithms.mahdrl_oe_ge_apf_score_fast.packing import frozen_array, frozen_observation


def prepare_decision(env, va, host, *, training):
    states = env.preview_vm_states()
    if set(states) != {env.host_ids[i] for i in np.flatnonzero(host["mask"] > .5)}:
        raise RuntimeError("Host legality and VM previews disagree")
    version, epsilon = va._updates, va.epsilon() if training else 0.
    with env.profiler.stage("va_preview"):
        obs = torch.as_tensor(np.stack([s["obs"] for s in states.values()]), dtype=torch.float32, device=va.device)
        masks = torch.as_tensor(np.stack([s["mask"] for s in states.values()]), dtype=torch.float32, device=va.device)
        with torch.no_grad():
            values = va.online(obs, masks).detach().cpu().numpy()
    ref = float(env._obs_ref("T_ref", env.task_baseline_norm))
    if not np.isfinite(ref) or ref <= 0 or not 0 < va.gamma < 1:
        raise ValueError("Preview requires positive source time scale and gamma in (0,1)")
    reward_ref = (1-env.alpha_delay_vm)*env.energy_reward_scale*100*ref + env.alpha_delay_vm
    extra = np.zeros((env.num_hosts, 4), np.float32)
    for i, (hid, state) in enumerate(states.items()):
        q = values[i].copy()
        p = behavior_probabilities(q, state["mask"], state["performance_times"], epsilon)
        rows = state["rows"]
        extra[env.host_ids.index(hid)] = [p @ rows[:, 0]/ref, p @ rows[:, 1]/(100*ref),
            p @ np.clip(rows[:, 2]/env.task_baseline_norm, 0, 1), (1-va.gamma)*(p @ q)/reward_ref]
        state.update(obs=frozen_array(state["obs"]), mask=frozen_array(state["mask"]),
            q=q, probabilities=p, va_version=version, epsilon=epsilon)
    if not np.isfinite(extra).all():
        raise ValueError("Non-finite VA-aware host features")
    obs = dict(host["obs"])
    if env.mechanism.va_preview:
        obs["candidates"] = np.concatenate((obs["candidates"], extra), axis=1)
    return dict(host, obs=frozen_observation(obs)), states, extra
