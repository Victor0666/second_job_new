"""Random-VA expectations without constructing or forwarding a VA network."""
import numpy as np
from algorithms.mahdrl_oe_ge_apf_score_fast.packing import frozen_array, frozen_observation


def uniform_action(mask, rng):
    legal = np.flatnonzero(np.asarray(mask) > .5)
    if not len(legal):
        raise ValueError("Random policy has no legal action")
    return int(rng.choice(legal))


def prepare_random_decision(env, host):
    states = env.preview_vm_states()
    if set(states) != {env.host_ids[i] for i in np.flatnonzero(host["mask"] > .5)}:
        raise RuntimeError("Random VA preview and legal hosts disagree")
    ref = float(env._obs_ref("T_ref", env.task_baseline_norm))
    extra = np.zeros((env.num_hosts, 4), np.float32)
    for hid, state in states.items():
        legal = state["mask"] > .5
        p = legal.astype(float)/legal.sum()
        rows = state["rows"]
        extra[env.host_ids.index(hid)] = [p @ rows[:, 0]/ref, p @ rows[:, 1]/(100*ref),
            p @ np.clip(rows[:, 2]/env.task_baseline_norm, 0, 1), 0.]
        state.update(obs=frozen_array(state["obs"]), mask=frozen_array(state["mask"]),
            probabilities=p, q=np.zeros(9), va_version=-1, epsilon=1.)
    obs = dict(host["obs"])
    if env.mechanism.va_preview:
        obs["candidates"] = np.concatenate((obs["candidates"], extra), axis=1)
    return dict(host, obs=frozen_observation(obs)), states, extra
