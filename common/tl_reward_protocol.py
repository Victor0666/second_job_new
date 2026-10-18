"""Prevent loading a same-shaped TL model as a TL_2 model."""
import json
from pathlib import Path


def write_reward_protocol(directory, scope):
    (Path(directory) / "reward_protocol.json").write_text(
        json.dumps({"vm_energy_scope": scope}, indent=2), encoding="utf-8")


def check_reward_protocol(directory, scope):
    path = Path(directory) / "reward_protocol.json"
    if not path.exists() and scope == "selected_host":
        return  # Existing original TL checkpoints predate the scope manifest.
    if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("vm_energy_scope") != scope:
        raise ValueError(f"Checkpoint VA energy scope must be {scope}: {path}")


def deferred_vm_phase_mean(env, info, last_assignment_count):
    """Attach deferred TL_2 energy to the most recent assignment phase's mean."""
    if (env.vm_energy_scope == "global" and last_assignment_count > 0
            and not info.get("lower_energy_already_emitted", True)):
        return ((1 - env.alpha_delay_vm) * float(info.get("r_energy_phase", 0.0))
                / last_assignment_count)
    return 0.0
