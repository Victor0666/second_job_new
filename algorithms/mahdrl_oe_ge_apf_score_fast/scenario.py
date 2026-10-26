"""Local scenario paths/parser, without registering or modifying older algorithms."""
import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from common.scenarios import Scenario, ROOT
from common.experiment_protocol import DEFAULT_PROTOCOL, protocol_hash
from common.f18_state_normalization import CURRENT_VERSION, DEFAULT_CONFIG, SHARED_VERSIONS
from common.f11 import LEARNING_MODEL
from common.ta_weights import TA_WEIGHT_MODEL
from common.workflow_opt import ENERGY_ACCOUNTING
from .config import ALGORITHM, DISPLAY_NAME, MODEL_VERSION, variant_metadata, implementation_metadata


@dataclass(frozen=True)
class ScoreScenario(Scenario):
    resource_config: str | None = None
    cache_file: str | None = None
    output_root: str = str(ROOT)
    device: str | None = None
    diagnostics_every: int = 100
    profile_mode: str = "sampled"
    profile_every: int = 100
    custom_vms: tuple = field(default=(), init=False)

    def __post_init__(self):
        if self.diagnostics_every < 1 or self.profile_every < 1 or self.profile_mode not in ("off", "sampled"):
            raise ValueError("Invalid diagnostics/profile intervals or mode")
        requested = self.resource
        if self.resource_config:
            obj = json.loads(Path(self.resource_config).read_text(encoding="utf-8"))
            values = obj.get("vms_per_host")
            if (not isinstance(values, list) or not values or
                    any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values)):
                raise ValueError("resource-config needs a nonempty positive-integer vms_per_host list")
            if self.mode == "train":
                raise ValueError("Train on S/M/L; custom topologies are evaluation/precompute targets")
            if self.mode == "eval" and not self.train_resource:
                raise ValueError("Custom topology evaluation requires --train-resource for source scales")
            object.__setattr__(self, "resource", "S")
            object.__setattr__(self, "custom_vms", tuple(values))
        super().__post_init__()
        if self.custom_vms:
            tag = "custom_" + hashlib.sha256(json.dumps(list(self.custom_vms)).encode()).hexdigest()[:12]
            if requested.startswith("custom_") and requested != tag:
                raise ValueError("Resource configuration changed since scenario creation")
            object.__setattr__(self, "resource", tag)

    @property
    def vms_per_host(self):
        return self.custom_vms or super().vms_per_host

    @property
    def cache_path(self):
        return Path(self.cache_file).resolve() if self.cache_file else super().cache_path

    def model_name(self, algorithm=ALGORITHM):
        task = self.task if self.mode == "train" else self.train_task
        resource = self.resource if self.mode == "train" else self.train_resource or self.resource
        deadline = self.deadline if self.mode == "train" else self.train_deadline or self.deadline
        protocol = "protocol_"+protocol_hash(self.experiment_protocol)[:12] if self.protocol() else "legacy"
        return (f"{MODEL_VERSION}_task{task}_res{resource}_{deadline}_model{self.model_seed}_"
                f"{LEARNING_MODEL}_{TA_WEIGHT_MODEL}_{self.normalization_mode}_{protocol}_{ENERGY_ACCOUNTING}")

    def checkpoint_dir(self, algorithm=ALGORITHM):
        if self.mode == "eval" and self.checkpoint_override:
            return Path(self.checkpoint_override).resolve()
        return Path(self.output_root)/"outputs/ckpts"/self.model_name()

    def log_dir(self, algorithm=ALGORITHM):
        return Path(self.output_root)/"outputs/logs"/self.model_name()

    def evaluation_dir(self):
        # Keep paths below Windows MAX_PATH in the user's deep workspace.
        version = MODEL_VERSION+"_"+hashlib.sha256(self.output_version.encode()).hexdigest()[:12]
        source = (f"from_task{self.train_task}_res{self.train_resource or self.resource}_"
                  f"{self.train_deadline or self.deadline}_model{self.model_seed}")
        return (Path(self.output_root)/"outputs/run"/ALGORITHM/version/source/
                f"eval_task{self.task}_res{self.resource}_{self.deadline}")

    def describe(self, algorithm=ALGORITHM):
        result = super().describe(None)
        result.update(algorithm=ALGORITHM, display_name=DISPLAY_NAME, variant=variant_metadata(), implementation=implementation_metadata(),
                      diagnostics_every=self.diagnostics_every, profile_mode=self.profile_mode, profile_every=self.profile_every,
                      checkpoint_dir=str(self.checkpoint_dir()), log_dir=str(self.log_dir()),
                      evaluation_dir=str(self.evaluation_dir()), device=self.device or "auto")
        return result


def parse_scenario(mode, argv=None):
    p = argparse.ArgumentParser(description=DISPLAY_NAME+": "+mode)
    p.add_argument("--task", choices="SML", default="S")
    p.add_argument("--resource", choices="SML", default="S")
    p.add_argument("--deadline", choices=("Tight", "Medium", "Loose"), default="Tight")
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seeds", default="0-1000" if mode == "precompute" else "700-729")
    p.add_argument("--train-task", choices="SML", default="S")
    p.add_argument("--train-resource", choices="SML")
    p.add_argument("--train-deadline", choices=("Tight", "Medium", "Loose"))
    p.add_argument("--checkpoint-dir")
    p.add_argument("--resource-config", help="JSON containing vms_per_host, for evaluation/precompute")
    p.add_argument("--deadline-cache", help="Explicit matching cache; checked before execution")
    p.add_argument("--out", help="Precompute output; never overwrites")
    p.add_argument("--experiment-protocol", default=str(DEFAULT_PROTOCOL))
    p.add_argument("--legacy-split", action="store_true")
    p.add_argument("--model-seed", type=int, default=1)
    p.add_argument("--normalization-mode", choices=(*SHARED_VERSIONS, "legacy"), default=CURRENT_VERSION)
    p.add_argument("--normalization-config", default=str(DEFAULT_CONFIG))
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--diagnostics-every", type=int, default=100)
    p.add_argument("--profile-mode", choices=("off", "sampled"), default="sampled")
    p.add_argument("--profile-every", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.out and mode != "precompute":
        p.error("--out is reserved for deadline precompute")
    try:
        scene = ScoreScenario(task=a.task, resource=a.resource, deadline=a.deadline, mode=mode,
            episodes=a.episodes, seed_spec=a.seeds, train_task=a.train_task,
            train_resource=a.train_resource, train_deadline=a.train_deadline,
            checkpoint_override=a.checkpoint_dir, cache_output=a.out,
            resource_config=a.resource_config, cache_file=a.deadline_cache,
            experiment_protocol=a.experiment_protocol, legacy_split=a.legacy_split,
            model_seed=a.model_seed, normalization_mode=a.normalization_mode,
            normalization_config=a.normalization_config, device=a.device,
            diagnostics_every=a.diagnostics_every, profile_mode=a.profile_mode, profile_every=a.profile_every)
    except (ValueError, OSError, TypeError) as exc:
        p.error(str(exc))
    if a.dry_run:
        print(json.dumps(scene.describe(), ensure_ascii=False, indent=2))
        raise SystemExit(0)
    return scene
