"""Local scenario paths/parser, without registering or modifying older algorithms."""
import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from common.scenarios import Scenario, ROOT, add_workload_argument
from common.experiment_protocol import DEFAULT_PROTOCOL
from common.f18_state_normalization import CURRENT_VERSION, DEFAULT_CONFIG, SHARED_VERSIONS
from common.r010_paths import training_name
from .config import ALGORITHM, DISPLAY_NAME, MODEL_VERSION, OUTPUT_TAG, variant_metadata, implementation_metadata
from .selection import SELECTION_TAG, selection_metadata


@dataclass(frozen=True)
class ScoreScenario(Scenario):
    algorithm = ALGORITHM
    display_name = DISPLAY_NAME
    model_version = MODEL_VERSION
    output_tag = OUTPUT_TAG

    def variant_metadata(self):
        return variant_metadata()

    def implementation_metadata(self):
        return implementation_metadata()

    def deadline_reference_scenario(self):
        return self

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

    def model_name(self, algorithm=None):
        original = training_name(self, self.model_version,
            dict(protocol=self.selection_protocol(), variant=self.variant_metadata(),
                 implementation=self.implementation_metadata()))
        return self.source_tag + original[original.index('_m'):]

    @staticmethod
    def scene_tag(task, resource, deadline):
        return task + resource.replace('custom_', 'C') + '_' + {
            'Tight': 'T', 'Medium': 'M', 'Loose': 'L'}[deadline]

    @property
    def source_tag(self):
        if self.mode != 'eval':
            return self.scene_tag(self.task, self.resource, self.deadline)
        return self.scene_tag(self.train_task, self.train_resource or self.resource,
                              self.train_deadline or self.deadline)

    @property
    def target_tag(self):
        return self.scene_tag(self.task, self.resource, self.deadline)

    def checkpoint_dir(self, algorithm=None):
        if self.mode == "eval" and self.checkpoint_override:
            return Path(self.checkpoint_override).resolve()
        return Path(self.output_root)/"outputs/ckpts"/self.output_tag/self.model_name()

    def log_dir(self, algorithm=None):
        return Path(self.output_root)/"outputs/logs"/self.output_tag/self.model_name()

    @property
    def evaluation_root(self):
        return Path(self.output_root)/"outputs/run"/self.output_tag

    def evaluation_csv(self):
        seeds = sorted(self.seeds)
        seed_tag = str(seeds[0]) if len(seeds) == 1 else f'{seeds[0]}-{seeds[-1]}'
        if seeds != list(range(seeds[0], seeds[-1]+1)):
            seed_tag += '_' + hashlib.sha256(json.dumps(seeds).encode()).hexdigest()[:8]
        config_tag = self.model_name().rsplit('_', 1)[-1]
        return self.evaluation_root / (
            f'eval_{self.output_tag}_noapf_sr092_{self.target_tag}_from_{self.source_tag}_'
            f'm{self.model_seed}_s{seed_tag}_{config_tag}{self.workload_suffix}.csv')

    def evaluation_identity(self):
        return dict(target_scene=self.target_tag, source_scene=self.source_tag,
            task=self.task, resource=self.resource, deadline=self.deadline,
            train_task=self.train_task, train_resource=self.train_resource or self.resource,
            train_deadline=self.train_deadline or self.deadline)

    def evaluation_dir(self):
        key = hashlib.sha256(self.evaluation_csv().name.encode()).hexdigest()[:16]
        return self.evaluation_root / '_details' / key

    @property
    def output_version(self):
        return self.model_version + "_" + super().output_version + "_" + SELECTION_TAG

    def selection_protocol(self):
        protocol = self.protocol()
        metadata = protocol.metadata() if protocol else dict(version="legacy", validation=self.validation_seeds())
        metadata.update(selection=SELECTION_TAG, checkpoint_selection=selection_metadata(), tl_policy=self.variant_metadata()["tl_policy"])
        return self.with_workload_metadata(metadata)

    def reproducibility_metadata(self):
        metadata = super().reproducibility_metadata()
        metadata.update(algorithm=self.algorithm, model_version=self.model_version,
                        data_split=self.selection_protocol(), checkpoint_selection=selection_metadata())
        return metadata

    def normalization_metadata(self):
        metadata = super().normalization_metadata()
        metadata.update(model_version=self.model_version, experiment_protocol=self.selection_protocol(),
                        checkpoint_selection=selection_metadata())
        return metadata

    def describe(self, algorithm=None):
        result = super().describe(None)
        result.update(algorithm=self.algorithm, display_name=self.display_name, variant=self.variant_metadata(), implementation=self.implementation_metadata(),
                      diagnostics_every=self.diagnostics_every, profile_mode=self.profile_mode, profile_every=self.profile_every,
                      model_version=self.model_version, checkpoint_selection=selection_metadata(),
                      experiment_protocol=self.selection_protocol(),
                      checkpoint_dir=str(self.checkpoint_dir()), log_dir=str(self.log_dir()),
                      evaluation_dir=str(self.evaluation_dir()), evaluation_csv=str(self.evaluation_csv()),
                      device=self.device or "auto")
        return result


def parse_scenario(mode, argv=None, *, scenario_class=ScoreScenario, resource_choices="SML", resource_default="S"):
    p = argparse.ArgumentParser(description=scenario_class.display_name+": "+mode)
    add_workload_argument(p)
    p.add_argument("--task", choices="SML", default="S")
    p.add_argument("--resource", choices=resource_choices, default=resource_default)
    p.add_argument("--deadline", choices=("Tight", "Medium", "Loose"), default="Tight")
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seeds", default="0-1000" if mode == "precompute" else "700-729")
    p.add_argument("--train-task", choices="SML", default="S")
    p.add_argument("--train-resource", choices=resource_choices)
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
        scene = scenario_class(task=a.task, resource=a.resource, deadline=a.deadline, mode=mode,
            episodes=a.episodes, seed_spec=a.seeds, train_task=a.train_task,
            train_resource=a.train_resource, train_deadline=a.train_deadline,
            checkpoint_override=a.checkpoint_dir, cache_output=a.out,
            resource_config=a.resource_config, cache_file=a.deadline_cache,
            experiment_protocol=a.experiment_protocol, legacy_split=a.legacy_split,
            model_seed=a.model_seed, workload_profile=a.workload_profile, normalization_mode=a.normalization_mode,
            normalization_config=a.normalization_config, device=a.device,
            diagnostics_every=a.diagnostics_every, profile_mode=a.profile_mode, profile_every=a.profile_every)
    except (ValueError, OSError, TypeError) as exc:
        p.error(str(exc))
    if a.dry_run:
        print(json.dumps(scene.describe(), ensure_ascii=False, indent=2))
        raise SystemExit(0)
    return scene
