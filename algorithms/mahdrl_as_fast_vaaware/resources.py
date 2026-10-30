"""Fixed physical topologies with explicit, shared R50 deadline and state references."""
import json
from pathlib import Path

from common.scenarios import Scenario, VM_COUNTS
from common.f18_state_normalization import scene_id
from common.f22_cache import check_cache
from common.precompute_deadlines import main as generate_reference
from .scenario import ScoreScenario
from .config import variant_metadata, implementation_metadata


class ResourceScenario(ScoreScenario):
    """Only concrete R100/R150 entry points select a physical resource preset."""

    def __post_init__(self):
        code = f"R{sum(self.physical_vms)}"
        if self.resource != code or self.train_resource not in (None, code):
            raise ValueError(f"This entry point fixes both source and target resources to {code}")
        if self.resource_config:
            raise ValueError("Fixed resource variants do not accept --resource-config")
        if self.workload_profile != "standard":
            raise ValueError("Fixed-load resource variants require standard: 50 workflows, arrival rate 0.03")
        # The shared parser validates S/M/L references; physical identity is restored immediately.
        original_source = self.train_resource
        object.__setattr__(self, "resource", "M")
        object.__setattr__(self, "train_resource", "M" if original_source else None)
        super().__post_init__()
        object.__setattr__(self, "resource", code)
        object.__setattr__(self, "train_resource", original_source)

    @property
    def vms_per_host(self):
        return self.physical_vms

    def deadline_reference_scenario(self):
        return Scenario(task=self.task, resource="M", deadline=self.deadline, mode=self.mode,
            episodes=self.episodes, seed_spec=self.seed_spec, model_seed=self.model_seed,
            train_task=self.train_task, train_resource="M", train_deadline=self.train_deadline,
            experiment_protocol=self.experiment_protocol, legacy_split=self.legacy_split,
            normalization_mode=self.normalization_mode, normalization_config=self.normalization_config,
            cache_output=self.cache_output)

    @property
    def cache_path(self):
        return Path(self.cache_file).resolve() if self.cache_file else self.deadline_reference_scenario().cache_path

    @property
    def precompute_path(self):
        return self.deadline_reference_scenario().precompute_path

    def model_name(self, algorithm=None):
        name = super().model_name()
        task = self.task if self.mode != "eval" else self.train_task
        deadline = self.deadline if self.mode != "eval" else (self.train_deadline or self.deadline)
        # The parent already identifies R100/R150; CSV scene tags retain the full scale.
        return self.scene_tag(task, "", deadline) + name[len(self.source_tag):]

    def normalization_kwargs(self):
        task = self.task if self.mode == "train" else self.train_task
        deadline = self.deadline if self.mode == "train" else (self.train_deadline or self.deadline)
        return dict(normalization_mode=self.normalization_mode, normalization_config=self.normalization_config,
                    source_train_scene=scene_id(task, "M", deadline))

    def resource_metadata(self):
        return dict(version="fixed_load_r50_reference_v1",
            physical_topology=dict(resource=self.resource, num_hosts=self.num_hosts,
                                   vms_per_host=list(self.vms_per_host), num_vms=sum(self.vms_per_host)),
            deadline_reference=dict(resource="M", num_hosts=6, vms_per_host=list(VM_COUNTS["M"]),
                policy="FCFS_task + VM_first_idle", arrival_lambda=.03, workflows_per_episode=50),
            normalization_reference_resource="M", arrival_lambda=.03, workflows_per_episode=50)

    def variant_metadata(self):
        metadata = variant_metadata(self.mechanism)
        metadata.update(algorithm=self.algorithm, scheme=self.model_version, model_version=self.model_version,
                        resource_experiment=self.resource_metadata())
        return metadata

    def implementation_metadata(self):
        metadata = implementation_metadata(self.mechanism)
        metadata.update(algorithm=self.algorithm, implementation_version=self.model_version,
                        scientific_variant=self.variant_metadata())
        return metadata

    def selection_protocol(self):
        return dict(super().selection_protocol(), resource_experiment=self.resource_metadata())

    def reference_identity(self):
        return dict(physical_source_scene=self.source_tag,
            deadline_reference_scene=scene_id(self.task, "M", self.deadline),
            normalization_reference_scene=self.normalization_kwargs()["source_train_scene"],
            resource_experiment=self.resource_metadata())

    def reproducibility_metadata(self):
        return dict(super().reproducibility_metadata(), **self.reference_identity())

    def normalization_metadata(self):
        return dict(super().normalization_metadata(), **self.reference_identity())

    def evaluation_identity(self):
        return dict(super().evaluation_identity(), physical_num_hosts=self.num_hosts,
            physical_num_vms=sum(self.vms_per_host), deadline_reference_resource="M",
            normalization_reference_scene=self.normalization_kwargs()["source_train_scene"],
            resource_reference_protocol="fixed_load_r50_reference_v1")

    def describe(self, algorithm=None):
        result = dict(super().describe(), **self.reference_identity())
        if self.mode == "precompute":
            result.update(output=str(self.precompute_path),
                          precompute_scope="R50 reference only; physical topology remains recorded separately")
        return result

    @staticmethod
    def scene_tag(task, resource, deadline):
        return ScoreScenario.scene_tag(task, resource.removeprefix("R"), deadline)


def precompute_reference(scene):
    """Reuse a validated complete reference; explicit outputs always refuse overwrites."""
    if scene.cache_file:
        raise ValueError("Use --out for reference precompute, --deadline-cache for train/evaluate")
    reference = scene.deadline_reference_scenario()
    output = reference.precompute_path
    if output.exists() and not scene.cache_output:
        check_cache(reference, path=output)
        obj = json.loads(output.read_text(encoding="utf-8"))
        rows = obj.get("data")
        available = {int(r["seed"]) for r in rows} if rows is not None else {int(k) for k in obj["seed_to_record"]}
        if not set(scene.seeds) <= available:
            raise ValueError("R50 reference does not cover requested precompute seeds")
        print(f"[reuse R50 reference] {output}", flush=True)
    else:
        generate_reference(reference)
    return output
