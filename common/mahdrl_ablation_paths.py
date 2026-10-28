"""Short training paths and portable, flat CSV names for the Fast ablations."""
import hashlib
import json
from pathlib import Path

from common.r010_paths import training_name
from common.f11 import LEARNING_MODEL
from common.ta_weights import TA_WEIGHT_MODEL


class AblationPaths:
    # Subclasses supply output_tag and path_settings() = (version, variant metadata).
    @staticmethod
    def scene_tag(task, resource, deadline):
        return task + resource.replace("custom_", "C") + "_" + {
            "Tight": "T", "Medium": "M", "Loose": "L"}[deadline]

    @property
    def target_tag(self):
        return self.scene_tag(self.task, self.resource, self.deadline)

    @property
    def source_tag(self):
        if self.mode != "eval":
            return self.target_tag
        return self.scene_tag(self.train_task, self.train_resource or self.resource,
                              self.train_deadline or self.deadline)

    def model_name(self, algorithm=None):
        version, variant = self.path_settings()
        name = training_name(self, version, dict(protocol=self.selection_protocol(),
            variant=variant, learning_model=LEARNING_MODEL, ta_weight_model=TA_WEIGHT_MODEL))
        return self.source_tag + name[name.index("_m"):]

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
        seed_tag = str(seeds[0]) if len(seeds) == 1 else f"{seeds[0]}-{seeds[-1]}"
        if seeds != list(range(seeds[0], seeds[-1] + 1)):
            seed_tag += "_" + hashlib.sha256(json.dumps(seeds).encode()).hexdigest()[:8]
        config_tag = self.model_name().rsplit("_", 1)[-1]
        return self.evaluation_root/(
            f"eval_{self.output_tag}_noapf_srgt092_{self.target_tag}_from_{self.source_tag}_"
            f"m{self.model_seed}_s{seed_tag}_{config_tag}.csv")

    def evaluation_details_dir(self):
        key = hashlib.sha256(self.evaluation_csv().name.encode()).hexdigest()[:16]
        return self.evaluation_root/"_details"/key

    def evaluation_dir(self):
        return self.evaluation_details_dir()

    def evaluation_identity(self):
        version, variant = self.path_settings()
        return dict(algorithm=variant["algorithm"], model_version=version,
            host_apf_enabled=False, vm_apf_enabled=False, scene=self.target_tag,
            task=self.task, resource=self.resource, deadline=self.deadline,
            source_scene=self.source_tag, train_task=self.train_task,
            train_resource=self.train_resource or self.resource,
            train_deadline=self.train_deadline or self.deadline)
