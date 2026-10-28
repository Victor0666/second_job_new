"""Independent source paths and metadata, sharing the validated Fast scenario rules."""
from common.mahdrl_ablation import SELECTION_TAG, selection_metadata
from common.mahdrl_ablation_paths import AblationPaths
import argparse
import json
from pathlib import Path
from common.experiment_protocol import DEFAULT_PROTOCOL
from common.f18_state_normalization import CURRENT_VERSION, DEFAULT_CONFIG, SHARED_VERSIONS
from algorithms.mahdrl_oe_ge_apf_score_fast.scenario import ScoreScenario as BaseScenario
from .config import get_config


class SyncScenario(AblationPaths, BaseScenario):
    @property
    def output_tag(self):
        return "mahdrl_sync_"+self.sync_mode

    def path_settings(self):
        cfg = self.sync_config
        return cfg.model_version, cfg.variant_metadata()

    sync_mode = None

    @property
    def sync_config(self):
        return get_config(self.sync_mode)


    @property
    def output_version(self):
        return self.sync_config.model_version + "_" + super().output_version.replace("select_sr099_minenergy_v1", SELECTION_TAG)


    def selection_protocol(self):
        metadata = super().selection_protocol()
        metadata.update(selection=SELECTION_TAG, checkpoint_selection=selection_metadata())
        metadata["synchronization"] = self.sync_config.synchronization_metadata()
        return metadata

    def reproducibility_metadata(self):
        metadata = super().reproducibility_metadata()
        cfg = self.sync_config
        metadata.update(algorithm=cfg.algorithm, model_version=cfg.model_version, checkpoint_selection=selection_metadata(),
            data_split=self.selection_protocol(), synchronization=cfg.synchronization_metadata())
        return metadata

    def normalization_metadata(self):
        metadata = super().normalization_metadata()
        cfg = self.sync_config
        metadata.update(model_version=cfg.model_version, checkpoint_selection=selection_metadata(), experiment_protocol=self.selection_protocol(),
            synchronization=cfg.synchronization_metadata())
        return metadata

    def describe(self, algorithm=None):
        result = super().describe()
        cfg = self.sync_config
        result.update(checkpoint_selection=selection_metadata(), experiment_protocol=self.selection_protocol(), algorithm=cfg.algorithm, display_name=cfg.display_name, model_version=cfg.model_version,
            variant=cfg.variant_metadata(), implementation=cfg.implementation_metadata(),
            synchronization=cfg.synchronization_metadata(), checkpoint_dir=str(self.checkpoint_dir()),
            log_dir=str(self.log_dir()), evaluation_dir=str(self.evaluation_dir()), evaluation_csv=str(self.evaluation_csv()))
        return result


def parse_scenario(scene_class, mode, argv=None):
    DISPLAY_NAME = get_config(scene_class.sync_mode).display_name
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
        scene = scene_class(task=a.task, resource=a.resource, deadline=a.deadline, mode=mode,
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
