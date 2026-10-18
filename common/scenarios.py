"""Shared, explicit scenario settings for training, evaluation and cache generation."""

from dataclasses import dataclass
from pathlib import Path
import argparse
from collections import Counter
import json

from common.read_xml_opt_Tsize import DATA_MODEL
from common.f11 import LEARNING_MODEL
from common.experiment_protocol import ExperimentProtocol, protocol_hash, DEFAULT_PROTOCOL
from common.f18_state_normalization import VERSION, CURRENT_VERSION, SHARED_VERSIONS, DEFAULT_CONFIG, scene_id, load_normalizer
from common.ta_weights import TA_WEIGHT_MODEL, TA_ALGORITHMS

ROOT = Path(__file__).resolve().parents[1]
VM_COUNTS = {"S": (9, 8, 8), "M": (9, 9, 8, 8, 8, 8), "L": (9, 9, 9, 8, 8, 8, 8, 8, 8)}
SMALL_PROBABILITY = {"Tight": 0.8, "Medium": 0.5, "Loose": 0.2}
DAX_GROUPS = (
    ("CyberShake_30.xml", "Epigenomics_24.xml", "LIGO_30.xml", "Montage_25.xml", "Sipht_29.xml"),
    ("CyberShake_50.xml", "Epigenomics_47.xml", "LIGO_50.xml", "Montage_50.xml", "Sipht_58.xml"),
    (
        "CyberShake_100.xml",
        "Epigenomics_100.xml",
        "LIGO_100.xml",
        "Montage_100.xml",
        "Sipht_97.xml",
    ),
)
MODEL_NAMES = {
    "mahdrl": "mahdrl_task{task}_res{resource}_seed1_r075_ddlMix_hv_{deadline}",
    "ablation_hostrand": "mahdrl_task{task}_res{resource}_seed1_r075_ddlMix_hostrand_hv_{deadline}",
    "ablation_mgrrand": "mahdrl_task{task}_res{resource}_seed1_mgrrand_r075_ddlMix_hv_{deadline}",
    "ablation_vmrand": "mahdrl_task{task}_res{resource}_seed1_r075_ddlMix_vmrand_hv_{deadline}",
    "ablation_nohost": "hrl2_task{task}_res{resource}_seed1_nohost_r075_ddlMix_{deadline}",
    "ablation_tl2": "tl2_task{task}_res{resource}_seed1_globalEnergy_r075_ddlMix_{deadline}",
    "marl": "marl_task{task}_res{resource}_eval1_ep1000_mix_{deadline}",
    "pd3qn": "pd3qn_task{task}_res{resource}_warmup_mix_nr_{deadline}",
    "irws": "irws_task{task}_res{resource}_mix_{deadline}_rewardPD3QN_{deadline}",
}


def parse_seeds(value):
    try:
        if "-" in value:
            first, last = map(int, value.split("-"))
            if first < 0 or last < first:
                raise ValueError
            seeds = tuple(range(first, last + 1))
        else:
            seeds = tuple(int(item) for item in value.split(","))
        if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
            raise ValueError
        return seeds
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(
            "Use an inclusive range such as 700-729, or distinct seeds such as 700,702,705"
        ) from None


@dataclass(frozen=True)
class Scenario:
    task: str = "S"
    resource: str = "S"
    deadline: str = "Tight"
    mode: str = "eval"
    train_task: str = "S"
    episodes: int = 600
    seed_spec: str = "700-729"
    checkpoint_override: str | None = None
    cache_output: str | None = None
    marl_options: dict | None = None
    paper_options: dict | None = None

    experiment_protocol: str | None = str(DEFAULT_PROTOCOL)
    legacy_split: bool = False
    model_seed: int = 1

    def protocol(self):
        return ExperimentProtocol.load(self.experiment_protocol) if self.experiment_protocol and not self.legacy_split else None

    def first_training_seed(self):
        return self.protocol().train[0] if self.protocol() else 1

    def training_seed(self, episode):
        protocol = self.protocol()
        if not 0 <= episode < self.episodes:
            raise ValueError("Episode outside the configured training budget")
        return protocol.train[episode] if protocol else 1 + episode

    def validation_seeds(self):
        return list(self.protocol().validation) if self.protocol() else [1]

    normalization_mode: str = CURRENT_VERSION
    normalization_config: str = str(DEFAULT_CONFIG)
    train_resource: str | None = None
    train_deadline: str | None = None

    def normalization_kwargs(self):
        source = scene_id(self.task if self.mode == 'train' else self.train_task,
                          self.resource if self.mode == 'train' else (self.train_resource or self.resource),
                          self.deadline if self.mode == 'train' else (self.train_deadline or self.deadline))
        return dict(normalization_mode=self.normalization_mode, normalization_config=self.normalization_config, source_train_scene=source)

    def normalization_metadata(self):
        kw = self.normalization_kwargs()
        n = load_normalizer(kw['normalization_mode'], kw['normalization_config'], kw['source_train_scene'])
        metadata = dict(n.metadata) if n else {'normalization_version':'legacy'}
        if self.protocol():
            metadata.update(experiment_protocol_hash=protocol_hash(self.experiment_protocol),
                            experiment_protocol=self.protocol().metadata())
        metadata['reproducibility'] = self.reproducibility_metadata()
        return metadata

    def reproducibility_metadata(self):
        from common.f22_instances import reproducibility_metadata
        return reproducibility_metadata(self)


    def __post_init__(self):
        if not 0 <= self.model_seed <= 0xffffffff:
            raise ValueError("Model seed must be an unsigned 32-bit integer")
        if (
            self.task not in VM_COUNTS
            or self.resource not in VM_COUNTS
            or self.train_task not in VM_COUNTS
        ):
            raise ValueError("Task and resource sizes must be S, M or L")
        if (
            self.deadline not in SMALL_PROBABILITY
            or self.mode not in {"train", "eval", "precompute"}
            or self.episodes < 1
        ):
            raise ValueError("Invalid deadline, mode or episode count")
        if self.train_resource is not None and self.train_resource not in VM_COUNTS:
            raise ValueError('Invalid source resource')
        if self.train_deadline is not None and self.train_deadline not in SMALL_PROBABILITY:
            raise ValueError('Invalid source deadline')
        if self.normalization_mode not in ('legacy', *SHARED_VERSIONS):
            raise ValueError('Unknown observation normalization mode')
        if not self.legacy_split and not self.experiment_protocol:
            raise ValueError("Use explicit --legacy-split for historical overlapping protocols")
        if self.protocol():
            protocol=self.protocol()
            if self.mode == 'train' and self.episodes > protocol.episodes:
                raise ValueError('Episode budget exceeds the shared protocol')
            if self.mode == 'eval' and set(parse_seeds(self.seed_spec)) & (set(protocol.train) | set(protocol.validation)):
                raise ValueError('Evaluation seeds overlap training or validation partition')
        parse_seeds(self.seed_spec)

    @property
    def vms_per_host(self):
        return VM_COUNTS[self.resource]

    @property
    def num_hosts(self):
        return len(self.vms_per_host)

    @property
    def small_probability(self):
        return SMALL_PROBABILITY[self.deadline]

    @property
    def deadline_probs(self):
        # Explicit values avoid float rounding differences from 1 - 0.8.
        return {"Tight": (0.8, 0.2), "Medium": (0.5, 0.5), "Loose": (0.2, 0.8)}[self.deadline]

    @property
    def seeds(self):
        return parse_seeds(self.seed_spec)

    @property
    def cache_name(self):
        return f"ddl_fcfs_task{self.task}_res{self.resource}_seed0-1000.json"

    @property
    def cache_path(self):
        return ROOT / "data" / "deadlines" / DATA_MODEL / self.cache_name

    @property
    def legacy_cache_path(self):
        return ROOT / "data" / "deadlines" / self.cache_name

    @property
    def precompute_path(self):
        if self.cache_output:
            return Path(self.cache_output).resolve()
        if self.seeds == tuple(range(1001)):
            return self.cache_path
        tag = (
            f"{self.seeds[0]}-{self.seeds[-1]}"
            if self.seeds == tuple(range(self.seeds[0], self.seeds[-1] + 1))
            else "_".join(map(str, self.seeds))
        )
        return self.cache_path.with_name(
            f"ddl_fcfs_task{self.task}_res{self.resource}_seed{tag}.json"
        )

    def dax_paths(self, directory=None):
        directory = Path(directory) if directory is not None else ROOT / "data" / "dax"
        groups = [[str(directory / name) for name in group] for group in DAX_GROUPS]
        weights = {"S": (1, 0, 0), "M": (3, 7, 0), "L": (2, 3, 5)}[self.task]
        return [path for group, weight in zip(groups, weights) for path in group * weight]

    def model_name(self, algorithm):
        task = self.task if self.mode == "train" else self.train_task
        name = MODEL_NAMES[algorithm].format(
            task=task, resource=(self.train_resource or self.resource) if self.mode == "eval" else self.resource,
            deadline=(self.train_deadline or self.deadline) if self.mode == "eval" else self.deadline
        ) + "_" + LEARNING_MODEL + ("_" + TA_WEIGHT_MODEL if algorithm in TA_ALGORITHMS else "") + ("_" + self.normalization_mode if self.normalization_mode in SHARED_VERSIONS else "") + ("_protocol_" + protocol_hash(self.experiment_protocol)[:12] if self.protocol() else "")
        return name.replace("_seed1_", "_seed" + str(self.first_training_seed()) + "_") if self.protocol() else name

    @property
    def output_version(self):
        return self.normalization_mode + ('_protocol_' + protocol_hash(self.experiment_protocol)[:12] if self.protocol() else '_Legacy')

    def checkpoint_dir(self, algorithm):
        if self.mode == "eval" and self.checkpoint_override:
            return Path(self.checkpoint_override).resolve()
        return ROOT / "outputs" / "ckpts" / self.model_name(algorithm)

    def log_dir(self, algorithm):
        return ROOT / "outputs" / "logs" / self.model_name(algorithm)

    def describe(self, algorithm=None):
        result = dict(
            task=self.task,
            resource=self.resource,
            deadline=self.deadline,
            mode=self.mode,
            experiment_protocol=self.protocol().metadata() if self.protocol() else "legacy_overlapping_seeds",
            model_seed=self.model_seed,
            normalization=self.normalization_kwargs() if self.mode != 'precompute' else None,
            num_hosts=self.num_hosts,
            vms_per_host=self.vms_per_host,
            dax_weights=dict(Counter(Path(p).name for p in self.dax_paths())),
            cache=str(self.cache_path),
            deadline_choices=(2.0, 3.0),
            deadline_probs=self.deadline_probs,
            episodes=self.episodes if self.mode == "train" else None,
            seeds=self.seeds if self.mode != "train" else None,
        )
        if self.mode == "precompute":
            result["output"] = str(self.precompute_path)
            result["deadline_mode"] = "none"
            for key in ("deadline", "deadline_choices", "deadline_probs", "cache"):
                result.pop(key, None)
        if self.marl_options:
            result["overrides"] = self.marl_options
        if algorithm in MODEL_NAMES:
            result["checkpoint_dir"] = str(self.checkpoint_dir(algorithm))
            result["model_training_task"] = self.task if self.mode == "train" else self.train_task
        return result


def parse_scenario(mode, algorithm=None):
    parser = argparse.ArgumentParser(description=f'{algorithm or "Deadline cache"}: {mode}')
    parser.add_argument("--task", choices=VM_COUNTS, default="S", help="Workflow size mixture")
    parser.add_argument("--resource", choices=VM_COUNTS, default="S", help="Cluster size")
    if mode != "precompute":
        parser.add_argument("--deadline", choices=SMALL_PROBABILITY, default="Tight")
    if mode == "precompute":
        parser.add_argument(
            "--out", help="Cache output file; partial seed runs otherwise use a separate filename"
        )
    if mode == "train":
        parser.add_argument("--episodes", type=int, default=600)
    else:
        parser.add_argument(
            "--seeds",
            default="0-1000" if mode == "precompute" else "700-729",
            help="Inclusive range or comma-separated seeds",
        )
    if mode == "eval" and algorithm != "ertf_fifs":
        parser.add_argument(
            "--train-task",
            choices=VM_COUNTS,
            default="S",
            help="Workflow size used to train the evaluated model",
        )
        parser.add_argument(
            "--checkpoint-dir",
            help="Explicit model directory; no automatic search across scenarios",
        )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected configuration without running experiments",
    )
    if algorithm == "marl" and mode == "eval":
        for flag in ("--deadline-cache", "--ckpt-global", "--ckpt-local", "--out", "--device"):
            parser.add_argument(flag)
    if mode != 'precompute':
        parser.add_argument('--experiment-protocol', default=str(DEFAULT_PROTOCOL))
        parser.add_argument('--legacy-split', action='store_true', help='Explicit historical data split; never labelled F22')
        parser.add_argument('--model-seed', type=int, default=1)
    if mode != 'precompute' and algorithm != 'ertf_fifs':
        parser.add_argument('--normalization-mode', choices=(*SHARED_VERSIONS, 'legacy'), default=CURRENT_VERSION)
        parser.add_argument('--normalization-config', default=str(DEFAULT_CONFIG))
        if mode == 'eval':
            parser.add_argument('--train-resource', choices=VM_COUNTS)
            parser.add_argument('--train-deadline', choices=SMALL_PROBABILITY)
    if algorithm == 'irws_paper':
        parser.add_argument('--resume-checkpoint')
        parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    try:
        scenario = Scenario(
            paper_options={k:getattr(args,k) for k in ('resume_checkpoint','device')} if algorithm == 'irws_paper' else None,
            experiment_protocol=getattr(args, 'experiment_protocol', str(DEFAULT_PROTOCOL)),
            legacy_split=getattr(args, 'legacy_split', False),
            model_seed=getattr(args, 'model_seed', 1),
            normalization_mode=getattr(args, 'normalization_mode', CURRENT_VERSION),
            normalization_config=getattr(args, 'normalization_config', str(DEFAULT_CONFIG)),
            train_resource=getattr(args, 'train_resource', None), train_deadline=getattr(args, 'train_deadline', None),
            task=args.task,
            resource=args.resource,
            deadline=getattr(args, "deadline", "Tight"),
            mode=mode,
            train_task=getattr(args, "train_task", "S"),
            episodes=getattr(args, "episodes", 600),
            seed_spec=getattr(args, "seeds", "700-729"),
            checkpoint_override=getattr(args, "checkpoint_dir", None),
            cache_output=getattr(args, "out", None) if mode == "precompute" else None,
            marl_options=(
                {
                    name: getattr(args, name)
                    for name in ("deadline_cache", "ckpt_global", "ckpt_local", "out", "device")
                    if getattr(args, name) is not None
                }
                if algorithm == "marl" and mode == "eval"
                else None
            ),
        )
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(scenario.describe(algorithm), indent=2, ensure_ascii=False))
        raise SystemExit(0)
    if mode != 'precompute' and not scenario.legacy_split:
        from common.f22_cache import check_cache
        cache_override = (scenario.marl_options or {}).get('deadline_cache')
        check_cache(scenario, path=cache_override)
    return scenario
