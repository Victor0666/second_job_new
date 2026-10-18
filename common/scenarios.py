"""Shared, explicit scenario settings for training, evaluation and cache generation."""

from dataclasses import dataclass
from pathlib import Path
import argparse
from collections import Counter
import json

from common.read_xml_opt_Tsize import DATA_MODEL
from common.f11 import LEARNING_MODEL
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
            "Use an inclusive range such as 2-31, or distinct seeds such as 2,5,8"
        ) from None


@dataclass(frozen=True)
class Scenario:
    task: str = "S"
    resource: str = "S"
    deadline: str = "Tight"
    mode: str = "eval"
    train_task: str = "S"
    episodes: int = 600
    seed_spec: str = "2-31"
    checkpoint_override: str | None = None
    cache_output: str | None = None
    marl_options: dict | None = None

    def __post_init__(self):
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
        return MODEL_NAMES[algorithm].format(
            task=task, resource=self.resource, deadline=self.deadline
        ) + "_" + LEARNING_MODEL + ("_" + TA_WEIGHT_MODEL if algorithm in TA_ALGORITHMS else "")

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
            default="0-1000" if mode == "precompute" else "2-31",
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
    args = parser.parse_args()
    try:
        scenario = Scenario(
            task=args.task,
            resource=args.resource,
            deadline=getattr(args, "deadline", "Tight"),
            mode=mode,
            train_task=getattr(args, "train_task", "S"),
            episodes=getattr(args, "episodes", 600),
            seed_spec=getattr(args, "seeds", "2-31"),
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
    return scenario
