"""Reuse parsing and short paths without changing the VAware implementation."""
from dataclasses import dataclass
from algorithms.mahdrl_as_fast_vaaware.scenario import ScoreScenario as BaseScenario
from algorithms.mahdrl_as_fast_vaaware.scenario import parse_scenario as base_parse
from .config import (ALGORITHM, DISPLAY_NAME, MODEL_VERSION, OUTPUT_TAG,
                     VM_CAPACITY, variant_metadata, implementation_metadata)
from .protocol import StrictSelection


@dataclass(frozen=True)
class ScoreScenario(StrictSelection, BaseScenario):
    algorithm = ALGORITHM
    display_name = DISPLAY_NAME
    model_version = MODEL_VERSION
    output_tag = OUTPUT_TAG

    train_seed_cycles: int = 1

    def __post_init__(self):
        if type(self.train_seed_cycles) is not int or self.train_seed_cycles < 1:
            raise ValueError('train-seed-cycles must be a positive integer')
        if self.train_seed_cycles > 1:
            protocol = self.protocol()
            if protocol is None or self.mode not in ('train', 'eval'):
                raise ValueError('Repeated training seeds require a disjoint train/eval protocol')
            if protocol.episodes != len(protocol.train):
                raise ValueError('Repeated training requires one protocol episode per unique training seed')
            if self.episodes != self.training_episode_limit(protocol):
                raise ValueError('--episodes must equal protocol episodes * --train-seed-cycles (also at evaluation)')
        super().__post_init__()
        if self.mode == 'eval' and self.protocol() and self.episodes > self.training_episode_limit(self.protocol()):
            raise ValueError('Evaluation source budget exceeds the protocol; specify --train-seed-cycles')
        if max(self.vms_per_host) > VM_CAPACITY:
            raise ValueError(f"Fixed VA capacity is {VM_CAPACITY} VMs per host; topology exceeds capacity")

    def training_episode_limit(self, protocol):
        return protocol.episodes * self.train_seed_cycles

    def training_seed(self, episode):
        if self.train_seed_cycles == 1:
            return super().training_seed(episode)
        if not 0 <= episode < self.episodes:
            raise ValueError('Episode outside the configured training budget')
        seeds = self.protocol().train
        return seeds[episode % len(seeds)]

    @property
    def training_schedule_tag(self):
        return f'e{self.episodes}r{self.train_seed_cycles}'

    def selection_protocol(self):
        metadata = super().selection_protocol()
        if self.train_seed_cycles > 1:
            metadata['training_schedule'] = dict(order='fixed_seed_cycle', episodes=self.episodes,
                train_seed_cycles=self.train_seed_cycles, unique_training_seeds=len(self.protocol().train))
        return metadata

    def model_name(self, algorithm=ALGORITHM):
        name = super().model_name(algorithm)
        if self.train_seed_cycles > 1:
            name = name.replace('_m', '_' + self.training_schedule_tag + '_m', 1)
        return name

    def evaluation_csv(self):
        path = super().evaluation_csv()
        if self.train_seed_cycles > 1:
            path = path.with_name(path.stem + '_' + self.training_schedule_tag + path.suffix)
        return path

    def variant_metadata(self):
        return variant_metadata(self.mechanism)

    def implementation_metadata(self):
        return implementation_metadata(self.mechanism)

    def evaluation_identity(self):
        identity = dict(super().evaluation_identity(), vm_capacity=VM_CAPACITY,
            vm_input_dim=93, vm_selection_scope="host_local_fixed_slots")
        if self.train_seed_cycles > 1:
            identity.update(training_episodes=self.episodes, train_seed_cycles=self.train_seed_cycles)
        return identity


def parse_scenario(mode, argv=None):
    return base_parse(mode, argv, scenario_class=ScoreScenario)
