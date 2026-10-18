"""Explicit shared experiment partitions. Historical runs without a protocol stay labelled legacy."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1]/'configs/paired_cloud_protocol_v1.json'


@dataclass(frozen=True)
class ExperimentProtocol:
    train: tuple
    validation: tuple
    test: tuple
    episodes: int = 600
    selection: str = 'minimum_validation_total_energy'
    version: str = 'paired_cloud_disjoint_v1'

    def __post_init__(self):
        groups=[set(self.train),set(self.validation),set(self.test)]
        if any(not g for g in groups) or any(len(g)!=len(seq) for g,seq in zip(groups,(self.train,self.validation,self.test))):
            raise ValueError('Empty or duplicate experiment seeds')
        if any(groups[i]&groups[j] for i in range(3) for j in range(i)):
            raise ValueError('Training, validation and test seeds must be disjoint')
        if min(min(g) for g in groups)<0 or len(self.train)<self.episodes:
            raise ValueError('Insufficient training seeds or invalid seeds')
        if self.selection!='minimum_validation_total_energy':
            raise ValueError('Selection must match the existing energy-only validation rule')
        # Existing loops increment a seed once per episode; reject unsupported gaps.
        if self.train!=tuple(range(self.train[0],self.train[0]+len(self.train))):
            raise ValueError('Shared existing training loops require contiguous explicit training seeds')

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(tuple(d['train']),tuple(d['validation']),tuple(d['test']),d['episodes'],d['selection'],d['version'])

    def metadata(self):
        d=dict(version=self.version,train=self.train,validation=self.validation,test=self.test,
               episodes=self.episodes,selection=self.selection)
        return json.loads(json.dumps(d))


def protocol_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()