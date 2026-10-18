"""F22 immutable physical-instance identities and run provenance; no policy features."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
from common.read_xml_opt_Tsize import data_model_metadata, workflow_instance_hash, dax_template_id


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=32)
def file_hash(path, size, modified):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_info(path):
    path = Path(path)
    if not path.is_file():
        return dict(path=str(path), status='missing')
    stat = path.stat()
    return dict(path=str(path), sha256=file_hash(str(path), stat.st_size, stat.st_mtime_ns))


@lru_cache(maxsize=1)
def git_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def reproducibility_metadata(scenario):
    protocol = scenario.protocol()
    return dict(data_split=protocol.metadata() if protocol else 'Legacy', model_seed=scenario.model_seed,
                source_train_scene=scenario.normalization_kwargs()['source_train_scene'],
                git_commit=git_commit(), deadline_cache=artifact_info(scenario.cache_path),
                instance_generator=data_model_metadata(), instance_identity_version='f22_instance_v1')


def instance_identity(env, scenario, instance_seed):
    """Call after all arrivals. Ignore scheduling order, observations and task subdeadlines."""
    if len(env.workflows) != env.workflows_per_episode:
        raise ValueError('Instance hash requires every workflow to have arrived')
    workflows = []
    for i, wf in enumerate(env.workflows):
        name = getattr(wf, 'dax_path', None) or env.wf_dax_name[i]
        workflows.append(dict(template=dax_template_id(name), arrival=float(wf.arrival_time),
                              deadline=float(wf.deadline), payload=workflow_instance_hash(wf.graph, wf.tasks)))
    resources = dict(hosts=[dict(id=k, vm_ids=h.vm_ids, total_pc=h.total_pc, total_bw=h.total_bw,
                                        power=[h.power(i/10) for i in range(11)]) for k, h in sorted(env.hosts.items())],
                     vms=[dict(id=k, **vars(v)) for k, v in sorted(env.vms.items())])
    # Resource dataclasses contain only physical configuration; JSON guards unexpected state.
    contents = dict(workflows=workflows, resources=resources)
    identifier = dict(version='f22_instance_v1', task=scenario.task, resource=scenario.resource,
                      deadline=scenario.deadline, instance_seed=int(instance_seed))
    return dict(**identifier, content_hash=stable_hash(contents), instance_hash=stable_hash(dict(**identifier, **contents)))


def write_run_metadata(directory, scenario):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'data_split.json').write_text(json.dumps(scenario.reproducibility_metadata(), indent=2), encoding='utf-8')


def episode_identity(env, scenario):
    seed = getattr(env, 'episode_seed', env.random_seed)
    if len(env.workflows) != env.workflows_per_episode:
        return dict(instance_seed=int(seed), status='incomplete_arrivals')
    return instance_identity(env, scenario, seed)
