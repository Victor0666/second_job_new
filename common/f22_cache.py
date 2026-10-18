"""Audit F22 cache coverage; optionally supplement missing seeds into a NEW file."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from common.experiment_protocol import DEFAULT_PROTOCOL, ExperimentProtocol
from common.read_xml_opt_Tsize import dax_template_id, DAX_TEMPLATE_ID_SCHEME
from common.workflow_opt import validate_deadline_cache


def inspect_cache(scenario, path=None):
    path = Path(path or scenario.cache_path)
    protocol = scenario.protocol() or ExperimentProtocol.load(DEFAULT_PROTOCOL)
    required = set(protocol.train) | set(protocol.validation) | set(protocol.test)
    if scenario.mode == 'eval': required.update(scenario.seeds)
    if not path.exists():
        return dict(path=str(path), status='missing', missing=sorted(required)), None
    obj = json.loads(path.read_text(encoding='utf-8'))
    validate_deadline_cache(obj, path)
    if obj['meta'].get('dax_template_id_scheme') != DAX_TEMPLATE_ID_SCHEME:
        raise ValueError('Deadline cache template identifier version mismatch')
    from common.precompute_deadlines import build_env_kwargs
    expected = build_env_kwargs(scenario)
    for key in ('arrival_lambda', 'workflows_per_episode', 'num_hosts', 'vms_per_host', 'pc_tiers', 'bw_tiers'):
        value = list(expected[key]) if isinstance(expected[key], tuple) else expected[key]
        if obj['meta'].get(key) != value:
            raise ValueError('Deadline cache configuration mismatch: ' + key)
    for key in ('task', 'resource'):
        if obj['meta'].get(key) != getattr(scenario, key):
            raise ValueError('Deadline cache scenario mismatch: ' + key)
    if [dax_template_id(x) for x in obj['meta'].get('dax_paths', [])] != [dax_template_id(x) for x in scenario.dax_paths()]:
        raise ValueError('Deadline cache DAX mixture mismatch')
    if obj['meta'].get('policy') != 'FCFS_task + VM_first_idle':
        raise ValueError('Deadline reference makespan policy mismatch')
    rows = obj.get('data')
    if rows is None: rows = [dict(v, seed=int(k)) for k, v in obj['seed_to_record'].items()]
    seeds = [int(r['seed']) for r in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError('Duplicate deadline cache seeds')
    for row in rows:
        if int(row['seed']) not in required: continue
        for key in ('arrival_times', 'wf_dax_names', 'wf_makespans', 'wf_instance_hashes'):
            if len(row.get(key, [])) != expected['workflows_per_episode']:
                raise ValueError(f'Deadline cache incomplete {key} for seed={row["seed"]}')
    missing = sorted(required-set(seeds))
    return dict(path=str(path), status='incomplete' if missing else 'ok', missing=missing,
                required_count=len(required), available_count=len(seeds)), dict(meta=obj['meta'], data=rows)


def check_cache(scenario, path=None):
    report, _ = inspect_cache(scenario, path)
    if report['missing']:
        raise ValueError(f'F22 deadline cache {report["status"]}: {report["path"]}; missing seeds={seed_ranges(report["missing"])}. '
                         'Run common/f22_cache.py to inspect and supplement; no seed fallback is allowed.')
    return report


def supplement(scenario, output):
    output = Path(output)
    if output.exists(): raise FileExistsError(f'Refusing to overwrite cache: {output}')
    report, original = inspect_cache(scenario)
    if not report['missing']: raise ValueError('Cache already covers F22; no regeneration needed')
    from common.precompute_deadlines import main as precompute
    with tempfile.TemporaryDirectory(prefix='f22_cache_') as directory:
        partial = Path(directory)/'partial.json'
        precompute(replace(scenario, mode='precompute', seed_spec=','.join(map(str, report['missing'])), cache_output=str(partial)))
        generated = json.loads(partial.read_text(encoding='utf-8'))
    rows = (original['data'] if original else []) + generated['data']
    merged = dict(meta=original['meta'] if original else generated['meta'], data=sorted(rows, key=lambda r:int(r['seed'])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream: json.dump(merged, stream, indent=2)
    return check_cache(scenario, output)


def seed_ranges(seeds):
    groups=[]
    for seed in sorted(set(seeds)):
        if groups and seed==groups[-1][-1]+1: groups[-1].append(seed)
        else: groups.append([seed])
    return ','.join(str(g[0]) if len(g)==1 else f'{g[0]}-{g[-1]}' for g in groups)


def compact_report(report):
    result=dict(report)
    if 'missing' in result:
        result['missing_count']=len(result['missing'])
        result['missing_seeds']=seed_ranges(result.pop('missing'))
    return result


def main():
    from common.scenarios import Scenario
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--task', choices='SML', default='S')
    parser.add_argument('--resource', choices='SML', default='S')
    parser.add_argument('--all', action='store_true', help='Audit all nine task/resource cache combinations')
    parser.add_argument('--supplement-out', type=Path, help='Explicitly generate missing seeds only, into a new file')
    args = parser.parse_args()
    if args.all and args.supplement_out: parser.error('Supplement one task/resource at a time')
    pairs = [(t,r) for t in 'SML' for r in 'SML'] if args.all else [(args.task,args.resource)]
    reports = []
    for task, resource in pairs:
        scene = Scenario(task=task,resource=resource,mode='precompute')
        try: reports.append(supplement(scene,args.supplement_out) if args.supplement_out else inspect_cache(scene)[0])
        except (OSError,ValueError,KeyError) as exc: reports.append(dict(task=task,resource=resource,status='incompatible',error=str(exc)))
    print(json.dumps([compact_report(r) for r in reports],indent=2))
    return 0 if all(r['status']=='ok' for r in reports) else 1


if __name__ == '__main__': raise SystemExit(main())
