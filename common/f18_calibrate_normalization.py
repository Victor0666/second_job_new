"""ERTF-FIFS training calibration; never loads a neural checkpoint or writes a deadline cache."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from common.scenarios import Scenario, parse_seeds
from common.read_xml_opt_Tsize import dax_template_id
from common.experiment_protocol import ExperimentProtocol, DEFAULT_PROTOCOL, protocol_binding
from common.precompute_deadlines import build_env_kwargs as reference_env_kwargs
from common.f18_state_normalization import VERSION, SCALE_UNITS, scene_id, queue_metrics
from common.workflow_opt import validate_deadline_cache, DEADLINE_ALPHA_SEED_OFFSET
from algorithms.ertf_fifs.evaluate import _force_fcfs_manager, _pick_first_idle_host, _pick_fifs_vm
from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents


def nine_scenarios(protocol_path=DEFAULT_PROTOCOL):
    return [Scenario(task='S', resource=r, deadline=d, mode='train', episodes=ExperimentProtocol.load(protocol_path).episodes, experiment_protocol=str(protocol_path))
            for r in ('S','M','L') for d in ('Loose','Medium','Tight')]


def percentiles(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size or not np.isfinite(a).all():
        raise ValueError('Missing or nonfinite calibration samples')
    return dict(zip(('P50','P95','P99','Max'), map(float, [*np.percentile(a, [50,95,99], method='linear'), a.max()])))


def validate_seeds(seeds, protocol_path=DEFAULT_PROTOCOL):
    protocol = ExperimentProtocol.load(protocol_path)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(protocol.train):
        raise ValueError('Calibration requires seeds from the selected protocol training partition')


def validate_manifest(manifest, protocol_path):
    binding = protocol_binding(protocol_path)
    if any(manifest.get(k) != v for k, v in binding.items()):
        raise ValueError('Calibration manifest protocol missing or mismatched; use a new --out directory')
    for spec in manifest.get('scenes', {}).values():
        validate_seeds(spec['seeds'], protocol_path)
    return binding


def cache_info(scenario, seeds):
    path = scenario.cache_path
    command = f'./run.cmd common/f22_cache.py --task {scenario.task} --resource {scenario.resource}'
    try:
        raw = path.read_bytes(); obj = json.loads(raw)
        validate_deadline_cache(obj, str(path))
        records = {int(x['seed']):x for x in obj['data']} if 'data' in obj else {int(k):v for k,v in obj['seed_to_record'].items()}
        if not set(seeds) <= set(records):
            raise ValueError('missing seeds: ' + str(sorted(set(seeds)-set(records))))
        meta = obj['meta']
        expected = reference_env_kwargs(scenario)
        for key in ('arrival_lambda', 'workflows_per_episode', 'num_hosts', 'vms_per_host', 'pc_tiers', 'bw_tiers'):
            target = list(expected[key]) if isinstance(expected[key], tuple) else expected[key]
            if meta.get(key) != target:raise ValueError('Cache configuration mismatch: ' + key)
        if meta.get('task') != scenario.task or meta.get('resource') != scenario.resource:
            raise ValueError('Cache task/resource mismatch')
        if [dax_template_id(p) for p in meta.get('dax_paths', [])] != [dax_template_id(p) for p in expected['dax_paths']]:
            raise ValueError('Cache DAX mixture mismatch')
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), meta=meta)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f'{exc}. Audit cache and missing seeds: {command}. Use --supplement-out with a new path for missing seeds only. Existing caches are never overwritten; incompatible caches require separate preparation.') from exc


def sample_episode(scenario, seed, raw_records=False):
    validate_seeds([seed], scenario.experiment_protocol)
    kwargs = reference_env_kwargs(scenario)
    kwargs.pop('deadline_alpha')
    kwargs.update(random_seed=seed, deadline_mode='cache_fcfs', deadline_cache_path=str(scenario.cache_path),
                  deadline_alpha_small_prob=scenario.small_probability, normalization_mode='legacy')
    env = CloudWorkflowEnv_VMAgents(**kwargs)
    env.reset()
    rows=[]; seen=set(); empty=0; actions=0
    while not env.done_flag:
        if float(env.current_time) not in seen:
            seen.add(float(env.current_time)); m=queue_metrics(env)
            row=dict(scene_id=scene_id(scenario.task,scenario.resource,scenario.deadline), task_scale=scenario.task,
                resource_scale=scenario.resource,deadline_level=scenario.deadline, seed=seed, episode_id=seed,
                decision_index=len(rows), current_time=float(env.current_time), **m,
                workflow_count=len(env.workflows), completed_workflow_count=env.completed_workflows)
            if m['ready_count'] and env._has_decision_point():rows.append(row)
            else:empty+=1
        _force_fcfs_manager(env)
        while True:
            host,ok=env.get_host_state_for_next_assignment()
            if not ok:break
            # F15 FCFS column is decreasing ReadyTime with (ReadyTime, ID) ties.
            expected=min(env.ready_task_ids,key=lambda i:(env.task_ready_time[i],i))
            if env._cur_tid != expected:raise AssertionError('Reference task order is not ERTF')
            ah=_pick_first_idle_host(env,host['mask']);env.host_select(ah)
            vm,ok=env.get_vm_state_for_current_task()
            if not ok:raise RuntimeError('Reference VM decision unavailable')
            av=_pick_fifs_vm(env,vm['mask'],ah);_,_,info=env.vm_assign(av)
            if info.get('invalid'):raise RuntimeError('Invalid reference assignment')
            actions+=1
            if actions>100000:raise RuntimeError('Reference episode exceeded assignment guard')
        old=env.current_time
        env.finish_phase_and_advance()
        if not env.done_flag and env.current_time==old and not env._has_decision_point():
            raise RuntimeError('Reference episode stalled')
    if env.completed_workflows!=50 or any(s!='Finished' for s in env.task_state):
        raise RuntimeError('Reference episode ended with unfinished tasks')
    physical = dict(workload_mi_ref=list(map(float,env.task_mi)), input_bits_ref=list(map(float,env.task_in_bits)),
        output_bits_ref=list(map(float,env.task_out_bits)),up_rank_ref=list(map(float,env.task_up_rank)),
        down_rank_ref=list(map(float,env.task_down_rank)))
    deadlines=[float(w.deadline-w.arrival_time) for w in env.workflows]
    if not np.isfinite(deadlines).all() or min(deadlines)<=0 or any(not np.isfinite(v).all() for v in physical.values()):
        raise RuntimeError('Invalid physical calibration samples')
    result=dict(status='ok',seed=seed,ready=[r['ready_count'] for r in rows],deadlines=deadlines,
        physical=physical,empty_observations=empty,decisions=len(rows),assignments=actions,
        vm_pc_ref=max(v.pc for v in env.vms.values()),vm_bw_ref=max(v.bw for v in env.vms.values()),
        host_pc_ref=max(h.total_pc for h in env.hosts.values()))
    if raw_records:result['raw_records']=rows
    return result


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(out, manifest, protocol_path=DEFAULT_PROTOCOL):
    binding = validate_manifest(manifest, protocol_path)
    ready_rows=[];deadline_rows=[];summary=[];per_seed=[];configs={};failures=0
    for scenario in nine_scenarios(protocol_path):
        sid=scene_id(scenario.task,scenario.resource,scenario.deadline)
        seeds=manifest.get('scenes',{}).get(sid,{}).get('seeds',[])
        episodes=[]
        for seed in seeds:
            path=out/'episodes'/f'{sid}_{seed}.json'
            ep=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(status='missing',seed=seed,error=manifest.get('scenes',{}).get(sid,{}).get('error','Episode record not generated'))
            if ep['status']=='ok':episodes.append(ep)
            else:failures+=1
            per_seed.append(dict(scene=sid,seed=seed,status=ep['status'], samples=len(ep.get('ready',[])),
                ready_Max=max(ep.get('ready',[]),default=''),workflows=len(ep.get('deadlines',[])),
                empty_observations=ep.get('empty_observations',''),error=ep.get('error','')))
        status='ok' if seeds and len(episodes)==len(seeds) else 'incomplete'
        rp=percentiles([v for e in episodes for v in e['ready']]) if status=='ok' else {k:'' for k in ('P50','P95','P99','Max')}
        tp=percentiles([v for e in episodes for v in e['deadlines']]) if status=='ok' else {k:'' for k in ('P50','P95','P99','Max')}
        rr=max(1,math.ceil(rp['P99'])) if status=='ok' else ''
        tr=tp['P95'] if status=='ok' else ''
        samples=sum(len(e['ready']) for e in episodes);workflows=sum(len(e['deadlines']) for e in episodes)
        ready_rows.append(dict(scene=sid,seeds=','.join(map(str,seeds)),samples=samples,**rp,R_ref=rr,status=status))
        deadline_rows.append(dict(scene=sid,seeds=','.join(map(str,seeds)),workflows=workflows,**tp,T_ref=tr,status=status))
        summary.append(dict(scene=sid,task='S',resource=scenario.resource,deadline=scenario.deadline,seed_count=len(seeds),
            ready_sample_count=samples,workflow_count=workflows,**{'ready_'+k:v for k,v in rp.items()},
            **{'deadline_'+k:v for k,v in tp.items()},R_ref=rr,T_ref=tr,status=status))
        if status=='ok':
            scales=dict(R_ref=rr,T_ref=tr)
            for key in episodes[0]['physical']:
                scales[key]=percentiles([v for e in episodes for v in e['physical'][key]])['P95']
            for key in ('vm_pc_ref','vm_bw_ref','host_pc_ref'):scales[key]=max(e[key] for e in episodes)
            if any(not np.isfinite(v) or v<=0 for v in scales.values()):raise ValueError('Nonpositive calibration reference '+sid)
            configs[sid]=scales
    for name,rows in [('ready_percentiles_9scenarios.csv',ready_rows),('deadline_percentiles_9scenarios.csv',deadline_rows),
                      ('f18_normalization_summary_9scenarios.csv',summary),('f18_calibration_per_seed.csv',per_seed)]:
        if rows:write_csv(out/name,rows)
    config=dict(normalization_version=VERSION,units=SCALE_UNITS,scenes=configs,
                generation=dict(**binding, code_fingerprint=manifest.get('code_fingerprint'),
                    source_scenes=manifest.get('scenes', {}), percentile_method='linear', R_ref='ceil(ready P99), minimum 1',T_ref='relative workflow deadline P95',
                    physical='task-weighted P95; VM/host capacity maxima', decision_weighting='equal distinct decision timestamps',
                    rank_definition='Workflow.compute_upward_ranks / compute_downward_ranks (mean PC/BW)',
                    calibration_seeds={s:v['seeds'] for s,v in manifest.get('scenes',{}).items()}))
    (out/'f18_shared_normalization_v1.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    manifest['successful_episodes']=sum(r['status']=='ok' for r in per_seed)
    manifest['failed_or_missing_episodes']=failures
    manifest['failed_episodes']=sum(r['status']=='failed' for r in per_seed)
    manifest['missing_episodes']=sum(r['status']=='missing' for r in per_seed)
    manifest['sample_counts']=summary
    manifest['result_files']=[str(p.resolve()) for p in out.glob('*') if p.is_file()]
    (out/'f18_calibration_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return failures


def code_fingerprint():
    files=[Path(__file__), ROOT/'environments/mahdrl_heft.py', ROOT/'environments/mahdrl_fcfs.py',
           ROOT/'common/workflow_opt.py', ROOT/'common/read_xml_opt_Tsize.py', ROOT/'common/resource_opt.py',
           ROOT/'common/precompute_deadlines.py', ROOT/'common/scenarios.py', ROOT/'common/ta_weights.py',
           ROOT/'common/f18_state_normalization.py', ROOT/'common/experiment_protocol.py',
           ROOT/'algorithms/ertf_fifs/evaluate.py', *sorted((ROOT/'data/dax').glob('*.xml'))]
    digest=hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(ROOT)).replace('\\','/').encode())
        digest.update(b'\0');digest.update(file.read_bytes())
    return digest.hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--all-nine',action='store_true');p.add_argument('--task',choices=['S'],default='S')
    p.add_argument('--resource',choices=['S','M','L'],default='S');p.add_argument('--deadline',choices=['Loose','Medium','Tight'],default='Tight')
    p.add_argument('--seeds',default='101-130');p.add_argument('--out',type=Path,default=ROOT/'outputs/f18_normalization_calibration')
    p.add_argument('--reference-policy',choices=['ERTF-FIFS'],default='ERTF-FIFS')
    for flag in ('dry-run','resume','summary-only','raw-records'):p.add_argument('--'+flag,action='store_true')
    p.add_argument('--experiment-protocol', default=str(DEFAULT_PROTOCOL))
    args=p.parse_args(argv)
    binding=protocol_binding(args.experiment_protocol)
    if args.summary_only:
        manifest=json.loads((args.out/'f18_calibration_manifest.json').read_text(encoding='utf-8'))
        return int(summarize(args.out,manifest,args.experiment_protocol)>0)
    seeds=parse_seeds(args.seeds);validate_seeds(seeds,args.experiment_protocol)
    scenes=nine_scenarios(args.experiment_protocol) if args.all_nine else [Scenario(task='S',resource=args.resource,deadline=args.deadline,mode='train',episodes=ExperimentProtocol.load(args.experiment_protocol).episodes,experiment_protocol=args.experiment_protocol)]
    if args.dry_run:
        print(json.dumps(dict(**binding,seeds=seeds,scenes=[s.describe() for s in scenes],reference=args.reference_policy,
            missing_caches=[str(s.cache_path) for s in scenes if not s.cache_path.exists()]),indent=2));return 0
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'episodes').mkdir(exist_ok=True)
    mp=args.out/'f18_calibration_manifest.json'
    if mp.exists() and not args.resume:raise FileExistsError('Use --resume or a new --out directory')
    manifest=json.loads(mp.read_text(encoding='utf-8')) if mp.exists() else dict(
        **binding,normalization_version=VERSION,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        date=datetime.now(timezone.utc).isoformat(),reference_policy=args.reference_policy,
        workflows_per_episode=50,percentile_method='linear',time_unit='s',
        alpha_rule=f'sample_deadline_alpha: seed,wf_id; offset={DEADLINE_ALPHA_SEED_OFFSET}',scenes={})
    validate_manifest(manifest,args.experiment_protocol)
    fingerprint=code_fingerprint()
    if manifest.get('code_fingerprint') not in (None, fingerprint):
        raise ValueError('Calibration code/topology changed; use a new --out directory')
    manifest['code_fingerprint']=fingerprint
    errors=0
    for scenario in scenes:
        sid=scene_id('S',scenario.resource,scenario.deadline)
        try:
            cache=cache_info(scenario,seeds)
            spec=json.loads(json.dumps(dict(seeds=list(seeds),scenario=scenario.describe(),cache=cache)))
            old=manifest['scenes'].get(sid)
            if old and 'cache' not in old:old=None
            if old:old={k:v for k,v in old.items() if k!='error'}
            if old and old!=spec:raise ValueError('Resume scenario/seeds/cache mismatch')
            manifest['scenes'][sid]=spec
            for seed in seeds:
                path=args.out/'episodes'/f'{sid}_{seed}.json'
                if args.resume and path.exists() and json.loads(path.read_text())['status']=='ok':continue
                try:result=sample_episode(scenario,seed,args.raw_records)
                except Exception as exc:result=dict(status='failed',seed=seed,error=f'{type(exc).__name__}: {exc}');errors+=1
                path.write_text(json.dumps(result),encoding='utf-8')
                print(sid,seed,result['status'],flush=True)
        except Exception as exc:
            errors+=1
            print(sid,str(exc),file=sys.stderr)
            manifest['scenes'].setdefault(sid,dict(seeds=list(seeds),scenario=scenario.describe()))['error']=str(exc)
        mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    failures=summarize(args.out,manifest,args.experiment_protocol)
    return int(errors>0 or failures>0)


if __name__=='__main__':raise SystemExit(main())