"""Training/evaluation/checkpoint orchestration, with no private physical model."""
from dataclasses import replace
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import numpy as np
import torch
from common.f18_state_normalization import SHARED_VERSIONS,check_checkpoint
from common.f11 import protect_training_outputs
from common.experiment_measurements import TrainingRecorder, EvaluationRecorder, collect_validation, episode_metrics
from common.scenarios import ROOT
from .scenario import parse_scenario
from common.r010_paths import training_name, evaluation_stem
from .config import ALGORITHM,DISPLAY_NAME,MODEL_VERSION,IMPLEMENTATION,BEST_CHECKPOINT,FINAL_CHECKPOINT,REWARD_PROTOCOL,COMMON_REWARD,PaperConfig,SCHEDULING_PROTOCOL
from .environment import PaperCloudEnv
from .learner import PaperLearner
from .selection import SELECTION_TAG, WORKFLOW_SUCCESS_THRESHOLD, selection_metadata, selection_protocol, validation_selection, selection_key


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def queue_settings(scenario):
    options=scenario.paper_options or {}
    return dict(queue_capacity=options.get('queue_capacity',10),slot_duration=options.get('slot_duration',5.))


def learning_settings(scenario):
    options=scenario.paper_options or {}
    return dict(rollout_size=options.get('rollout_size',64),precision=options.get('precision','float32'))


def env_kwargs(scenario,seed):
    return dict(dax_paths=scenario.dax_paths(),horizon=1e9,arrival_lambda=scenario.arrival_lambda,random_seed=seed,
        workflows_per_episode=50,num_hosts=scenario.num_hosts,vms_per_host=scenario.vms_per_host,
        pc_tiers=(1.,2.,4.,6.,8.),bw_tiers=(1000.,2000.,4000.,6000.,8000.),
        deadline_mode='cache_fcfs',deadline_cache_path=str(scenario.cache_path),deadline_cache_strict=True,
        deadline_alpha_choices=list(scenario.deadline_choices),deadline_alpha_probs=list(scenario.deadline_probs),
        max_ready_tasks='auto',**queue_settings(scenario),**COMMON_REWARD,**scenario.normalization_kwargs())


def output_dir(scenario,kind):
    if scenario.protocol() is None:raise ValueError('Formal paper IRWS requires the explicit disjoint common protocol')
    if scenario.normalization_mode not in SHARED_VERSIONS:raise ValueError('Formal IRWS must use the existing shared configuration')
    root=ROOT/'outputs'/kind/ALGORITHM
    return root if kind=='run' else root/run_name(scenario)


def run_name(scenario):
    return training_name(scenario,MODEL_VERSION,dict(protocol=selection_protocol(scenario),
        reward=COMMON_REWARD,implementation=IMPLEMENTATION,queue=queue_settings(scenario),
        learning=learning_settings(scenario)))


def evaluation_directory(scenario):
    # Keep detailed measurement paths short; the downloadable filenames carry the scene.
    key=hashlib.sha256(artifact_name(scenario,'eval').encode()).hexdigest()[:16]
    return output_dir(scenario,'run')/'_details'/key


def artifact_name(scenario,kind):
    if kind=='train':return 'train.jsonl'
    stem=evaluation_stem(scenario,ALGORITHM,run_name(scenario))
    return stem+'.csv' if kind=='eval' else stem.replace('eval_','summary_',1)+'.json'


def run_episode(env,learner,seed,training=False,evaluation=None):
    if getattr(env,'scheduling_protocol',None)!=SCHEDULING_PROTOCOL:
        raise ValueError('IDLE scheduling protocol mismatch')
    if any(getattr(env,k)!=v for k,v in COMMON_REWARD.items()):
        raise ValueError('IRWS-R-FE_IDLE requires the configured energy-tardiness reward')
    learner.numerics.reset();env.reset(seed=seed);learner.reset_history()
    updates=[];weights=[];boundaries=0;external_return=0.;physical_cost=0.
    measurement=EvaluationRecorder(evaluation[0],ALGORITHM,env,{DISPLAY_NAME:learner},evaluation[1],
        checkpoint_directory=str(evaluation[0].checkpoint_override or output_dir(evaluation[0],'ckpts'))) if evaluation else None
    decision=learner.prepare(env,deterministic=not training) if not env.done_flag else None
    while decision is not None:
        action,logp,value=learner.act(decision,deterministic=not training)
        reward,_,info=env.paper_step(action)
        external_return+=reward;physical_cost+=info['outer_cost']
        weights.append(info['fusion_weight']);boundaries+=int(info['fusion_boundary'])
        # Scheduling ticks and frozen-policy rollout windows are independent clocks.
        if not env.can_dispatch():env.finish_phase_and_advance()
        done=env.done_flag
        if training and not done and len(learner.buffer)+1>=learner.cfg.rollout_size:
            # Estimate the boundary without selecting a task. Algorithm 1's next
            # interval ordering must use the newly updated embedding network.
            boundary=learner.boundary_snapshot(env)
            learner.remember(decision,action,logp,value,reward,False,info,None,boundary=boundary)
            updates.append(learner.update(boundary.value))
            successor=learner.prepare(env)
            learner.resolve_boundary(successor)
        else:
            successor=None if done else learner.prepare(env,deterministic=not training)
            if training:
                learner.remember(decision,action,logp,value,reward,done,info,successor)
                if done:updates.append(learner.update(0.))
        decision=successor
        if len(weights)>100000:raise RuntimeError('Episode exceeded the decision guard')
    if measurement is not None:measurement.finish()
    metrics=env.get_episode_metrics()
    if env.completed_workflows!=env.workflows_per_episode or any(state!='Finished' for state in env.task_state):
        raise RuntimeError('Queue episode ended before actual completion')
    if not np.isfinite(list(metrics.values())).all():raise FloatingPointError('Nonfinite physical metrics')
    return dict(metrics=metrics,objective=dict(protocol=REWARD_PROTOCOL,settings=COMMON_REWARD,
        external_return=float(external_return),physical_cost=float(physical_cost)),updates=updates,
        implementation=dict(IMPLEMENTATION),numerics=learner.numerics.snapshot(),assignments=len(weights),queue=dict(capacity=env.queue_capacity,
        slot_duration=env.slot_duration,interval_index=env.interval_index,
        scheduling_protocol=SCHEDULING_PROTOCOL,waiting_enabled=False),fusion=dict(
        minimum=min(weights,default=0.),maximum=max(weights,default=0.),mean=float(np.mean(weights)) if weights else 0.,
        outside_unit_interval=sum(w<0 or w>1 for w in weights),zero_rank_boundaries=boundaries))


def git_commit():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def save_checkpoint(path,learner,scenario,episode,candidate,best,validation,environment_config,role):
    if learner.buffer or learner._pending_boundary is not None:
        raise RuntimeError('Checkpoints are written only at completed episode/rollout boundaries')
    if role not in ('best','final') or validation_selection(validation)!=candidate:
        raise ValueError('Invalid checkpoint role or validation selection')
    modules={n:getattr(learner,n).state_dict() for n in ('task','actor','critic','intrinsic')}
    optimizers={n:getattr(learner,n+'_opt').state_dict() for n in ('task','actor','critic','intrinsic')}
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
             cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    obj=dict(algorithm=ALGORITHM,display_name=DISPLAY_NAME,model_version=MODEL_VERSION,implementation=dict(IMPLEMENTATION),model_seed=scenario.model_seed,reward_protocol=REWARD_PROTOCOL,reward_settings=COMMON_REWARD,modules=modules,optimizers=optimizers,paper_config=learner.cfg.as_dict(),
        episode=episode,steps=learner.steps,updates=learner.updates,
        checkpoint_selection=selection_metadata(),scheduling_protocol=SCHEDULING_PROTOCOL,validation_selection=candidate,
        best_validation_selection=best,validation=validation,role=role,
        normalization=scenario.normalization_metadata(),protocol=selection_protocol(scenario),
        environment=environment_config,scenario=scenario.describe(),rng=rng,git_commit=git_commit(),
        recurrent_hidden=learner.hidden.detach().cpu(),vm_count=learner.vm_count,numerical_precision=learner.cfg.precision)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save(obj,path)


def load_checkpoint(path,scenario,device='cpu',restore_training=False):
    # Locally produced training files contain RNG/optimizer state, not weights alone.
    obj=torch.load(path,map_location=device,weights_only=False)
    if obj.get('algorithm')!=ALGORITHM or obj.get('model_version')!=MODEL_VERSION:
        raise ValueError('IRWS-R-FE_IDLE checkpoint algorithm/version mismatch; train with algorithms/irws_r_fe_idle/train.py')
    if obj.get('implementation')!=IMPLEMENTATION:
        raise ValueError('IRWS-R checkpoint implementation settings mismatch')
    if obj.get('scheduling_protocol')!=SCHEDULING_PROTOCOL:
        raise ValueError('Checkpoint scheduling protocol mismatch')
    if obj.get('model_seed')!=scenario.model_seed:raise ValueError('IRWS-R checkpoint model seed mismatch')
    if obj.get('reward_protocol')!=REWARD_PROTOCOL or obj.get('reward_settings')!=COMMON_REWARD:
        raise ValueError('Checkpoint reward objective mismatch; retrain adapted IRWS')
    if {k:obj.get('environment',{}).get(k) for k in queue_settings(scenario)}!=queue_settings(scenario):
        raise ValueError('Checkpoint queue/interval settings mismatch')
    expected=scenario.normalization_metadata()
    # Before F22, paper checkpoints stored the protocol separately from normalization.
    if scenario.protocol().version != 'f22_split_v1' and not (obj.get('normalization') or {}).get('experiment_protocol_hash'):
        expected.pop('experiment_protocol_hash',None)
    check_checkpoint(obj.get('normalization'),expected)
    if obj.get('protocol')!=selection_protocol(scenario):raise ValueError('Checkpoint experiment protocol mismatch')
    if obj.get('checkpoint_selection')!=selection_metadata():raise ValueError('Checkpoint selection rule mismatch')
    expected_role = 'final' if restore_training else 'best'
    if obj.get('role')!=expected_role:raise ValueError('Checkpoint role mismatch: expected '+expected_role)
    rows = obj.get('validation', [])
    if [r['seed'] for r in rows] != list(scenario.protocol().validation):
        raise ValueError('Checkpoint validation seeds mismatch')
    if validation_selection(rows)!=obj.get('validation_selection'):
        raise ValueError('Checkpoint validation metrics mismatch')
    best = obj.get('best_validation_selection', {})
    best_metrics=validation_selection([dict(wf_count=best.get('workflow_count',0),
        wf_success_count=best.get('workflow_success_count',0),
        wf_success_rate=best.get('workflow_success_rate',float('nan')),
        total_energy=best.get('mean_total_energy',float('nan')))])
    best_metrics['validation_instances']=len(rows)
    if (not isinstance(best.get('episode'), int) or not 0 <= best['episode'] <= obj['episode']
            or best.get('completed_episodes') != best['episode']+1
            or set(best) != set(obj['validation_selection']) | {'episode', 'completed_episodes'}
            or any(best.get(key)!=value for key,value in best_metrics.items())
            or selection_key(best) > selection_key(obj['validation_selection'])):
        raise ValueError('Checkpoint best selection mismatch')
    if expected_role=='best' and best!=dict(obj['validation_selection'],episode=obj['episode'],completed_episodes=obj['episode']+1):
        raise ValueError('Best checkpoint selection mismatch')
    cfg=dict(obj['paper_config']);cfg['device']=device
    if any(cfg.get(k)!=v for k,v in learning_settings(scenario).items()) or obj.get('numerical_precision')!=cfg['precision']:
        raise ValueError('Checkpoint rollout/precision settings mismatch')
    for k in ('actor_hidden','critic_hidden'):cfg[k]=tuple(cfg[k])
    learner=PaperLearner(obj['vm_count'],PaperConfig(**cfg))
    if learner.vm_count!=sum(scenario.vms_per_host):raise ValueError('Source/target VM dimensions are incompatible')
    for name,weights in obj['modules'].items():getattr(learner,name).load_state_dict(weights)
    learner.steps=obj['steps'];learner.updates=obj['updates']
    if restore_training:
        for name,state in obj['optimizers'].items():getattr(learner,name+'_opt').load_state_dict(state)
        random.setstate(obj['rng']['python']);np.random.set_state(obj['rng']['numpy']);torch.set_rng_state(obj['rng']['torch'].cpu())
        if torch.cuda.is_available() and obj['rng']['cuda'] is not None:torch.cuda.set_rng_state_all(obj['rng']['cuda'])
    return learner,obj


@contextmanager
def isolated_evaluation(learner):
    """Validation must not advance training RNG, RNN history, caches or diagnostics."""
    if learner.buffer or learner._pending_boundary is not None:
        raise RuntimeError('Evaluate only after completing the training rollout')
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    modules = {m: m.training for name in ('task','actor','critic','intrinsic')
               for m in getattr(learner,name).modules()}
    names = ('hidden','steps','updates','_embedding_cache','_resource_tensors','_resource_revision',
             '_resource_mask','_device_mask')
    state = {name:getattr(learner,name) for name in names if hasattr(learner,name)}
    graph = learner.task.embedding
    graphs, bundles = graph._graphs, graph._bundles
    diagnostics = {name:getattr(learner.numerics,name) for name in ('profiles','clipping','_pending')}
    try:
        for module in modules:module.training=False
        with torch.no_grad():yield
    finally:
        for module,mode in modules.items():module.training=mode
        for name in names:
            if name in state:setattr(learner,name,state[name])
            elif hasattr(learner,name):delattr(learner,name)
        graph._graphs,graph._bundles=graphs,bundles
        for name,value in diagnostics.items():setattr(learner.numerics,name,value)
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        if rng[3] is not None:torch.cuda.set_rng_state_all(rng[3])


def evaluate(learner,scenario,seeds,split='test',kwargs=None,on_result=None):
    protocol=scenario.protocol()
    allowed=protocol.validation if split=='validation' else protocol.test
    seeds=list(seeds)
    if (split not in ('validation','test') or not seeds or len(set(seeds))!=len(seeds)
            or not set(seeds)<=set(allowed) or set(seeds)&set(protocol.train)
            or (split!='validation' and set(seeds)&set(protocol.validation))):
        raise ValueError('Evaluation attempted to cross the frozen data partition')
    rows=[]
    with isolated_evaluation(learner):
        for seed in seeds:
            env=PaperCloudEnv(**(dict(kwargs,random_seed=seed) if kwargs is not None else env_kwargs(scenario,seed)))
            try:
                directory=evaluation_directory(scenario)/'measurements'
                result=run_episode(env,learner,seed,False,evaluation=(scenario,directory) if split=='test' else None)
                counts=episode_metrics(env)
                row=dict(seed=seed,**result['metrics'])
                row.update({key:counts[key] for key in ('wf_count','wf_completed_count','wf_success_count','wf_success_rate')})
                if split=='validation':collect_validation(env)
                rows.append(row)
                if on_result is not None:on_result(row)
            finally:env.close()
    names=[k for k in rows[0] if k!='seed']
    summary={k:dict(mean=float(np.mean([r[k] for r in rows])),std=float(np.std([r[k] for r in rows],ddof=1)) if len(rows)>1 else 0.) for k in names}
    return rows,summary


def train(scenario,config=None,kwargs=None,out=None,resume=None):
    protocol=scenario.protocol()
    if protocol is None or not 1<=scenario.episodes<=protocol.episodes:raise ValueError('A valid fixed experiment budget is required')
    if scenario.normalization_mode not in SHARED_VERSIONS:raise ValueError('Paper training requires f18_shared_v1')
    directory=Path(out) if out is not None else (Path(resume).resolve().parent if resume else output_dir(scenario,'ckpts'))
    if resume:
        if Path(resume).name != FINAL_CHECKPOINT:raise ValueError(f'Resume from {FINAL_CHECKPOINT} to preserve the latest run state')
        if directory.resolve()!=Path(resume).resolve().parent:
            raise ValueError('Resume must continue its own output directory')
    else:protect_training_outputs(directory,directory/'logs')
    kwargs=dict(kwargs or env_kwargs(scenario,protocol.train[0]))
    for key,value in queue_settings(scenario).items():
        if key in kwargs and kwargs[key]!=value:raise ValueError('Scenario/environment queue mismatch')
        kwargs[key]=value
    env=PaperCloudEnv(**kwargs)
    scenario.normalization_metadata();seed_everything(scenario.model_seed)
    cfg=config or PaperConfig(device=(scenario.paper_options or {}).get('device','cpu'),**learning_settings(scenario))
    if any(getattr(cfg,k)!=v for k,v in learning_settings(scenario).items()):
        raise ValueError('Scenario/learner rollout or precision mismatch')
    start=0;best=None
    if resume:
        if not (directory/BEST_CHECKPOINT).is_file():
            raise ValueError('Resume requires its historical best checkpoint')
        checked_best,best_obj=load_checkpoint(directory/BEST_CHECKPOINT,scenario,cfg.device)
        del checked_best
        learner,obj=load_checkpoint(resume,scenario,cfg.device,True)
        start=obj['episode']+1;best=obj['best_validation_selection']
        if best_obj['best_validation_selection']!=best:
            raise ValueError('Resume best checkpoint does not match final selection history')
        if start>scenario.episodes:raise ValueError('Resume checkpoint exceeds the requested training budget')
    else:learner=PaperLearner(env.num_vms,cfg)
    from common.f22_instances import write_run_metadata
    write_run_metadata(directory,scenario)
    directory.mkdir(parents=True,exist_ok=True);log=directory/artifact_name(scenario,'train')
    print(f'[{DISPLAY_NAME} train] algorithm={ALGORITHM} version={MODEL_VERSION} task={scenario.task} resource={scenario.resource} deadline={scenario.deadline} '
          f'model_seed={scenario.model_seed} episodes={scenario.episodes} start_episode={start}',flush=True)
    print(f'训练日志: {log.resolve()}',flush=True)
    print(f'模型保存目录: {directory.resolve()}',flush=True)
    print(f'调度: {SCHEDULING_PROTOCOL}; slot_duration={queue_settings(scenario)["slot_duration"]}; VM waiting disabled',flush=True)
    print(f'选模规则: {SELECTION_TAG}; workflow success >= {WORKFLOW_SUCCESS_THRESHOLD}, then minimum validation energy',flush=True)
    recorder=TrainingRecorder(directory/'logs',scenario,ALGORITHM,env,{DISPLAY_NAME:learner})
    recorder.index=start
    try:
        for episode in range(start,scenario.episodes):
            seed=protocol.train[episode]
            result=run_episode(env,learner,seed,True)
            rows,validation=recorder.validation(evaluate,learner,scenario,protocol.validation,'validation',kwargs)
            candidate=validation_selection(rows)
            selected=best is None or selection_key(candidate)<selection_key(best)
            if selected:
                best=dict(candidate,episode=episode,completed_episodes=episode+1)
                recorder.checkpoint(save_checkpoint)(directory/BEST_CHECKPOINT,learner,scenario,episode,candidate,best,rows,kwargs,'best')
            recorder.checkpoint(save_checkpoint)(directory/FINAL_CHECKPOINT,learner,scenario,episode,candidate,best,rows,kwargs,'final')
            from common.f22_instances import instance_identity
            result.update(algorithm=ALGORITHM,display_name=DISPLAY_NAME,episode=episode,seed=seed,validation=validation,model_version=MODEL_VERSION,
                          instance=instance_identity(env,scenario,seed),checkpoint_selection=selection_metadata(),
                          validation_selection=candidate,best_validation_selection=best,selected_as_best=selected)
            with log.open('a',encoding='utf-8') as stream:stream.write(json.dumps(result,allow_nan=False)+'\n')
            record=dict(episode=episode,seed=seed,validation=rows,checkpoint_selection=selection_metadata(),
                        validation_selection=candidate,best_validation_selection=best,selected_as_best=selected)
            with (directory/'validation_selection.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n')
            print(f'{DISPLAY_NAME} episode={episode} seed={seed} energy={result["metrics"]["total_energy"]:.6f} '
                  f'validation_energy={candidate["mean_total_energy"]:.6f} '
                  f'validation_success={candidate["workflow_success_count"]}/{candidate["workflow_count"]} '
                  f'feasible={candidate["feasible"]} selected={selected} best_episode={best["episode"]}',flush=True)
    finally:
        recorder.close();env.close()
    summary=dict(algorithm=ALGORITHM,model_version=MODEL_VERSION,checkpoint_selection=selection_metadata(),scheduling_protocol=SCHEDULING_PROTOCOL,
                 protocol=selection_protocol(scenario),completed_episodes=scenario.episodes,
                 best_validation_selection=best,feasible_model_found=best['feasible'],
                 best_checkpoint=str(directory/BEST_CHECKPOINT),final_checkpoint=str(directory/FINAL_CHECKPOINT))
    (directory/'training_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(f'[{DISPLAY_NAME} train completed] 训练完成：{scenario.episodes}/{scenario.episodes} episodes | '
          f'task={scenario.task} resource={scenario.resource} deadline={scenario.deadline}',flush=True)
    print(f'最佳模型已保存到: {(directory/BEST_CHECKPOINT).resolve()}',flush=True)
    print(f'最终模型已保存到: {(directory/FINAL_CHECKPOINT).resolve()}',flush=True)
    print(f'训练日志已保存到: {log.resolve()}',flush=True)
    print(f'训练统计目录: {(directory/"logs").resolve()}',flush=True)
    print(f'feasible_model_found={best["feasible"]}; best_episode={best["episode"]}',flush=True)
    return learner,directory


def train_main():
    scenario=parse_scenario('train',ALGORITHM)
    train(scenario,resume=(scenario.paper_options or {}).get('resume_checkpoint'))


def evaluate_main():
    scenario=parse_scenario('eval',ALGORITHM);device=(scenario.paper_options or {}).get('device','cpu')
    directory=Path(scenario.checkpoint_override) if scenario.checkpoint_override else output_dir(scenario,'ckpts')
    root=output_dir(scenario,'run')
    out=evaluation_directory(scenario)
    csv_path=root/artifact_name(scenario,'eval')
    summary_path=root/artifact_name(scenario,'summary')
    for path in (csv_path,summary_path):
        if path.exists():raise FileExistsError(f'Refusing to overwrite evaluation artifact: {path}')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {DISPLAY_NAME} evaluation artifacts: {out}')
    learner,checkpoint=load_checkpoint(directory/BEST_CHECKPOINT,scenario,device)
    out.mkdir(parents=True,exist_ok=True)
    print(f'[{DISPLAY_NAME} eval] algorithm={ALGORITHM} version={MODEL_VERSION} task={scenario.task} resource={scenario.resource} deadline={scenario.deadline} '
          f'model_seed={scenario.model_seed} seeds={scenario.seed_spec}',flush=True)
    print(f'加载模型: {(directory/BEST_CHECKPOINT).resolve()}',flush=True)
    print(f'选中回合: {checkpoint["episode"]}; validation={checkpoint["validation_selection"]}',flush=True)
    print(f'评测 CSV: {csv_path.resolve()}',flush=True)
    with csv_path.open('x',encoding='utf-8',newline='') as f:
        writer=None
        completed=0
        def save_seed(row):
            nonlocal writer,completed
            row=dict(algorithm=ALGORITHM,display_name=DISPLAY_NAME,model_version=MODEL_VERSION,model_seed=scenario.model_seed,**row)
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
            writer.writerow(row);f.flush()
            completed+=1
            print(f'[{DISPLAY_NAME} eval {completed}/{len(scenario.seeds)}] task={scenario.task} resource={scenario.resource} deadline={scenario.deadline} seed={row["seed"]} energy={row["total_energy"]:.6f} saved',flush=True)
        rows,summary=evaluate(learner,scenario,scenario.seeds,on_result=save_seed)
    report=dict(algorithm=ALGORITHM,display_name=DISPLAY_NAME,model_version=MODEL_VERSION,implementation=dict(IMPLEMENTATION),model_seed=scenario.model_seed,summary=summary,normalization=scenario.normalization_metadata(),
        protocol=selection_protocol(scenario),reward_protocol=REWARD_PROTOCOL,
        checkpoint_selection=selection_metadata(),scheduling_protocol=SCHEDULING_PROTOCOL,selected_checkpoint_episode=checkpoint['episode'],
        selected_checkpoint_validation=checkpoint['validation_selection'],
        feasible_model_found=checkpoint['validation_selection']['feasible'],
        reward_settings=COMMON_REWARD,source_checkpoint=str(directory/BEST_CHECKPOINT),
        evaluation_csv=str(csv_path),evaluation_details=str(out))
    with summary_path.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(f'[{DISPLAY_NAME} eval completed] 评测完成：{len(rows)}/{len(scenario.seeds)} seeds | '
          f'task={scenario.task} resource={scenario.resource} deadline={scenario.deadline}',flush=True)
    print(f'平均能耗: {summary["total_energy"]["mean"]:.6f} J',flush=True)
    print(f'评测 CSV 已保存到: {csv_path.resolve()}',flush=True)
    print(f'汇总 JSON 已保存到: {summary_path.resolve()}',flush=True)
    print(f'逐种子明细目录: {(out/"measurements").resolve()}',flush=True)
