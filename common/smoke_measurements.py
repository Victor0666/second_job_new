"""Temporary small-network integration of all train runners; no formal outputs."""
from contextlib import ExitStack,redirect_stdout
from dataclasses import replace
import importlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch
from common.scenarios import Scenario
from common.test_f18 import fixture_config
from common.precompute_deadlines import main as precompute
from common.experiment_measurements import TrainingRecorder, EvaluationRecorder, episode_metrics


def run(training_metrics=True):
    torch.set_num_threads(1)
    report={}
    with tempfile.TemporaryDirectory(prefix='measurement_train_smoke_') as directory:
        root=Path(directory);scales=root/'scales.json';fixture_config(scales)
        class LocalScenario(Scenario):
            @property
            def cache_path(self):return root/'cache.json'
            def model_name(self,algorithm):return str(root/'logs'/super().model_name(algorithm))
            def checkpoint_dir(self,algorithm):return root/'models'/super().model_name(algorithm)
            def log_dir(self,algorithm):return root/'logs'/super().model_name(algorithm)
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode='precompute',seed_spec='0,600,700',cache_output=str(root/'cache.json')))
        names=('mahdrl','pd3qn','marl','ablation_hostrand','ablation_mgrrand','ablation_vmrand','ablation_nohost','ablation_tl2','irws')
        for name in names:
            entry='ablation_nohost' if name=='ablation_tl2' else name
            module=importlib.import_module('algorithms.'+entry+'.train')
            scene=LocalScenario(mode='train',episodes=1,normalization_config=str(scales))
            env_name={'pd3qn':'CloudWorkflowEnv_PD3QN_Cost','marl':'CloudWorkflowEnv_MARL','irws':'IRWSEnvRouteA'}.get(entry,'CloudWorkflowEnv_VMAgents')
            original_env=getattr(module,env_name)
            if name=='ablation_tl2':
                from environments.ablation_tl2 import CloudWorkflowEnv_VMAgents
                original_env=CloudWorkflowEnv_VMAgents
            def env_factory(*args,_original=original_env,**kwargs):
                kwargs['workflows_per_episode']=2
                return _original(*args,**kwargs)
            captured=[];constructor_kwargs=[]
            original_factory=env_factory
            def env_factory(*args,**kwargs):
                constructor_kwargs.append(dict(kwargs))
                return original_factory(*args,**kwargs)
            def capture_recorder(*args,**kwargs):
                with patch.dict('os.environ',{'EXPERIMENT_METRICS':'1' if training_metrics else '0'}):
                    recorder=TrainingRecorder(*args,**kwargs)
                captured.append(recorder);return recorder
            with ExitStack() as stack,(root/(name+'.log')).open('w',encoding='utf-8') as log,redirect_stdout(log):
                stack.enter_context(patch.object(module,'SCENARIO',scene))
                stack.enter_context(patch.object(module,'TrainingRecorder',side_effect=capture_recorder))
                stack.enter_context(patch.object(module,env_name,side_effect=env_factory))
                if hasattr(module,'D3QNAgent'):
                    original=module.D3QNAgent
                    def dqn(*args,_original=original,**kwargs):
                        kwargs.update(hidden_dims=(16,8),batch_size=4,buffer_size=32,device=torch.device('cpu'))
                        return _original(*args,**kwargs)
                    stack.enter_context(patch.object(module,'D3QNAgent',side_effect=dqn))
                elif entry=='marl':
                    original=module.QConfig
                    def cfg(*args,**kwargs):
                        kwargs.update(hidden_dims=(16,8),device='cpu');return original(*args,**kwargs)
                    stack.enter_context(patch.object(module,'QConfig',side_effect=cfg))
                elif entry=='irws':
                    original=module.PPOVMAgent
                    def ppo(cfg,hidden):return original(cfg,hidden=(16,8))
                    stack.enter_context(patch.object(module,'PPOVMAgent',side_effect=ppo))
                if entry=='ablation_nohost':module.train(scene,algorithm=name,env_cls=env_factory)
                else:module.train()
            ev=importlib.import_module('algorithms.'+entry+'.evaluate')
            eval_scene=replace(scene,mode='eval',seed_spec='700')
            agents=captured[0].agents
            kw=constructor_kwargs[0];kw['random_seed']=700
            for key in ('energy_reward_scale','task_baseline_norm','energy_norm_per_mi_ref','alpha_delay_host','alpha_delay_vm'):
                if hasattr(captured[0].env,key):kw[key]=getattr(captured[0].env,key)
            if name in ('ablation_nohost','ablation_tl2'):kw.pop('alpha_delay_host',None)
            def capture_evaluation(*args,**kwargs):
                kwargs['directory']=root/'evaluation'/name
                return EvaluationRecorder(*args,**kwargs)
            with patch.object(ev,'SCENARIO',eval_scene),patch.object(ev,'EvaluationRecorder',side_effect=capture_evaluation),redirect_stdout(io.StringIO()):
                if name=='mahdrl':ev.evaluate_hrl_mix_one_seed(env_factory,kw,agents['VA'],agents['HA'],agents['TA'])
                elif name=='pd3qn':ev.evaluate_one_episode(env_factory,kw,agents['PD3QN'])
                elif name=='marl':
                    with patch.object(ev,'CloudWorkflowEnv_MARL',side_effect=env_factory):
                        locals = object.__new__(ev.IndependentLocalAgents)
                        locals.agents = [agents[f'Local_{i}'] for i in range(scene.num_hosts)]
                        locals.num_hosts = scene.num_hosts
                        ev.evaluate_one_seed(kw,agents['Global'],locals,700,0)
                elif name=='ablation_hostrand':ev.evaluate_hostrand_one_seed(env_factory,kw,agents['VA'],agents['TA'])
                elif name=='ablation_mgrrand':ev.evaluate_mgrrand_one_seed(env_factory,kw,agents['VA'],agents['HA'])
                elif name=='ablation_vmrand':ev.evaluate_hrl_vm_random_one_seed(env_factory,kw,agents['HA'],agents['TA'])
                elif name in ('ablation_nohost','ablation_tl2'):ev.evaluate_nohost_one_seed(env_factory,kw,agents['VA'],agents['TA'])
                elif name=='irws':
                    with patch.object(ev,'IRWSEnvRouteA',side_effect=env_factory):
                        ev.evaluate_one_seed(kw,agents['Task'],agents['VM'],700)
            evaluation=json.loads((root/'evaluation'/name/'episodes/seed700.json').read_text())
            assert evaluation['timing']['allocation_count']==evaluation['metrics']['task_count'],name
            assert evaluation['metrics']['wf_completed_count']==2,name
            parameter_hashes={}
            for role,agent in agents.items():
                digest=hashlib.sha256()
                for attr,network in sorted(vars(agent).items()):
                    if isinstance(network,torch.nn.Module):
                        for key,tensor in network.state_dict().items():
                            digest.update((attr+'.'+key).encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                parameter_hashes[role]=digest.hexdigest()
            if not training_metrics:
                report[name]=dict(metrics=episode_metrics(captured[0].env),parameter_hashes=parameter_hashes,evaluation_metrics=evaluation['metrics'])
                print('Uninstrumented training control passed:',name,flush=True)
                continue
            rows=[json.loads(line) for line in (scene.log_dir(name)/'training_measurements.jsonl').read_text().splitlines()]
            assert len(rows)==1,(name,len(rows))
            record=rows[0]
            assert record['metrics']['wf_completed_count']==2,name
            assert record['assignment_calls']==record['metrics']['task_count'],name
            assert record['validation'],name
            assert sum(record['optimizer_steps'].values())>0,name
            assert all(record['optimizer_sample_presentations'][k]>0 for k,v in record['optimizer_steps'].items() if v),name
            report[name]=dict(parameter_hashes=parameter_hashes,evaluation_metrics=evaluation['metrics'],evaluation_allocation_count=evaluation['timing']['allocation_count'],assignment_calls=record['assignment_calls'],optimizer_steps=record['optimizer_steps'],optimizer_sample_presentations=record['optimizer_sample_presentations'],
                validation_instances=len(record['validation']),metrics=record['metrics'],train_seconds=record['train_seconds'],
                validation_seconds=record['validation_seconds'],checkpoint_seconds=record['checkpoint_seconds'])
            print('Training measurement smoke passed:',name,flush=True)
        from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents as HeuristicEnv
        from algorithms.mahdrl import evaluate as main_evaluation
        from algorithms.ertf_fifs import evaluate as heuristic
        eval_scene=replace(scene,mode='eval',seed_spec='700')
        with patch.object(main_evaluation,'SCENARIO',eval_scene):kw=main_evaluation.build_env_kwargs_hrl_mix(random_seed=700)
        def heuristic_env(**kwargs):
            kwargs['workflows_per_episode']=2;return HeuristicEnv(**kwargs)
        def heuristic_recorder(*args,**kwargs):
            kwargs['directory']=root/'evaluation/ertf_fifs';return EvaluationRecorder(*args,**kwargs)
        with patch.object(heuristic,'SCENARIO',eval_scene),patch.object(heuristic,'EvaluationRecorder',side_effect=heuristic_recorder),redirect_stdout(io.StringIO()):
            result=heuristic.run_one_episode(heuristic_env,kw,700)
        row=json.loads((root/'evaluation/ertf_fifs/episodes/seed700.json').read_text())
        assert row['timing']['allocation_count']==row['metrics']['task_count']
        report['ertf_fifs']=dict(metrics=row['metrics'],evaluation_allocation_count=row['timing']['allocation_count'])
        print('Heuristic measurement smoke passed: ertf_fifs',flush=True)
    return dict(verification_only=True,episodes=1,workflows_per_episode=2,small_networks=True,device='cpu',training_metrics_enabled=training_metrics,algorithms=report)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',required=True);p.add_argument('--without-training-metrics',action='store_true');args=p.parse_args()
    Path(args.out).write_text(json.dumps(run(not args.without_training_metrics),indent=2),encoding='utf-8')
