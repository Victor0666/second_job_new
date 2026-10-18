"""Run actual F11 training loops in temporary outputs using small verification-only networks."""
import argparse
from contextlib import redirect_stdout
import importlib
import json
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch

import torch
from common.scenarios import Scenario
from common.precompute_deadlines import main as precompute
from common.f11 import LEARNING_MODEL


def run_smoke(normalization_mode='legacy', normalization_config=None, irws_task='L', verify_checkpoints=False):
    progress=sys.stdout
    torch.set_num_threads(1)
    report = dict(learning_model=LEARNING_MODEL, verification_overrides=dict(hidden_dims=[16, 8],
                  dqn_batch_size=4, dqn_buffer_size=32), algorithms={})
    with tempfile.TemporaryDirectory(prefix='F11_smoke_') as directory:
        root = Path(directory)
        class SmokeScenario(Scenario):
            @property
            def cache_path(self):
                return root / ('cache_' + self.task + '.json')
            def checkpoint_dir(self, algorithm):
                return root / 'ckpts' / self.model_name(algorithm)
            def log_dir(self, algorithm):
                return root / 'logs' / self.model_name(algorithm)
            def model_name(self, algorithm):
                # Training scripts that construct paths from ROOT_DIR still write below the temporary root.
                return str(root / 'logs' / super().model_name(algorithm))
        with (root / 'console.txt').open('w', encoding='utf-8') as console, redirect_stdout(console):
            for task in sorted({'S', irws_task}):
                precompute(Scenario(task=task, mode='precompute', seed_spec='1-2',
                                    cache_output=str(root / ('cache_' + task + '.json'))))
            for algorithm in ('mahdrl', 'ablation_hostrand', 'ablation_mgrrand', 'ablation_vmrand',
                              'ablation_nohost', 'pd3qn', 'marl', 'irws'):
                module = importlib.import_module('algorithms.' + algorithm + '.train')
                scenario = SmokeScenario(mode='train', episodes=1, task=irws_task if algorithm == 'irws' else 'S',
                    normalization_mode=normalization_mode, normalization_config=normalization_config)
                built = []
                if algorithm in ('mahdrl', 'ablation_hostrand', 'ablation_mgrrand', 'ablation_vmrand', 'ablation_nohost', 'pd3qn'):
                    original = module.D3QNAgent
                    def make_agent(*args, **kwargs):
                        kwargs.update(hidden_dims=(16, 8), batch_size=4, buffer_size=32, device=torch.device('cpu'))
                        agent = original(*args, **kwargs)
                        built.append(agent)
                        return agent
                    patches = [patch.object(module, 'D3QNAgent', side_effect=make_agent)]
                elif algorithm == 'marl':
                    original_cfg = module.A2CConfig
                    def make_cfg(*args, **kwargs):
                        kwargs.update(hidden_dims=(16, 8), device='cpu')
                        return original_cfg(*args, **kwargs)
                    patches = [patch.object(module, 'A2CConfig', side_effect=make_cfg)]
                else:
                    original = module.PPOVMAgent
                    def make_ppo(cfg, hidden):
                        agent = original(cfg, hidden=(16, 8))
                        built.append(agent)
                        return agent
                    patches = [patch.object(module, 'PPOVMAgent', side_effect=make_ppo)]
                from contextlib import ExitStack
                with ExitStack() as stack:
                    original_logger = module.CSVLogger
                    def make_logger(*args, **kwargs):
                        logger = original_logger(*args, **kwargs)
                        stack.callback(logger.close)
                        return logger
                    stack.enter_context(patch.object(module, 'CSVLogger', side_effect=make_logger))
                    stack.enter_context(patch.object(module, 'SCENARIO', scenario))
                    for item in patches:
                        stack.enter_context(item)
                    module.train()
                logdir = scenario.log_dir(algorithm)
                # HRL/PD3QN/MARL use os.path.join(ROOT_DIR, 'outputs', 'logs', model_name).
                diagnostic = logdir / 'f11_diagnostics.jsonl'
                records = [json.loads(line) for line in diagnostic.read_text(encoding='utf-8').splitlines()]
                assert records and records[-1]['pending'] == 0, algorithm
                learners = records[-1]['learners']
                assert all(v['gradient_updates'] > 0 for v in learners.values()), algorithm
                assert all(v['nonterminal'] > 0 and v['bootstrap_nonzero_ratio'] > 0 for v in learners.values()), algorithm
                if 'reward_audit' in records[-1]:
                    assert all(abs(value) < 1e-6 for key, value in records[-1]['reward_audit'].items() if key.endswith('balance')), algorithm
                if algorithm == 'irws' and irws_task == 'L':
                    assert learners['VM_PPO']['nonterminal_rollouts'] > 0, algorithm
                if verify_checkpoints:
                    from common.f18_state_normalization import check_checkpoint
                    files=list(scenario.checkpoint_dir(algorithm).glob('*.pth'))
                    assert files, algorithm
                    for file in files:
                        ckpt=torch.load(file,map_location='cpu',weights_only=False)
                        try:check_checkpoint(ckpt.get('normalization'),scenario.normalization_metadata())
                        except ValueError as exc:raise ValueError(f'{algorithm}/{file.name}: {exc}') from exc
                    # Exercise each original evaluation checkpoint loader with matching small networks.
                    ev=importlib.import_module('algorithms.'+algorithm+'.evaluate')
                    with ExitStack() as eval_stack:
                        eval_stack.enter_context(patch.object(ev,'SCENARIO',scenario))
                        if hasattr(ev,'D3QNAgent'):
                            original_eval=ev.D3QNAgent
                            def small_eval(*args,**kwargs):
                                kwargs.update(hidden_dims=(16,8),batch_size=4,buffer_size=32,device=torch.device('cpu'))
                                return original_eval(*args,**kwargs)
                            eval_stack.enter_context(patch.object(ev,'D3QNAgent',side_effect=small_eval))
                        elif algorithm=='marl':
                            original_eval_cfg=ev.A2CConfig
                            def small_eval_cfg(*args,**kwargs):
                                kwargs.update(hidden_dims=(16,8),device='cpu')
                                return original_eval_cfg(*args,**kwargs)
                            eval_stack.enter_context(patch.object(ev,'A2CConfig',side_effect=small_eval_cfg))
                        else:
                            original_eval_ppo=ev.PPOVMAgent
                            def small_eval_ppo(cfg,hidden):return original_eval_ppo(cfg,hidden=(16,8))
                            eval_stack.enter_context(patch.object(ev,'PPOVMAgent',side_effect=small_eval_ppo))
                        if algorithm=='marl':
                            kw=ev.build_env_kwargs(str(scenario.cache_path))
                            agents=ev.build_agents_and_probe_dims(kw,1,'cpu',str(scenario.checkpoint_dir(algorithm)/'best_global.pth'),
                                str(scenario.checkpoint_dir(algorithm)/'best_local_actor_critics.pth'))
                        elif algorithm=='pd3qn':ev.build_agent_and_env_kwargs()
                        elif algorithm=='irws':ev.build_agents_and_load(ev.build_env_kwargs_irws(random_seed=1))
                        elif algorithm=='ablation_nohost':ev.build_two_layer_agents_and_load(ev.build_env_kwargs_nohost(random_seed=1))
                        elif algorithm in ('ablation_hostrand','ablation_mgrrand','ablation_vmrand'):
                            # Builder names vary with the retained layers.
                            builder=next(getattr(ev,n) for n in dir(ev) if n.startswith('build_') and n.endswith('agents_and_load'))
                            kw_builder=next(getattr(ev,n) for n in dir(ev) if n.startswith('build_env_kwargs'))
                            builder(kw_builder(random_seed=1))
                        else:ev.build_three_layer_agents_and_load(ev.build_env_kwargs_hrl_mix(random_seed=1))
                    records[-1]['f18_checkpoints_checked']=len(files)
                    records[-1]['f18_evaluation_loader_checked']=True
                report['algorithms'][algorithm] = records[-1]
                print('smoke passed:',algorithm,file=progress,flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = run_smoke()
    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"F11: {len(report['algorithms'])} actual training loops completed; report: {args.out}")
