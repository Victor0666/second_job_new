"""Temporary real training, validation, checkpoint and evaluation of MARL Q."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from algorithms.marl import train, evaluate
from algorithms.marl.q_agents import MODEL_VERSION, QAgent, QConfig, QDecisionTrace
from algorithms.marl.identity import evaluation_directory, evaluation_csv, SCHEDULING_POLICY
from common.precompute_deadlines import main as precompute
from common.scenarios import Scenario
from algorithms.marl.scenario import MARLScenario
from common.test_f18 import fixture_config


def run_smoke(hidden_dims=(16,8), workload_profile='standard'):
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='marl_q_') as directory:
        root = Path(directory)
        config = root/'normalization.json'; fixture_config(config)

        class SmokeScenario(MARLScenario):
            @property
            def cache_path(self): return root/'cache.json'
            def model_name(self, algorithm): return str(root/'logs'/super().model_name(algorithm))
            def checkpoint_dir(self, algorithm): return root/'ckpts'
            def log_dir(self, algorithm): return Path(self.model_name(algorithm))

        scene = SmokeScenario(mode='train', episodes=1, normalization_config=str(config), workload_profile=workload_profile)
        original = train.CloudWorkflowEnv_MARL
        assignment_checks = dict(count=0)
        original_assign = original._assign_task_to_vm
        def assign(env, gid, vm_index):
            assert env.task_state[gid] == 'Ready'
            assert env._vm_idle_mask()[vm_index], 'Policy assigned a busy VM'
            now = env.current_time
            result = original_assign(env, gid, vm_index)
            assert env.records[-1].start_time == now
            assert env.task_state[gid] == 'Running'
            assert 'Queued' not in env.task_state
            assignment_checks['count'] += 1
            return result
        def environment(**kwargs):
            kwargs.update(workflows_per_episode=2, dump_deadlines=False, print_deadlines=False)
            return original(**kwargs)
        def cfg(**kwargs):
            kwargs.update(hidden_dims=hidden_dims, device='cpu')
            return QConfig(**kwargs)

        pending = set()
        original_record, original_update = QDecisionTrace.record, QAgent.update
        original_select = QAgent.select_action
        original_target = QAgent.successor_q
        target_calls = dict(Global=0, Local=0)
        chosen = {}
        def record(trace, *args, **kwargs):
            assert args[0].action == chosen[trace.agent], 'Recorded a different action from the cached draw'
            original_record(trace, *args, **kwargs)
            pending.add(trace.agent)
        def update(agent, *args, **kwargs):
            assert agent in pending
            result = original_update(agent, *args, **kwargs)
            pending.remove(agent)
            return result
        def select(agent, *args, **kwargs):
            action = original_select(agent, *args, **kwargs)
            if not kwargs.get('deterministic', False):
                chosen[agent] = action
            return action

        def target(agent, decision):
            target_calls['Global' if agent.peer_act_dim else 'Local'] += 1
            assert decision.action == chosen[agent]
            return original_target(agent, decision)

        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode='precompute', seed_spec='0,600,700', cache_output=str(scene.cache_path)))
            with patch.object(original,'_assign_task_to_vm',assign), \
                    patch.object(train,'SCENARIO',scene), patch.object(train,'QConfig',side_effect=cfg), \
                    patch.object(train,'CloudWorkflowEnv_MARL',side_effect=environment), \
                    patch.object(QDecisionTrace,'record',record), patch.object(QAgent,'update',update), \
                    patch.object(QAgent,'select_action',select), patch.object(QAgent,'successor_q',target):
                train.train()
                saved = {path:hashlib.sha256(path.read_bytes()).hexdigest()
                         for directory in (scene.checkpoint_dir('marl'),scene.log_dir('marl'))
                         for path in directory.rglob('*') if path.is_file()}
                try:
                    train.train()
                except FileExistsError:
                    pass
                else:
                    raise AssertionError('Repeated training did not refuse existing output')
                assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path,digest in saved.items())
            assert not pending
            records = [json.loads(line) for line in (scene.log_dir('marl')/'f11_diagnostics.jsonl').read_text().splitlines()]
            final = records[-1]
            assert final['pending'] == 0
            assert abs(final['global_reward_balance']) < 1e-7
            assert abs(final['local_reward_balance']) < 1e-7
            assert all(a['gradient_updates'] > 0 for a in final['learners'].values())
            measurement = json.loads((scene.log_dir('marl')/'training_measurements.jsonl').read_text().splitlines()[-1])
            assert measurement['policy_calls']['Global'] == measurement['assignment_calls']
            assert sum(v for k,v in measurement['policy_calls'].items() if k.startswith('Local_')) == measurement['assignment_calls']
            assert all(v > 0 for v in measurement['optimizer_steps'].values())
            assert measurement['algorithm'] == MODEL_VERSION
            with (scene.log_dir('marl')/f'train_metrics_{MODEL_VERSION}.csv').open(newline='') as stream:
                assert all(row['algorithm'] == MODEL_VERSION for row in csv.DictReader(stream))
            for role, diagnostics in final['learners'].items():
                assert measurement['optimizer_steps'][role+'.actor_optim'] == diagnostics['transitions']
                assert measurement['optimizer_steps'][role+'.critic_optim'] == diagnostics['transitions']
            assert target_calls['Global'] == final['learners']['Global']['nonterminal']
            assert target_calls['Local'] == sum(v['nonterminal'] for k,v in final['learners'].items() if k.startswith('Local_'))

            eval_scene = replace(scene, mode='eval', seed_spec='700')
            recorder = evaluate.EvaluationRecorder
            def evaluation_recorder(*args, **kwargs):
                kwargs['directory'] = root/'evaluation'
                return recorder(*args, **kwargs)
            with patch.object(original,'_assign_task_to_vm',assign), \
                    patch.object(evaluate,'SCENARIO',eval_scene), patch.object(evaluate,'QConfig',side_effect=cfg), \
                    patch.object(evaluate,'CloudWorkflowEnv_MARL',side_effect=environment), \
                    patch.object(evaluate,'EvaluationRecorder',side_effect=evaluation_recorder), \
                    patch.object(QAgent,'successor_q',side_effect=AssertionError('Evaluation read TD target')):
                kwargs = evaluate.build_env_kwargs(str(scene.cache_path))
                kwargs.update(eval_scene.normalization_kwargs())
                g, locals, dims = evaluate.build_agents_and_probe_dims(kwargs,700,'cpu',
                    str(scene.checkpoint_dir('marl')/'best_global.pth'),
                    str(scene.checkpoint_dir('marl')/'best_local_actor_critics.pth'))
                if scene.is_stress_workload:
                    from algorithms.marl.runtime import validate_pair
                    try:
                        validate_pair(g, locals, replace(eval_scene, workload_profile='standard'))
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('Stress checkpoint accepted by standard profile')
                before = [{name:{k:v.clone() for k,v in getattr(a,name).state_dict().items()}
                           for name in ('actor','critic')} for a in (g,*locals.agents)]
                result = evaluate.evaluate_one_seed(kwargs,g,locals,700,0)
                for agent, weights in zip((g,*locals.agents),before):
                    for name, state in weights.items():
                        assert all(torch.equal(v,getattr(agent,name).state_dict()[k]) for k,v in state.items())
                assert np.isfinite(result['energy']) and result['energy_check_diff'] < 1e-6
            # Exercise the full CSV entry point as well as the direct per-seed API.
            cli_scene = replace(eval_scene, marl_options={'device':'cpu'})
            out = evaluation_csv(cli_scene, root)
            with patch.object(original,'_assign_task_to_vm',assign), \
                    patch.object(evaluate,'SCENARIO',cli_scene), patch.object(evaluate,'QConfig',side_effect=cfg), \
                    patch.object(evaluate,'CloudWorkflowEnv_MARL',side_effect=environment), \
                    patch.object(evaluate,'evaluation_csv',side_effect=lambda s, _: evaluation_csv(s,root)), \
                    patch.object(evaluate,'evaluation_directory',side_effect=lambda s, _: evaluation_directory(s,root)), \
                    patch.object(QAgent,'successor_q',side_effect=AssertionError('Evaluation read TD target')):
                evaluate.main()
                with out.open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                assert len(rows) == 2 and all(row['algorithm'] == MODEL_VERSION for row in rows)
                artifacts = evaluation_directory(cli_scene, root)
                episode = json.loads((artifacts/'measurements'/'episodes'/'seed700.json').read_text())
                assert episode['algorithm'] == MODEL_VERSION and episode['checkpoints']
                before = {path:path.read_bytes() for path in artifacts.rglob('*') if path.is_file()}
                before[out] = out.read_bytes()
                try:
                    evaluate.main()
                except FileExistsError:
                    pass
                else:
                    raise AssertionError('Repeated evaluation did not refuse existing output')
                assert all(path.read_bytes() == contents for path,contents in before.items())
        return dict(model_version=MODEL_VERSION, hidden_dims=hidden_dims, dims=dims, learners=final['learners'],
                    scheduling_policy=SCHEDULING_POLICY, idle_assignment_checks=assignment_checks['count'],
                    evaluation_csv_name=out.name, evaluation_csv_flat=out.parent.name == MODEL_VERSION,
                    target_estimator='actual_successor_action_q', target_q_calls=target_calls,
                    numerics=final['numerics'],
                    reward_balances=dict(global_balance=final['global_reward_balance'],local_balance=final['local_reward_balance']),
                    assignment_count=measurement['assignment_calls'], policy_calls=measurement['policy_calls'],
                    optimizer_steps=measurement['optimizer_steps'], cached_action_before_update=True,
                    checkpoint_reload=True, evaluation_no_update=True,
                    distinct_algorithm_identity=True, evaluation_csv=True,
                    repeated_training_preserves_outputs=True, repeated_evaluation_preserves_outputs=True)


if __name__ == '__main__':
    print(json.dumps(run_smoke(), indent=2))
