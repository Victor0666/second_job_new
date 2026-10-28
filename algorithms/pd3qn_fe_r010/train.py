"""Independent PD3QN-FE-R010 training; historical PD3QN updates and warmup are retained."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from algorithms.pd3qn_fe_r010 import ALGORITHM, DISPLAY_NAME, MODEL_VERSION, reward_metadata
from algorithms.pd3qn_fe_r010.runtime import (env_kwargs, evaluate_instances, make_agent,
    save_checkpoint, set_seed)
from algorithms.pd3qn_fe_r010.scenario import parse_scenario
from algorithms.pd3qn_fe_r010.selection import selection_key, selection_metadata, validation_selection
from common.experiment_measurements import TrainingRecorder
from common.f11 import F11Logger, pd3qn_transition, protect_training_outputs
from common.f18_state_normalization import normalization_diagnostics
from common.f22_instances import episode_identity, write_run_metadata
from common.metrics_logger import CSVLogger
from environments.pd3qn import CloudWorkflowEnv_PD3QN_Cost


def train(scenario):
    save_dir, logs = scenario.checkpoint_dir(), scenario.log_dir()
    protect_training_outputs(save_dir, logs)
    scenario.normalization_metadata()
    for folder in (save_dir, logs):
        write_run_metadata(folder, scenario)
    kwargs = env_kwargs(scenario)
    set_seed(scenario.model_seed)
    env = CloudWorkflowEnv_PD3QN_Cost(**kwargs)
    agent = make_agent(scenario, env)
    recorder = TrainingRecorder(logs, scenario, ALGORITHM, env, {DISPLAY_NAME: agent})
    f11_log = F11Logger(logs)
    metric_fields = ['task_avg_lateness', 'task_avg_tardiness', 'pct_tardy', 'wf_avg_lateness',
        'wf_avg_tardiness', 'wf_lateness_sum', 'wf_tardiness_sum', 'makespan']
    fields = ['step', 'episode', 'type', 'eps', 'ep_length', 'episode_energy', *metric_fields,
        'eval_energy', 'eval_workflow_success_rate', 'eval_feasible',
        'selected_as_best', 'best_episode']
    logger = CSVLogger(str(logs / 'train_metrics_pd3qn_fe_r010_cost.csv'), fields)
    best = None
    global_step = 0
    print(f'[{DISPLAY_NAME} train] {scenario.model_name()} device={agent.device}', flush=True)
    try:
        for episode in range(scenario.episodes):
            obs, _ = env.reset()
            agent.begin_episode()
            done = False
            steps = 0
            while not done:
                obs = np.array(obs, copy=True)
                mask = np.array(env.get_action_mask(), copy=True)
                action = agent.select_action(obs, mask, deterministic=False, count_step=True)
                obs2, reward, terminated, truncated, _ = env.step(int(action))
                done = bool(terminated or truncated)
                pd3qn_transition(agent, obs, mask, int(action), float(reward), obs2,
                                  env.get_action_mask(), terminated, truncated)
                if len(agent.buffer) >= max(agent.batch_size, int(agent.buffer_size * .03)):
                    agent.update()
                obs = obs2
                steps += 1
                global_step += 1
            f11_log.log(episode, {DISPLAY_NAME: agent}, normalization=normalization_diagnostics(env),
                instance=episode_identity(env, scenario), pending=0,
                terminated=bool(terminated), truncated=bool(truncated))
            metrics = env.get_episode_metrics()
            rows = recorder.validation(evaluate_instances, agent, scenario,
                                       scenario.validation_seeds(), kwargs=kwargs)
            candidate = validation_selection(rows)
            selected = best is None or selection_key(candidate) < selection_key(best)
            if selected:
                best = dict(candidate, episode=episode, completed_episodes=episode + 1)
                save_checkpoint(agent, save_dir / 'best_pd3qn_fe_r010.pth', scenario,
                                episode, candidate, best, rows, 'best')
            if global_step % 2000 == 0:
                save_checkpoint(agent, save_dir / f'pd3qn_fe_r010_step{global_step}.pth', scenario,
                                episode, candidate, best, rows, 'periodic')
            logger.log(step=global_step, episode=episode, type='episode', eps=agent.epsilon(),
                ep_length=steps, episode_energy=float(metrics['total_energy']),
                **{key: float(metrics[key]) for key in metric_fields},
                eval_energy=candidate['mean_total_energy'],
                eval_workflow_success_rate=candidate['workflow_success_rate'],
                eval_feasible=candidate['feasible'],
                selected_as_best=selected, best_episode=best['episode'])
            record = dict(algorithm=ALGORITHM, episode=episode,
                instance_seed=scenario.training_seed(episode), validation=rows,
                validation_selection=candidate, best_validation_selection=dict(best), selected_as_best=selected)
            with (logs / 'validation_selection.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(f'[{DISPLAY_NAME} {episode + 1}/{scenario.episodes}] '
                f'validation_energy={candidate["mean_total_energy"]:.6f} '
                f'SR={candidate["workflow_success_rate"]:.4f} '
                f'feasible={candidate["feasible"]} selected={selected}', flush=True)
            if episode + 1 < scenario.episodes:
                env.random_seed = scenario.training_seed(episode + 1)
                kwargs['random_seed'] = int(env.random_seed)
        save_checkpoint(agent, save_dir / 'pd3qn_fe_r010_final.pth', scenario,
                        episode, candidate, best, rows, 'final')
        summary = dict(algorithm=ALGORITHM, model_version=MODEL_VERSION,
            reward=reward_metadata(), checkpoint_selection=selection_metadata(),
            protocol=scenario.selection_protocol(), completed_episodes=scenario.episodes,
            best_validation_selection=best, feasible_model_found=best['feasible'], resume_supported=False)
        (logs / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f'[{DISPLAY_NAME} train completed] best={save_dir / "best_pd3qn_fe_r010.pth"} '
              f'feasible_model_found={best["feasible"]}', flush=True)
        return agent, summary
    finally:
        recorder.close()
        logger.close()
        if hasattr(env, 'close'):
            env.close()


if __name__ == '__main__':
    train(parse_scenario('train'))
