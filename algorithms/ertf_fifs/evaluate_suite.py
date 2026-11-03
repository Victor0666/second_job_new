"""Evaluate standard, stress, or 27-scenario bursty workloads with paired test seeds."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import csv
import json
import math
import os
from pathlib import Path
import statistics
import time

from scipy.stats import t

from algorithms.ertf_fifs import evaluate
from common.bursty_workload import PROFILE as BURSTY_PROFILE, apply_arrival_profile, workload_metadata
from common.f22_cache import check_cache
from common.f22_instances import artifact_info
from common.scenarios import Scenario, parse_seeds

PROFILES = ('standard', 'stress_a1_l006_ref003', 'stress_a15_l006_ref003')
SUPPORTED_PROFILES = (*PROFILES, BURSTY_PROFILE)
ESP_THRESHOLD = .92
METRICS = ('total_energy', 'makespan', 'wf_success_rate', 'wf_avg_tardiness')
ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, seeds, *, esp_threshold=None):
    if sorted(int(row['seed']) for row in rows) != sorted(seeds):
        raise ValueError('Expected each requested test seed exactly once')
    result = dict(n=len(rows))
    for key in METRICS:
        values = [float(row[key]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Non-finite metric: ' + key)
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.
        half = float(t.ppf(.975, len(values)-1))*std/math.sqrt(len(values)) if len(values) > 1 else 0.
        result.update({key+'_mean': mean, key+'_std': std,
                       key+'_ci95_low': mean-half, key+'_ci95_high': mean+half})
    if esp_threshold is not None:
        result.update(esp_threshold=esp_threshold, esp=sum(
            int(row['wf_success_count']) >= math.ceil(esp_threshold * int(row['wf_count']) - 1e-12)
            for row in rows) / len(rows))
    return result


def validate_episode(scene, native, record, workflows):
    metrics = record['metrics']
    if metrics['wf_count'] != 50 or metrics['wf_completed_count'] != 50 or len(workflows) != 50:
        raise ValueError('Incomplete workflow episode')
    if native['task_finished'] != metrics['task_count'] or record['timing']['allocation_count'] != metrics['task_count']:
        raise ValueError('Task completion/allocation mismatch')
    for key, alias in (('total_energy', 'total_energy'), ('makespan', 'makespan'),
                       ('wf_success_rate', 'wf_avg_success'), ('wf_avg_tardiness', 'wf_avg_tardiness')):
        if not math.isclose(float(native[alias]), metrics[key], rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError('Native/common measurement mismatch: ' + key)
    if not math.isfinite(native['energy_check']) or not math.isclose(native['energy_check'], native['total_energy'], rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError('Independent energy integration mismatch')
    if scene.is_stress_workload and any(not math.isclose(float(row['deadline_alpha']), scene.deadline_choices[0], abs_tol=1e-9) for row in workflows):
        raise ValueError('Stress deadline alpha mismatch')
    if scene.is_bursty_workload:
        expected = apply_arrival_profile([0.] * 50, BURSTY_PROFILE, int(record['seed']), 50, .03)
        if [float(row['arrival']) for row in workflows] != expected:
            raise ValueError('Bursty arrival times differ from the paired generator')
        for row in workflows:
            alpha = float(row['deadline_alpha'])
            if not any(math.isclose(alpha, choice, abs_tol=1e-9) for choice in scene.deadline_choices) or not math.isclose(
                    float(row['deadline']), float(row['arrival']) + alpha * float(row['reference_makespan']),
                    rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError('Bursty relative deadline budget mismatch')


def run_scene(job):
    scene, output = job
    os.environ['EXPERIMENT_METRICS'] = '1'
    evaluate.torch.set_num_threads(1)
    evaluate.SCENARIO = scene
    evaluate.set_seed(scene.model_seed)
    tag = f'{scene.workload_profile}_{scene.task}{scene.resource}_{scene.deadline}'
    details = output/'_details'/tag
    details.mkdir(parents=True, exist_ok=True)
    csv_path = output/f'eval_{tag}.csv'
    rows = []
    if csv_path.exists():
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        present = [int(row['seed']) for row in rows]
        if len(present) != len(set(present)) or not set(present) <= set(scene.seeds):
            raise ValueError('Unexpected or duplicate existing seeds: '+tag)
        for row in rows:
            path = details/'episodes'/f"seed{row['seed']}.json"
            recorded = json.loads(path.read_text(encoding='utf-8'))['metrics']
            if any(not math.isclose(float(row[key]), recorded[key], rel_tol=1e-10) for key in METRICS):
                raise ValueError('Existing CSV/detail mismatch: '+str(path))
    completed = {int(row['seed']) for row in rows}
    with (output/'logs'/f'{tag}.log').open('a', encoding='utf-8', buffering=1) as log, redirect_stdout(log):
        kwargs = evaluate.build_env_kwargs(scene)
        for seed in scene.seeds:
            if seed in completed:
                continue
            started = time.perf_counter()
            native = evaluate.run_one_episode(evaluate.MaskOnlyERTFEnv, kwargs, seed, measurement_directory=details)
            record = json.loads((details/'episodes'/f'seed{seed}.json').read_text(encoding='utf-8'))
            with (details/'workflows'/f'seed{seed}.csv').open(encoding='utf-8', newline='') as stream:
                workflows = list(csv.DictReader(stream))
            validate_episode(scene, native, record, workflows)
            row = dict(algorithm='ERTF-H-FIFS', workload_profile=scene.workload_profile,
                       task=scene.task, resource=scene.resource, deadline=scene.deadline, seed=seed,
                       **{key: record['metrics'][key] for key in METRICS},
                       energy_check=native['energy_check'], wf_count=record['metrics']['wf_count'],
                       wf_success_count=record['metrics']['wf_success_count'],
                       task_count=record['metrics']['task_count'], wall_seconds=time.perf_counter()-started)
            with csv_path.open('a', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if not rows:
                    writer.writeheader()
                writer.writerow(row)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            write_json(details/'progress.json', dict(completed=len(rows), expected=len(scene.seeds), last_seed=seed))
    summary = dict(workload_profile=scene.workload_profile, task=scene.task, resource=scene.resource,
                   deadline=scene.deadline, **summarize(rows, scene.seeds, esp_threshold=ESP_THRESHOLD))
    write_json(details/'summary.json', summary)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seeds', default='700-729')
    parser.add_argument('--profiles', nargs='+', choices=SUPPORTED_PROFILES, default=list(PROFILES))
    parser.add_argument('--tasks', nargs='+', choices=list('SML'), default=list('SML'))
    parser.add_argument('--resources', nargs='+', choices=list('SML'), default=list('SML'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    seeds = parse_seeds(args.seeds)
    scenes = [Scenario(task=task, resource=resource, deadline=deadline, workload_profile=profile, seed_spec=args.seeds)
              for profile in dict.fromkeys(args.profiles) for task in dict.fromkeys(args.tasks)
              for resource in dict.fromkeys(args.resources)
              for deadline in (('Tight', 'Medium', 'Loose') if profile in ('standard', BURSTY_PROFILE) else ('Tight',))]
    for task, resource in sorted({(s.task, s.resource) for s in scenes}):
        check_cache(Scenario(task=task, resource=resource))
    sources = sorted(set((ROOT/'algorithms/ertf_fifs').glob('*.py')) |
                     set((ROOT/'common').glob('*.py')) | set((ROOT/'environments').glob('*.py')))
    identity = dict(algorithm='ERTF-H-FIFS', seeds=list(seeds), metrics=list(METRICS), workflows_per_episode=50,
                    energy_accounting='host_task_window_v1', observation_builder='mask_only_equivalent_to_original',
                    makespan_definition='last completion minus first arrival',
                    tardiness_definition='mean over all completed workflows, including zero tardiness',
                    success_rate_unit='fraction [0,1]', timing_comparable=False,
                    timing_note='Concurrent CPU simulation; wall/decision timings are not cross-algorithm efficiency evidence',
                    scenarios=[dict(profile=s.workload_profile, task=s.task, resource=s.resource, deadline=s.deadline,
                                    arrival_lambda=s.arrival_lambda, deadline_choices=s.deadline_choices, deadline_probs=s.deadline_probs) for s in scenes],
                    sources=[artifact_info(path) for path in sources],
                    protocol=artifact_info(ROOT/'configs/f22_split_v1.json'),
                    caches=[artifact_info(path) for path in sorted({s.cache_path for s in scenes})])
    if BURSTY_PROFILE in args.profiles:
        identity.update(bursty_workload=workload_metadata(), esp_threshold=ESP_THRESHOLD,
                        esp_definition='fraction of seeds with workflow deadline success rate >= 0.92')
    output = args.out.resolve()
    identity = json.loads(json.dumps(identity))
    if args.resume:
        previous = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        if previous != identity:
            raise ValueError('Resume identity mismatch: source, cache, protocol or requested scenes changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'manifest.json', identity)
    (output/'logs').mkdir(exist_ok=True)
    started = time.perf_counter()
    rows, summaries = [], []
    print(f'START scenes={len(scenes)} episodes={len(scenes)*len(seeds)} workers={args.workers} output={output}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(run_scene, (scene, output)): scene for scene in scenes}
        for future in as_completed(pending):
            scene_rows, summary = future.result()
            rows.extend(scene_rows)
            summaries.append(summary)
            print(f'DONE {len(summaries)}/{len(scenes)} {summary}', flush=True)
            write_json(output/'progress.json', dict(completed_scenes=len(summaries), expected_scenes=len(scenes),
                       completed_episodes=len(rows), expected_episodes=len(scenes)*len(seeds), wall_seconds=time.perf_counter()-started))
    summaries.sort(key=lambda r: (SUPPORTED_PROFILES.index(r['workload_profile']), r['deadline'], r['task'], r['resource']))
    rows.sort(key=lambda r: (SUPPORTED_PROFILES.index(r['workload_profile']), r['deadline'], r['task'], r['resource'], int(r['seed'])))
    write_csv(output/'all_episodes.csv', rows)
    write_csv(output/'summary_45_scenarios.csv', summaries)
    if BURSTY_PROFILE in args.profiles:
        write_csv(output/'summary_scenarios.csv', summaries)
    write_json(output/'summary.json', dict(status='complete', scenarios=len(summaries), episodes=len(rows),
               wall_seconds=time.perf_counter()-started, summaries=summaries))
    print(f'ALL_DONE scenes={len(summaries)} episodes={len(rows)}', flush=True)


if __name__ == '__main__':
    main()
