"""Exercise the documented Bash jobs without launching training or evaluation."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

from algorithms.mahdrl_oe_ge_apf_score_fast.scenario import parse_scenario


ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which('git')
GIT_BASH = Path(GIT).resolve().parents[1] / 'bin/bash.exe' if GIT else Path('missing-bash')
BASH = str(GIT_BASH) if os.name == 'nt' and GIT_BASH.is_file() else shutil.which('bash')
SOURCES = [(res, ddl) for res in 'SML' for ddl in ('Tight', 'Medium', 'Loose')]


@unittest.skipUnless(BASH, 'Bash is required for the documented Linux commands')
class NoAPFCommandTests(unittest.TestCase):
    def run_jobs(self, mode, calls, failure=False):
        script = ROOT / f'docs/mahdrl_as_fast_noapf_v3_{mode}.sh'
        subprocess.run([BASH, '-n', str(script)], check=True, capture_output=True)
        function = ('train_mahdrl_as_fast_noapf_v3' if mode == 'train'
                    else 'eval_mahdrl_as_fast_noapf_v3_all')
        commands = '\n'.join(
            f'{function} {call}\n'
            f'if wait "$!"; then test {int(failure)} -eq 0; else test "$?" -eq {int(failure)}; fi'
            for call in calls)
        harness = f'''set -e
source {shlex.quote(script.as_posix())}
test ! -e outputs
nohup() {{ "$@"; }}
python() {{
    printf '%s\\t' "$CUDA_VISIBLE_DEVICES" "$@" >> calls.tsv
    printf '\\n' >> calls.tsv
    return {7 if failure else 0}
}}
export -f python
{commands}
'''
        with tempfile.TemporaryDirectory(prefix='noapf_commands_') as directory:
            result = subprocess.run([BASH, '-c', harness], cwd=directory,
                                    env=dict(os.environ, GPU='3'), capture_output=True,
                                    text=True, encoding='utf-8', timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            root = Path(directory)
            rows = [line.split('\t')[:-1] for line in (root / 'calls.tsv').read_text().splitlines()]
            log_paths = list(root.glob('outputs/**/*.log'))
            logs = [path.read_text(encoding='utf-8') for path in log_paths]
            pids = list(root.glob('outputs/**/*.pid'))
            self.assertEqual(len(logs), len(calls))
            self.assertEqual(len(pids), len(calls))
            if mode == 'train':
                self.assertTrue(all(len(path.name) <= 90 for path in log_paths))
            self.assertTrue(all(path.read_text().strip().isdigit() for path in pids))
            self.assertEqual(all('[ALL_DONE]' in log for log in logs), not failure)
            return rows

    def assert_command(self, row, mode, res, ddl, task='S'):
        self.assertEqual(row[:4], ['3', '-u', 'launch.py',
                                  f'algorithms/mahdrl_oe_ge_apf_score_fast/{mode}.py'])
        options = dict(zip(row[4::2], row[5::2]))
        expected = {'--task': task, '--resource': res, '--deadline': ddl,
                    '--model-seed': '1', '--experiment-protocol': 'configs/f22_split_v1.json',
                    '--normalization-mode': 'f19_shared_v1', '--device': 'cuda',
                    '--diagnostics-every': '100', '--profile-mode': 'sampled', '--profile-every': '100'}
        if mode == 'train':
            expected['--episodes'] = '600'
        else:
            expected.update({'--train-task': 'S', '--train-resource': res,
                             '--train-deadline': ddl, '--seeds': '700-729'})
        self.assertEqual(options, expected)
        parse_scenario('train' if mode == 'train' else 'eval', row[4:])

    def test_nine_source_training_jobs(self):
        rows = self.run_jobs('train', [f'{res} {ddl}' for res, ddl in SOURCES])
        self.assertEqual(len(rows), 9)
        for row, (res, ddl) in zip(rows, SOURCES):
            self.assert_command(row, 'train', res, ddl)

    def test_nine_source_jobs_evaluate_27_targets(self):
        rows = self.run_jobs('evaluate', [f'{res} {ddl}' for res, ddl in SOURCES])
        self.assertEqual(len(rows), 27)
        for row, (res, ddl, task) in zip(rows, [(r, d, t) for r, d in SOURCES for t in 'SML']):
            self.assert_command(row, 'evaluate', res, ddl, task)

    def test_target_subset_and_failure_stops_remaining_targets(self):
        rows = self.run_jobs('evaluate', ['M Loose "M L"'])
        self.assertEqual(len(rows), 2)
        for row, task in zip(rows, 'ML'):
            self.assert_command(row, 'evaluate', 'M', 'Loose', task)
        rows = self.run_jobs('evaluate', ['S Tight'], failure=True)
        self.assertEqual(len(rows), 1)
        self.assert_command(rows[0], 'evaluate', 'S', 'Tight')


if __name__ == '__main__':
    unittest.main()
