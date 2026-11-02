"""Run the actual background Bash functions with a harmless Python substitute."""
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which('git')
BASH = shutil.which('bash') or (str(Path(GIT).parents[1]/'bin/bash.exe') if GIT else None)
if BASH and not Path(BASH).is_file():
    BASH = None
VARIANTS = ('mahdrl_as_fast_hscore_speed', 'mahdrl_as_fast_hscore_speed2',
            'mahdrl_as_fast_hscore_speed2_sync_task')
MOCK = '''#!/usr/bin/env bash
printf 'GPU=%s\\t' "$CUDA_VISIBLE_DEVICES" >> "$TRACE_FILE"
printf '%s\\t' "$@" >> "$TRACE_FILE"
printf '\\n' >> "$TRACE_FILE"
previous=""
for arg in "$@"; do
    if [[ "$previous" == --task ]]; then task="$arg"; fi
    previous="$arg"
done
if [[ -n "${FAIL_TASK:-}" && "$task" == "$FAIL_TASK" ]]; then exit 7; fi
'''


@unittest.skipUnless(BASH, 'Bash unavailable')
class CommandTests(unittest.TestCase):
    def run_bash(self, script):
        with tempfile.TemporaryDirectory(prefix='hscore_commands_') as directory:
            root = Path(directory)
            (root/'bin').mkdir()
            mock = root/'bin/python'
            mock.write_text(MOCK, encoding='utf8', newline='\n')
            mock.chmod(0o755)
            setup = 'set -e\nexport PATH="$PWD/bin:$PATH"\nexport TRACE_FILE="$PWD/calls.tsv"\n'
            subprocess.run([BASH, '--noprofile', '--norc', '-s'], input=setup+script, text=True,
                           encoding='utf8', cwd=root, capture_output=True, check=True, timeout=90)
            trace = root/'calls.tsv'
            rows = [line.rstrip('\t').split('\t') for line in trace.read_text(encoding='utf8').splitlines()] if trace.exists() else []
            logs = {p.name: p.read_text(encoding='utf8') for p in root.glob('outputs/server_*_logs/*.log')}
            return rows, logs

    def source(self, variant):
        return 'source '+shlex.quote((ROOT/'docs'/f'{variant}_commands.sh').as_posix())+'\n'

    def test_syntax_all_nine_sources_and_all_27_target_calls(self):
        script = ''
        for variant in VARIANTS:
            subprocess.run([BASH, '-n', str(ROOT/'docs'/f'{variant}_commands.sh')], check=True)
            script += self.source(variant)
        script += '[[ ! -d outputs ]]\n'
        for variant in VARIANTS:
            script += f'''
for res in S M L; do
    for ddl in Tight Medium Loose; do
        train_{variant} "$res" "$ddl"
        wait
        eval_{variant}_all "$res" "$ddl"
        wait
    done
done
'''
        rows, logs = self.run_bash(script)
        self.assertEqual((len(rows), len(logs)), (36*len(VARIANTS), 18*len(VARIANTS)))
        for variant in VARIANTS:
            selected = [r for r in rows if r[3].startswith(f'algorithms/{variant}/')]
            self.assertEqual(len(selected), 36)
            train = [r for r in selected if r[3].endswith('/train.py')]
            evaluation = [r for r in selected if r[3].endswith('/evaluate.py')]
            self.assertEqual((len(train), len(evaluation)), (9, 27))
            for row in selected:
                self.assertEqual(row[:3], ['GPU=0', '-u', 'launch.py'])
                args = row[4:]
                value = lambda flag: args[args.index(flag)+1]
                self.assertEqual(value('--model-seed'), '1')
                self.assertEqual(value('--train-seed-cycles'), '1')
                self.assertEqual(value('--device'), 'cuda')
                self.assertEqual(value('--ha-update-period'), '4')
                self.assertIn('--va-preview', args)
                self.assertEqual(value('--diagnostics-every'), '100')
                self.assertEqual(value('--profile-mode'), 'sampled')
                self.assertEqual(value('--profile-every'), '100')
                self.assertEqual(value('--experiment-protocol'), 'configs/f22_split_v1.json')
                self.assertEqual(value('--normalization-mode'), 'f19_shared_v1')
                if row in train:
                    self.assertEqual(value('--episodes'), '600')
                    self.assertEqual(value('--task'), 'S')
                else:
                    self.assertEqual(value('--seeds'), '700-729')
                    self.assertEqual(value('--train-task'), 'S')
                    self.assertEqual(value('--train-resource'), value('--resource'))
                    self.assertEqual(value('--train-deadline'), value('--deadline'))
            for res in 'SML':
                for ddl in ('Tight', 'Medium', 'Loose'):
                    targets = [r[r.index('--task')+1] for r in evaluation
                               if r[r.index('--resource')+1] == res and r[r.index('--deadline')+1] == ddl]
                    self.assertEqual(targets, ['S', 'M', 'L'])
        self.assertTrue(all('[DONE]' in log for log in logs.values()))

    def test_failure_stops_remaining_targets(self):
        for variant in VARIANTS:
            script = self.source(variant)+f'''
export FAIL_TASK=M
eval_{variant}_all S Tight
pid=$(cat outputs/server_eval_logs/*.pid)
if wait "$pid"; then exit 9; else [[ $? == 1 ]]; fi
'''
            rows, logs = self.run_bash(script)
            self.assertEqual([r[r.index('--task')+1] for r in rows], ['S', 'M'])
            self.assertEqual(len(logs), 1)
            self.assertNotIn('[ALL_DONE]', next(iter(logs.values())))

    def test_6000_train_and_evaluation_share_the_cycle_settings(self):
        script = ''
        for variant in VARIANTS:
            script += self.source(variant)+f'''
train_{variant} M Tight 10
wait
eval_{variant}_all M Tight "S M L" 10
wait
'''
        rows, logs = self.run_bash(script)
        self.assertEqual((len(rows), len(logs)), (4*len(VARIANTS), 2*len(VARIANTS)))
        for row in rows:
            self.assertEqual(row[row.index('--episodes')+1], '6000')
            self.assertEqual(row[row.index('--train-seed-cycles')+1], '10')
        self.assertTrue(all('_e6000r10_' in name for name in logs))

    def test_invalid_parameters_start_no_jobs(self):
        script = ''
        for variant in VARIANTS:
            script += self.source(variant)+f'''
if train_{variant} X Tight; then exit 9; fi
if train_{variant} S invalid; then exit 9; fi
if train_{variant} S Tight 0; then exit 9; fi
if train_{variant} S Tight -1; then exit 9; fi
if train_{variant} S Tight 1.5; then exit 9; fi
if eval_{variant}_all S Tight "S M L" 0; then exit 9; fi
if eval_{variant}_all S Tight "S invalid"; then exit 9; fi
'''
        script += '[[ ! -d outputs ]]\n'
        self.assertEqual(self.run_bash(script), ([], {}))


if __name__ == '__main__':
    unittest.main()
