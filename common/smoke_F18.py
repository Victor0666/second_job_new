"""F18 integration: genuine temporary training calibration and eight original train/eval entry points."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import sys
from common.scenarios import Scenario
from common.precompute_deadlines import main as precompute
from common.f18_calibrate_normalization import nine_scenarios, sample_episode, summarize, cache_info
from common.f18_state_normalization import VERSION, scene_id
from common.smoke_F11 import run_smoke


def run():
    progress=sys.stdout
    with tempfile.TemporaryDirectory(prefix='F18_integration_') as directory:
        root=Path(directory);(root/'episodes').mkdir()
        class LocalScenario(Scenario):
            @property
            def cache_path(self):return root/('cache_'+self.resource+'.json')
        manifest=dict(normalization_version=VERSION,scenes={},verification_only=True,seeds=[101])
        with redirect_stdout(io.StringIO()):
            for resource in 'SML':
                precompute(Scenario(resource=resource,mode='precompute',seed_spec='101',cache_output=str(root/('cache_'+resource+'.json'))))
            for original in nine_scenarios():
                scenario=LocalScenario(mode='train',resource=original.resource,deadline=original.deadline)
                sid=scene_id('S',scenario.resource,scenario.deadline)
                manifest['scenes'][sid]=dict(seeds=[101],scenario=scenario.describe(),cache=cache_info(scenario,[101]))
                result=sample_episode(scenario,101)
                (root/'episodes'/f'{sid}_101.json').write_text(json.dumps(result),encoding='utf-8')
                print('temporary calibration passed:',sid,file=progress,flush=True)
        assert summarize(root,manifest)==0
        config=root/'f18_shared_normalization_v1.json'
        statistics=json.loads(config.read_text(encoding='utf-8'))
        report=run_smoke(normalization_mode=VERSION,normalization_config=str(config),irws_task='S',verify_checkpoints=True)
        report.update(normalization_version=VERSION,calibration_verification_only=True,
            calibration_seeds=[101],calibration_summary=manifest['sample_counts'],calibration_parameters=statistics)
        return report


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',required=True);args=p.parse_args()
    result=run();Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('F18: nine temporary calibration scenes and eight original training/evaluation loaders passed')