"""F17: actual TA training entry points with temporary outputs and verification-only small networks."""
import argparse
from contextlib import ExitStack, redirect_stdout
import importlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from common.scenarios import Scenario
from common.precompute_deadlines import main as precompute
from common.ta_weights import TA_WEIGHT_MODEL
from environments.mahdrl_heft import CloudWorkflowEnv_VMAgents as Base, MANAGER_ACTION_TABLE


def run():
    torch.set_num_threads(1)
    report = dict(ta_weight_model=TA_WEIGHT_MODEL, verification_overrides=dict(hidden_dims=[16,8],
                  dqn_batch_size=4, dqn_buffer_size=32, episodes=1), algorithms={})
    with tempfile.TemporaryDirectory(prefix='F17_smoke_') as directory:
        root=Path(directory)
        class SmokeScenario(Scenario):
            @property
            def cache_path(self):return root/'cache.json'
            def checkpoint_dir(self,algorithm):return root/'ckpts'/algorithm
            def log_dir(self,algorithm):return root/'logs'/algorithm
            def model_name(self,algorithm):return str(self.log_dir(algorithm))
        with (root/'console.txt').open('w',encoding='utf-8') as console, redirect_stdout(console):
            precompute(Scenario(mode='precompute',seed_spec='1-2',cache_output=str(root/'cache.json')))
            for algorithm in ('mahdrl','ablation_hostrand','ablation_mgrrand','ablation_vmrand','ablation_nohost'):
                module=importlib.import_module('algorithms.'+algorithm+'.train')
                original_agent=module.D3QNAgent
                original_apply=Base.apply_manager_delta
                original_reset=Base.reset
                count=0; resets=0
                def apply(env,delta):
                    nonlocal count
                    candidate=env.combo_weights.astype(np.float64)+np.asarray(delta,dtype=np.float64)
                    matches=np.isclose(MANAGER_ACTION_TABLE,np.asarray(delta)[None,:],rtol=0,atol=1e-8).all(axis=1)
                    assert matches.sum()==1
                    assert env.get_manager_action_mask()[np.flatnonzero(matches)[0]]==1
                    original_apply(env,delta)
                    expected=np.maximum(candidate,0);expected/=expected.sum()
                    np.testing.assert_allclose(env.combo_weights,expected,atol=3e-8)
                    np.testing.assert_array_equal(env.get_manager_state()[-5:],env.combo_weights)
                    assert not hasattr(env,'manager_raw_w')
                    assert env.get_manager_action_mask().any()
                    count+=1
                def reset(env,*args,**kwargs):
                    nonlocal resets
                    value=original_reset(env,*args,**kwargs)
                    np.testing.assert_array_equal(env.combo_weights,np.full(5,.2,dtype=np.float32))
                    resets+=1
                    return value
                def make_agent(*args,**kwargs):
                    kwargs.update(hidden_dims=(16,8),batch_size=4,buffer_size=32,device=torch.device('cpu'))
                    return original_agent(*args,**kwargs)
                with ExitStack() as stack:
                    original_logger=module.CSVLogger
                    def make_logger(*args,**kwargs):
                        logger=original_logger(*args,**kwargs);stack.callback(logger.close);return logger
                    stack.enter_context(patch.object(module,'CSVLogger',side_effect=make_logger))
                    stack.enter_context(patch.object(module,'SCENARIO',SmokeScenario(mode='train',episodes=1,normalization_mode='legacy')))
                    stack.enter_context(patch.object(module,'D3QNAgent',side_effect=make_agent))
                    stack.enter_context(patch.object(Base,'apply_manager_delta',apply))
                    stack.enter_context(patch.object(Base,'reset',reset))
                    module.train()
                log=root/'logs'/algorithm/'f11_diagnostics.jsonl'
                last=json.loads(log.read_text(encoding='utf-8').splitlines()[-1])
                assert last['pending']==0
                assert all(v['gradient_updates']>0 for v in last['learners'].values())
                report['algorithms'][algorithm]=dict(actual_train_entry=True,train_and_validation_weight_actions=count,
                          checked_resets=resets,diagnostics=last)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',required=True);args=parser.parse_args()
    result=run();Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f"F17: {len(result['algorithms'])} actual training/validation entry points passed")
