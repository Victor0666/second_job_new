"""F19 real-input smoke: temporary training-only references; no policy training."""
import argparse
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import numpy as np
import torch
from common.scenarios import Scenario
from common.precompute_deadlines import main as precompute, build_env_kwargs
from common.f18_calibrate_normalization import sample_episode
from common.f18_state_normalization import VERSION,CURRENT_VERSION,SCALE_UNITS,normalization_diagnostics
from common.d3qn_agent_EXDehid import D3QNAgent
from environments.mahdrl_fcfs import CloudWorkflowEnv_VMAgents


def run():
    with tempfile.TemporaryDirectory(prefix='f19_real_input_') as directory:
        root=Path(directory)
        class ScratchScenario(Scenario):
            @property
            def cache_path(self):return root/'cache.json'
        scene=ScratchScenario(mode='train')
        with redirect_stdout(io.StringIO()):
            precompute(Scenario(mode='precompute',seed_spec='0',cache_output=str(scene.cache_path)))
            calibration=sample_episode(scene,0)
        scales=dict(R_ref=max(1,math.ceil(np.percentile(calibration['ready'],99))),
                    T_ref=float(np.percentile(calibration['deadlines'],95)))
        scales.update({key:float(np.percentile(values,95)) for key,values in calibration['physical'].items()})
        scales.update({key:calibration[key] for key in ('vm_pc_ref','vm_bw_ref','host_pc_ref')})
        config=root/'scales.json'
        config.write_text(json.dumps(dict(normalization_version=VERSION,units=SCALE_UNITS,scenes={'SS_T':scales})),encoding='utf-8')
        kwargs=build_env_kwargs(scene)
        kwargs.pop("deadline_alpha")
        kwargs.update(random_seed=0,deadline_mode='cache_fcfs',deadline_cache_path=str(scene.cache_path),
                      workflows_per_episode=2,deadline_alpha_small_prob=scene.small_probability,
                      normalization_config=str(config),source_train_scene='SS_T')
        env=CloudWorkflowEnv_VMAgents(**dict(kwargs,normalization_mode=CURRENT_VERSION))
        old=CloudWorkflowEnv_VMAgents(**dict(kwargs,normalization_mode=VERSION))
        env.reset();old.reset();torch.manual_seed(1);agent=None;assignments=0
        while not env.done_flag:
            assert env.ready_task_ids==old.ready_task_ids
            gid=min(env.ready_task_ids,key=lambda i:(env.task_ready_time[i],i))
            vm=int(np.flatnonzero(env._vm_idle_mask())[0]);host=env.vms[vm].host_id
            obs,mask=env._build_vm_obs_for_task_host(gid,host)
            if agent is None:agent=D3QNAgent(len(obs),len(mask),hidden_dims=(16,8),device=torch.device('cpu'))
            with torch.no_grad():agent.online(torch.as_tensor(obs).unsqueeze(0))
            for world in (env,old):
                world._assign_task_to_specific_vm(gid,vm);world._advance_until_decision_energy_only()
            assert env.task_end_time==old.task_end_time and env.total_energy==old.total_energy
            assignments+=1
        assert old.done_flag and assignments>0
        return dict(verification_only=True,formal_calibration=False,training_executed=False,
            calibration_scene='SS_T',calibration_instance_seeds=[0],model_seed=1,reference_workflows=50,
            replay_instance_seed=0,replay_workflows=2,assignments=assignments,
            normalization_version=CURRENT_VERSION,reference_scales=scales,
            ratio_diagnostics=normalization_diagnostics(env)['numerics'],
            untrained_VA_network_inputs=agent.f11.numerics.snapshot(),
            total_energy=env.total_energy,f18_f19_energy_gap=abs(env.total_energy-old.total_energy),
            notes='Temporary one-seed training references, not formal nine-scene calibration. Small untrained network reads real VA observations; no learning updates.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',default='docs/F19_real_input_smoke.json');args=parser.parse_args()
    result=run();Path(args.out).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('F19 real-input smoke passed:',result['assignments'],'assignments; no policy training')
