"""Independent CLI using shared scenario validation; existing parsers stay untouched."""
import argparse
import json
from pathlib import Path
from common.scenarios import Scenario,VM_COUNTS,SMALL_PROBABILITY,DEFAULT_PROTOCOL
from common.f18_state_normalization import CURRENT_VERSION,SHARED_VERSIONS,DEFAULT_CONFIG
from .config import ALGORITHM,DISPLAY_NAME,MODEL_VERSION,IMPLEMENTATION,REWARD_PROTOCOL,BEST_CHECKPOINT,COMMON_REWARD,PaperConfig,SCHEDULING_PROTOCOL
from .selection import selection_metadata, selection_protocol


class FEScenario(Scenario):
    """Preserve shared data/calibration bindings and report the local selector."""
    def reproducibility_metadata(self):
        metadata = super().reproducibility_metadata()
        metadata.update(data_split=selection_protocol(self), checkpoint_selection=selection_metadata(),
                        reward_settings=dict(COMMON_REWARD), scheduling_protocol=SCHEDULING_PROTOCOL)
        return metadata

    def normalization_metadata(self):
        metadata = super().normalization_metadata()
        metadata.update(experiment_protocol=selection_protocol(self), checkpoint_selection=selection_metadata())
        return metadata

    def describe(self, algorithm=None):
        metadata = super().describe()
        metadata.update(algorithm=ALGORITHM, display_name=DISPLAY_NAME, model_version=MODEL_VERSION,
                        experiment_protocol=selection_protocol(self), checkpoint_selection=selection_metadata(),
                        reward_settings=dict(COMMON_REWARD), scheduling_protocol=SCHEDULING_PROTOCOL)
        return metadata


def parse_scenario(mode,algorithm=None,argv=None):
    parser=argparse.ArgumentParser(description=DISPLAY_NAME+': '+mode)
    parser.add_argument('--task',choices=VM_COUNTS,default='S')
    parser.add_argument('--resource',choices=VM_COUNTS,default='S')
    parser.add_argument('--deadline',choices=SMALL_PROBABILITY,default='Tight')
    parser.add_argument('--experiment-protocol',default=str(DEFAULT_PROTOCOL))
    parser.add_argument('--model-seed',type=int,default=1)
    parser.add_argument('--normalization-mode',choices=SHARED_VERSIONS,default=CURRENT_VERSION)
    parser.add_argument('--normalization-config',default=str(DEFAULT_CONFIG))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--queue-capacity',type=int,default=10,help='Positive layout-compatibility value; IDLE never queues tasks on VMs')
    parser.add_argument('--slot-duration',type=float,default=5.,help='Scheduling tick in seconds')
    parser.add_argument('--rollout-size',type=int,default=64,help='Frozen-policy decisions per update, at most 64')
    parser.add_argument('--precision',choices=('float32','float64'),default='float32')
    parser.add_argument('--dry-run',action='store_true')
    if mode=='train':
        parser.add_argument('--episodes',type=int,default=600)
        parser.add_argument('--resume-checkpoint')
    else:
        parser.add_argument('--seeds',default='700-729')
        parser.add_argument('--train-task',choices=VM_COUNTS,default='S')
        parser.add_argument('--train-resource',choices=VM_COUNTS)
        parser.add_argument('--train-deadline',choices=SMALL_PROBABILITY)
        parser.add_argument('--checkpoint-dir')
    args=parser.parse_args(argv)
    if args.queue_capacity<1 or not 0<args.slot_duration<float('inf'):
        parser.error('queue-capacity and slot-duration must be finite and positive')
    if not 1<=args.rollout_size<=64:parser.error('rollout-size must be between 1 and 64')
    try:
        scenario=FEScenario(task=args.task,resource=args.resource,deadline=args.deadline,mode=mode,
            episodes=getattr(args,'episodes',600),seed_spec=getattr(args,'seeds','700-729'),
            train_task=getattr(args,'train_task','S'),train_resource=getattr(args,'train_resource',None),
            train_deadline=getattr(args,'train_deadline',None),checkpoint_override=getattr(args,'checkpoint_dir',None),
            experiment_protocol=args.experiment_protocol,model_seed=args.model_seed,
            normalization_mode=args.normalization_mode,normalization_config=args.normalization_config,
            paper_options=dict(device=args.device,resume_checkpoint=getattr(args,'resume_checkpoint',None),
                               queue_capacity=args.queue_capacity,slot_duration=args.slot_duration,
                               rollout_size=args.rollout_size,precision=args.precision))
    except (ValueError,argparse.ArgumentTypeError) as error:parser.error(str(error))
    if args.dry_run:
        info=scenario.describe(ALGORITHM)
        info.update(algorithm=ALGORITHM,display_name=DISPLAY_NAME,model_version=MODEL_VERSION,implementation=dict(IMPLEMENTATION),reward_protocol=REWARD_PROTOCOL,
                    queue=dict(capacity=args.queue_capacity,slot_seconds=args.slot_duration,discipline='idle_only',waiting_enabled=False),
                    learning=PaperConfig(device=args.device,rollout_size=args.rollout_size,precision=args.precision).as_dict(),
                    retained_deadline_context=False,fixed_Q_head=True,entropy=False)
        from .runner import output_dir, artifact_name, evaluation_directory
        checkpoint_dir=Path(scenario.checkpoint_override).resolve() if scenario.checkpoint_override else output_dir(scenario,'ckpts')
        info.update(checkpoint_dir=str(checkpoint_dir),
                    run_dir=str(output_dir(scenario,'run')),
                    best_checkpoint=str(checkpoint_dir/BEST_CHECKPOINT))
        if mode != 'train':
            target = output_dir(scenario,'run')
            info.update(evaluation_directory=str(target), evaluation_csv=str(target/artifact_name(scenario,'eval')),
                        evaluation_summary=str(target/artifact_name(scenario,'summary')),
                        evaluation_details_directory=str(evaluation_directory(scenario)))
        print(json.dumps(info,indent=2,ensure_ascii=False));raise SystemExit(0)
    scenario.validate_calibration()
    from common.f22_cache import check_cache
    check_cache(scenario)
    return scenario
