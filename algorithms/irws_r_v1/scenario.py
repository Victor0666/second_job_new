"""Independent CLI using shared scenario validation; existing parsers stay untouched."""
import argparse
import json
from common.scenarios import Scenario,VM_COUNTS,SMALL_PROBABILITY,DEFAULT_PROTOCOL
from common.f18_state_normalization import CURRENT_VERSION,SHARED_VERSIONS,DEFAULT_CONFIG
from .config import ALGORITHM,DISPLAY_NAME,MODEL_VERSION,REWARD_PROTOCOL


def parse_scenario(mode,algorithm=None):
    parser=argparse.ArgumentParser(description=DISPLAY_NAME+': '+mode)
    parser.add_argument('--task',choices=VM_COUNTS,default='S')
    parser.add_argument('--resource',choices=VM_COUNTS,default='S')
    parser.add_argument('--deadline',choices=SMALL_PROBABILITY,default='Tight')
    parser.add_argument('--experiment-protocol',default=str(DEFAULT_PROTOCOL))
    parser.add_argument('--model-seed',type=int,default=1)
    parser.add_argument('--normalization-mode',choices=SHARED_VERSIONS,default=CURRENT_VERSION)
    parser.add_argument('--normalization-config',default=str(DEFAULT_CONFIG))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--queue-capacity',type=int,default=10,help='Waiting tasks per VM, excluding the running task; paper unspecified')
    parser.add_argument('--slot-duration',type=float,default=1.,help='Policy interval in seconds; paper unspecified')
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
    args=parser.parse_args()
    if args.queue_capacity<1 or not 0<args.slot_duration<float('inf'):
        parser.error('queue-capacity and slot-duration must be finite and positive')
    try:
        scenario=Scenario(task=args.task,resource=args.resource,deadline=args.deadline,mode=mode,
            episodes=getattr(args,'episodes',600),seed_spec=getattr(args,'seeds','700-729'),
            train_task=getattr(args,'train_task','S'),train_resource=getattr(args,'train_resource',None),
            train_deadline=getattr(args,'train_deadline',None),checkpoint_override=getattr(args,'checkpoint_dir',None),
            experiment_protocol=args.experiment_protocol,model_seed=args.model_seed,
            normalization_mode=args.normalization_mode,normalization_config=args.normalization_config,
            paper_options=dict(device=args.device,resume_checkpoint=getattr(args,'resume_checkpoint',None),
                               queue_capacity=args.queue_capacity,slot_duration=args.slot_duration))
    except (ValueError,argparse.ArgumentTypeError) as error:parser.error(str(error))
    if args.dry_run:
        info=scenario.describe(ALGORITHM)
        info.update(algorithm=ALGORITHM,display_name=DISPLAY_NAME,model_version=MODEL_VERSION,reward_protocol=REWARD_PROTOCOL,
                    queue=dict(capacity=args.queue_capacity,slot_seconds=args.slot_duration,discipline='FIFO'),
                    retained_deadline_context=False,fixed_Q_head=True,entropy=False)
        print(json.dumps(info,indent=2,ensure_ascii=False));raise SystemExit(0)
    scenario.validate_calibration()
    from common.f22_cache import check_cache
    check_cache(scenario)
    return scenario
