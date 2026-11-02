#!/usr/bin/env bash
# Source from the repository root. Defines functions; starts no jobs.
# Fixed scientific settings: model seed 1, preview on, HA period 4, strict SR > 0.92.
# Optional seed cycles: train RESOURCE DEADLINE 10; eval RESOURCE DEADLINE "S M L" 10.

train_mahdrl_as_fast_hscore_speed() {
    local res="${1:-}" ddl="${2:-}"
    local cycles="${3:-1}"
    [[ "$cycles" =~ ^[1-9][0-9]{0,5}$ ]] || { printf 'seed cycles must be 1-999999\n' >&2; return 2; }
    local episodes=$((600 * cycles))
    case "$res" in S|M|L) ;; *) printf 'resource must be S/M/L\n' >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) printf 'deadline must be Tight/Medium/Loose\n' >&2; return 2 ;; esac
    local prefix="outputs/server_train_logs/train_hscore_speed_S${res}_${ddl}_e${episodes}r${cycles}_p1h4_srgt092_m1_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_train_logs || return

    CUDA_VISIBLE_DEVICES=0 nohup bash -c '
        res="$1"
        ddl="$2"
        cycles="$3"
        episodes="$4"

        echo "[START] train=S${res}/${ddl}"
        python -u launch.py \
            algorithms/mahdrl_as_fast_hscore_speed/train.py \
            --task S --resource "$res" --deadline "$ddl" \
            --episodes "$episodes" --train-seed-cycles "$cycles" --model-seed 1 \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --va-preview --ha-update-period 4 \
            --diagnostics-every 100 \
            --profile-mode sampled --profile-every 100 \
            || exit 1
        echo "[DONE] train=S${res}/${ddl}"
    ' mahdrl_as_fast_hscore_speed_train "$res" "$ddl" "$cycles" "$episodes" \
        > "${prefix}.log" 2>&1 < /dev/null &

    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}

eval_mahdrl_as_fast_hscore_speed_all() {
    local res="${1:-}" ddl="${2:-}"
    local targets="${3:-S M L}" target
    local cycles="${4:-1}"
    [[ "$cycles" =~ ^[1-9][0-9]{0,5}$ ]] || { printf 'seed cycles must be 1-999999\n' >&2; return 2; }
    local episodes=$((600 * cycles))
    case "$res" in S|M|L) ;; *) printf 'resource must be S/M/L\n' >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) printf 'deadline must be Tight/Medium/Loose\n' >&2; return 2 ;; esac
    for target in $targets; do
        case "$target" in S|M|L) ;; *) printf 'targets must contain only S/M/L\n' >&2; return 2 ;; esac
    done
    local prefix="outputs/server_eval_logs/eval_hscore_speed_sourceS${res}_${ddl}_targets${targets// /}_e${episodes}r${cycles}_p1h4_srgt092_s700-729_m1_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_eval_logs || return

    CUDA_VISIBLE_DEVICES=0 nohup bash -c '
        res="$1"
        ddl="$2"
        targets="$3"
        cycles="$4"
        episodes="$5"

        for task in $targets; do
            echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"
            python -u launch.py \
                algorithms/mahdrl_as_fast_hscore_speed/evaluate.py \
                --task "$task" --resource "$res" --deadline "$ddl" \
                --train-task S --train-resource "$res" \
                --train-deadline "$ddl" \
                --seeds 700-729 --model-seed 1 \
                --episodes "$episodes" --train-seed-cycles "$cycles" \
                --experiment-protocol configs/f22_split_v1.json \
                --normalization-mode f19_shared_v1 --device cuda \
                --va-preview --ha-update-period 4 \
                --diagnostics-every 100 \
                --profile-mode sampled --profile-every 100 \
                || exit 1
            echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
        done
        echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
    ' mahdrl_as_fast_hscore_speed_eval "$res" "$ddl" "$targets" "$cycles" "$episodes" \
        > "${prefix}.log" 2>&1 < /dev/null &

    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}
