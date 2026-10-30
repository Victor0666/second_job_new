#!/usr/bin/env bash
# Source from the repository root. Loading this file never starts a job.

_vaaware_resource_tl_validate() {
    case "$1" in R100|R150) ;; *) printf 'resource must be R100 or R150\n' >&2; return 2 ;; esac
    case "$2" in Tight|Medium|Loose) ;; *) printf 'deadline must be Tight/Medium/Loose\n' >&2; return 2 ;; esac
    [[ "$3" =~ ^[0-9]+$ ]] || { printf 'model seed must be a nonnegative integer\n' >&2; return 2; }
}

train_mahdrl_as_fast_vaaware_resource_tl() {
    local res="$1" ddl="${2:-}" seed="${3:-1}"
    [[ $# -le 3 ]] || { printf 'usage: train resource deadline [model-seed]\n' >&2; return 2; }
    _vaaware_resource_tl_validate "$res" "$ddl" "$seed" || return
    local scale="${res#R}"
    local prefix="outputs/server_train_logs/train_va${scale}_tl_sourceS${scale}_${ddl}_m${seed}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_train_logs || return
    CUDA_VISIBLE_DEVICES=${GPU:-0} nohup bash -c '
        scale="$1"; ddl="$2"; seed="$3"
        echo "[START] VAware-R${scale}-TL source=S/${ddl} model=${seed}"
        python -u launch.py "algorithms/mahdrl_as_fast_vaaware_r${scale}_tl/train.py" \
            --task S --deadline "$ddl" --episodes 600 --model-seed "$seed" \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --diagnostics-every 100 --profile-mode sampled --profile-every 100 || exit 1
        echo "[DONE] VAware-R${scale}-TL source=S/${ddl} model=${seed}"
    ' vaaware_resource_tl_train "$scale" "$ddl" "$seed" \
        > "${prefix}.log" 2>&1 < /dev/null &
    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}

eval_mahdrl_as_fast_vaaware_resource_tl_all() {
    local res="$1" ddl="${2:-}" targets="${3:-S M L}" seed="${4:-1}"
    [[ $# -le 4 ]] || { printf 'usage: eval resource deadline [targets] [model-seed]\n' >&2; return 2; }
    _vaaware_resource_tl_validate "$res" "$ddl" "$seed" || return
    local scale="${res#R}" task found=0
    for task in $targets; do
        case "$task" in S|M|L) ;; *) printf 'targets must contain only S/M/L\n' >&2; return 2 ;; esac
        found=1
    done
    (( found )) || { printf 'targets must not be empty\n' >&2; return 2; }
    local prefix="outputs/server_eval_logs/eval_va${scale}_tl_sourceS${scale}_${ddl}_targets${targets// /}_s700-729_m${seed}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_eval_logs || return
    CUDA_VISIBLE_DEVICES=${GPU:-0} nohup bash -c '
        scale="$1"; ddl="$2"; targets="$3"; seed="$4"
        for task in $targets; do
            echo "[START] VAware-R${scale}-TL source=S/${ddl} target=${task} model=${seed}"
            python -u launch.py "algorithms/mahdrl_as_fast_vaaware_r${scale}_tl/evaluate.py" \
                --task "$task" --deadline "$ddl" --train-task S --train-deadline "$ddl" \
                --seeds 700-729 --model-seed "$seed" \
                --experiment-protocol configs/f22_split_v1.json \
                --normalization-mode f19_shared_v1 --device cuda \
                --diagnostics-every 100 --profile-mode sampled --profile-every 100 || exit 1
            echo "[DONE] VAware-R${scale}-TL target=${task} model=${seed}"
        done
        echo "[ALL_DONE] VAware-R${scale}-TL source=S/${ddl} model=${seed}"
    ' vaaware_resource_tl_eval "$scale" "$ddl" "$targets" "$seed" \
        > "${prefix}.log" 2>&1 < /dev/null &
    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}

# Train: deadline, optional model-seed. Eval: deadline, optional targets, model-seed.
train_mahdrl_as_fast_vaaware_r100_tl() { train_mahdrl_as_fast_vaaware_resource_tl R100 "$@"; }
train_mahdrl_as_fast_vaaware_r150_tl() { train_mahdrl_as_fast_vaaware_resource_tl R150 "$@"; }
eval_mahdrl_as_fast_vaaware_r100_tl_all() { eval_mahdrl_as_fast_vaaware_resource_tl_all R100 "$@"; }
eval_mahdrl_as_fast_vaaware_r150_tl_all() { eval_mahdrl_as_fast_vaaware_resource_tl_all R150 "$@"; }

# Explicitly start desired jobs after sourcing; fit concurrency to available hardware.
# train_mahdrl_as_fast_vaaware_resource_tl R100 Tight
# train_mahdrl_as_fast_vaaware_resource_tl R100 Medium
# train_mahdrl_as_fast_vaaware_resource_tl R100 Loose
# train_mahdrl_as_fast_vaaware_resource_tl R150 Tight
# train_mahdrl_as_fast_vaaware_resource_tl R150 Medium
# train_mahdrl_as_fast_vaaware_resource_tl R150 Loose
# eval_mahdrl_as_fast_vaaware_resource_tl_all R100 Tight
# eval_mahdrl_as_fast_vaaware_resource_tl_all R100 Medium
# eval_mahdrl_as_fast_vaaware_resource_tl_all R100 Loose
# eval_mahdrl_as_fast_vaaware_resource_tl_all R150 Tight
# eval_mahdrl_as_fast_vaaware_resource_tl_all R150 Medium
# eval_mahdrl_as_fast_vaaware_resource_tl_all R150 Loose
