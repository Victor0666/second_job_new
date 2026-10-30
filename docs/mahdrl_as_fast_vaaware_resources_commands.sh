#!/usr/bin/env bash
# Source from the repository root. Loading this file never starts a job.

_vaaware_resource_validate() {
    case "$1" in 100|150) ;; *) printf 'scale must be 100 or 150\n' >&2; return 2 ;; esac
    case "$2" in Tight|Medium|Loose) ;; *) printf 'deadline must be Tight/Medium/Loose\n' >&2; return 2 ;; esac
    [[ "$3" =~ ^[0-9]+$ ]] || { printf 'model seed must be a nonnegative integer\n' >&2; return 2; }
    case "$4" in on|off) ;; *) printf 'preview must be on/off\n' >&2; return 2 ;; esac
    [[ "$5" =~ ^[1-9][0-9]*$ ]] || { printf 'HA period must be a positive integer\n' >&2; return 2; }
}

train_mahdrl_as_fast_vaaware_resource() {
    local res="$1" ddl="${2:-}" seed="${3:-1}" preview="${4:-on}" period="${5:-4}"
    case "$res" in R100|R150) ;; *) printf 'resource must be R100 or R150\n' >&2; return 2 ;; esac
    local scale="${res#R}"
    _vaaware_resource_validate "$scale" "$ddl" "$seed" "$preview" "$period" || return
    local flag="--va-preview"
    [[ "$preview" == off ]] && flag="--no-va-preview"
    local prefix="outputs/server_train_logs/train_va${scale}_sourceS${scale}_${ddl}_p${preview}h${period}_m${seed}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_train_logs || return
    CUDA_VISIBLE_DEVICES=${GPU:-0} nohup bash -c '
        scale="$1"; ddl="$2"; seed="$3"; flag="$4"; period="$5"
        echo "[START] VAware-R${scale} source=S/${ddl} model=${seed}"
        python -u launch.py "algorithms/mahdrl_as_fast_vaaware_r${scale}/train.py" \
            --task S --deadline "$ddl" --episodes 600 --model-seed "$seed" \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --diagnostics-every 100 --profile-mode sampled --profile-every 100 \
            "$flag" --ha-update-period "$period" || exit 1
        echo "[DONE] VAware-R${scale} source=S/${ddl} model=${seed}"
    ' vaaware_resource_train "$scale" "$ddl" "$seed" "$flag" "$period" \
        > "${prefix}.log" 2>&1 < /dev/null &
    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}

eval_mahdrl_as_fast_vaaware_resource_all() {
    local res="$1" ddl="${2:-}" targets="${3:-S M L}" seed="${4:-1}" preview="${5:-on}" period="${6:-4}"
    case "$res" in R100|R150) ;; *) printf 'resource must be R100 or R150\n' >&2; return 2 ;; esac
    local scale="${res#R}"
    _vaaware_resource_validate "$scale" "$ddl" "$seed" "$preview" "$period" || return
    local task found=0 flag="--va-preview"
    for task in $targets; do
        case "$task" in S|M|L) ;; *) printf 'targets must contain only S/M/L\n' >&2; return 2 ;; esac
        found=1
    done
    (( found )) || { printf 'targets must not be empty\n' >&2; return 2; }
    [[ "$preview" == off ]] && flag="--no-va-preview"
    local prefix="outputs/server_eval_logs/eval_va${scale}_sourceS${scale}_${ddl}_targets${targets// /}_s700-729_p${preview}h${period}_m${seed}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_eval_logs || return
    CUDA_VISIBLE_DEVICES=${GPU:-0} nohup bash -c '
        scale="$1"; ddl="$2"; targets="$3"; seed="$4"; flag="$5"; period="$6"
        for task in $targets; do
            echo "[START] VAware-R${scale} source=S/${ddl} target=${task} model=${seed}"
            python -u launch.py "algorithms/mahdrl_as_fast_vaaware_r${scale}/evaluate.py" \
                --task "$task" --deadline "$ddl" --train-task S --train-deadline "$ddl" \
                --seeds 700-729 --model-seed "$seed" \
                --experiment-protocol configs/f22_split_v1.json \
                --normalization-mode f19_shared_v1 --device cuda \
                --diagnostics-every 100 --profile-mode sampled --profile-every 100 \
                "$flag" --ha-update-period "$period" || exit 1
            echo "[DONE] VAware-R${scale} target=${task} model=${seed}"
        done
        echo "[ALL_DONE] VAware-R${scale} source=S/${ddl} model=${seed}"
    ' vaaware_resource_eval "$scale" "$ddl" "$targets" "$seed" "$flag" "$period" \
        > "${prefix}.log" 2>&1 < /dev/null &
    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}

# Train arguments: deadline, optional model-seed, preview(on/off), HA-period.
train_mahdrl_as_fast_vaaware_r100() { train_mahdrl_as_fast_vaaware_resource R100 "$@"; }
train_mahdrl_as_fast_vaaware_r150() { train_mahdrl_as_fast_vaaware_resource R150 "$@"; }
# Evaluate arguments: deadline, optional targets, model-seed, preview, HA-period.
eval_mahdrl_as_fast_vaaware_r100_all() { eval_mahdrl_as_fast_vaaware_resource_all R100 "$@"; }
eval_mahdrl_as_fast_vaaware_r150_all() { eval_mahdrl_as_fast_vaaware_resource_all R150 "$@"; }

# Explicitly run the desired jobs after sourcing; schedule concurrency to fit hardware.
# train_mahdrl_as_fast_vaaware_r100 Tight
# train_mahdrl_as_fast_vaaware_r100 Medium
# train_mahdrl_as_fast_vaaware_r100 Loose
# train_mahdrl_as_fast_vaaware_r150 Tight
# train_mahdrl_as_fast_vaaware_r150 Medium
# train_mahdrl_as_fast_vaaware_r150 Loose
# eval_mahdrl_as_fast_vaaware_r100_all Tight
# eval_mahdrl_as_fast_vaaware_r100_all Medium
# eval_mahdrl_as_fast_vaaware_r100_all Loose
# eval_mahdrl_as_fast_vaaware_r150_all Tight
# eval_mahdrl_as_fast_vaaware_r150_all Medium
# eval_mahdrl_as_fast_vaaware_r150_all Loose
