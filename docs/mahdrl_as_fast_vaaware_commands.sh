#!/usr/bin/env bash
# Source from the repository root; this file only defines functions.

# Required: resource deadline. Optional: preview(on/off) HA-period model-seed.
train_mahdrl_as_fast_vaaware() {
    local res="$1" ddl="$2"
    local preview="${3:-on}" period="${4:-4}" seed="${5:-1}"
    local preview_flag="--va-preview" p=1
    case "$preview" in
        on) ;;
        off) preview_flag="--no-va-preview"; p=0 ;;
        *) printf 'preview must be on/off\n' >&2; return 2 ;;
    esac
    local prefix="outputs/server_train_logs/train_vaaware_S${res}_${ddl}_p${p}h${period}_sr092_m${seed}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_train_logs

    CUDA_VISIBLE_DEVICES=${GPU:-0} nohup bash -c '
        res="$1"
        ddl="$2"
        preview_flag="$3"
        period="$4"
        seed="$5"

        echo "[START] train=S${res}/${ddl} preview=${preview_flag} HA-period=${period} model=${seed}"
        python -u launch.py \
            algorithms/mahdrl_as_fast_vaaware/train.py \
            --task S --resource "$res" --deadline "$ddl" \
            --episodes 600 --model-seed "$seed" \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            "$preview_flag" --ha-update-period "$period" \
            --diagnostics-every 100 \
            --profile-mode sampled --profile-every 100 \
            || exit 1
        echo "[DONE] train=S${res}/${ddl} model=${seed}"
    ' mahdrl_as_fast_vaaware_train "$res" "$ddl" "$preview_flag" "$period" "$seed" \
        > "${prefix}.log" 2>&1 < /dev/null &

    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}

# Required: resource deadline. Optional: targets preview(on/off) HA-period model-seed.
eval_mahdrl_as_fast_vaaware_all() {
    local res="$1" ddl="$2"
    local targets="${3:-S M L}"
    local preview="${4:-on}" period="${5:-4}" seed="${6:-1}"
    local preview_flag="--va-preview" p=1
    case "$preview" in
        on) ;;
        off) preview_flag="--no-va-preview"; p=0 ;;
        *) printf 'preview must be on/off\n' >&2; return 2 ;;
    esac
    local prefix="outputs/server_eval_logs/eval_vaaware_S${res}_${ddl}_targets${targets// /}_p${p}h${period}_sr092_s700-729_m${seed}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_eval_logs

    CUDA_VISIBLE_DEVICES=${GPU:-0} nohup bash -c '
        res="$1"
        ddl="$2"
        targets="$3"
        preview_flag="$4"
        period="$5"
        seed="$6"

        for task in $targets; do
            echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"

            python -u launch.py \
                algorithms/mahdrl_as_fast_vaaware/evaluate.py \
                --task "$task" --resource "$res" --deadline "$ddl" \
                --train-task S --train-resource "$res" \
                --train-deadline "$ddl" \
                --seeds 700-729 --model-seed "$seed" \
                --experiment-protocol configs/f22_split_v1.json \
                --normalization-mode f19_shared_v1 --device cuda \
                "$preview_flag" --ha-update-period "$period" \
                --diagnostics-every 100 \
                --profile-mode sampled --profile-every 100 \
                || exit 1

            echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
        done

        echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
    ' mahdrl_as_fast_vaaware_eval "$res" "$ddl" "$targets" "$preview_flag" "$period" "$seed" \
        > "${prefix}.log" 2>&1 < /dev/null &

    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}

# Run these explicitly after sourcing. Default: preview on, HA period 4, model seed 1.
# train_mahdrl_as_fast_vaaware S Tight
# train_mahdrl_as_fast_vaaware S Medium
# train_mahdrl_as_fast_vaaware S Loose
# train_mahdrl_as_fast_vaaware M Tight
# train_mahdrl_as_fast_vaaware M Medium
# train_mahdrl_as_fast_vaaware M Loose
# train_mahdrl_as_fast_vaaware L Tight
# train_mahdrl_as_fast_vaaware L Medium
# train_mahdrl_as_fast_vaaware L Loose
#
# SS source: targets SS, MS, LS
# eval_mahdrl_as_fast_vaaware_all S Tight
# eval_mahdrl_as_fast_vaaware_all S Medium
# eval_mahdrl_as_fast_vaaware_all S Loose
# SM source: targets SM, MM, LM
# eval_mahdrl_as_fast_vaaware_all M Tight
# eval_mahdrl_as_fast_vaaware_all M Medium
# eval_mahdrl_as_fast_vaaware_all M Loose
# SL source: targets SL, ML, LL
# eval_mahdrl_as_fast_vaaware_all L Tight
# eval_mahdrl_as_fast_vaaware_all L Medium
# eval_mahdrl_as_fast_vaaware_all L Loose
