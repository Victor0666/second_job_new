#!/usr/bin/env bash
# Source from the repository root; this file only defines functions.

# Required: resource deadline. Optional: model-seed.
train_mahdrl_as_fast_hscore_tl() {
    local res="$1" ddl="$2"
    local seed="${3:-1}"
    local prefix="outputs/server_train_logs/train_hscore_tl_S${res}_${ddl}_sr092_m${seed}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_train_logs

    CUDA_VISIBLE_DEVICES=${GPU:-0} nohup bash -c '
        res="$1"
        ddl="$2"
        seed="$3"

        echo "[START] train=S${res}/${ddl} model=${seed}"
        python -u launch.py \
            algorithms/mahdrl_as_fast_hscore_tl/train.py \
            --task S --resource "$res" --deadline "$ddl" \
            --episodes 600 --model-seed "$seed" \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --diagnostics-every 100 \
            --profile-mode sampled --profile-every 100 \
            || exit 1
        echo "[DONE] train=S${res}/${ddl} model=${seed}"
        echo "[ALL_DONE] train=S${res}/${ddl} model=${seed}"
    ' mahdrl_as_fast_hscore_tl_train "$res" "$ddl" "$seed" \
        > "${prefix}.log" 2>&1 < /dev/null &

    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}

# Required: resource deadline. Optional: targets model-seed.
eval_mahdrl_as_fast_hscore_tl_all() {
    local res="$1" ddl="$2"
    local targets="${3:-S M L}"
    local seed="${4:-1}"
    local prefix="outputs/server_eval_logs/eval_hscore_tl_S${res}_${ddl}_targets${targets// /}_sr092_s700-729_m${seed}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_eval_logs

    CUDA_VISIBLE_DEVICES=${GPU:-0} nohup bash -c '
        res="$1"
        ddl="$2"
        targets="$3"
        seed="$4"

        for task in $targets; do
            echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"

            python -u launch.py \
                algorithms/mahdrl_as_fast_hscore_tl/evaluate.py \
                --task "$task" --resource "$res" --deadline "$ddl" \
                --train-task S --train-resource "$res" \
                --train-deadline "$ddl" \
                --seeds 700-729 --model-seed "$seed" \
                --experiment-protocol configs/f22_split_v1.json \
                --normalization-mode f19_shared_v1 --device cuda \
                --diagnostics-every 100 \
                --profile-mode sampled --profile-every 100 \
                || exit 1

            echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
        done

        echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
    ' mahdrl_as_fast_hscore_tl_eval "$res" "$ddl" "$targets" "$seed" \
        > "${prefix}.log" 2>&1 < /dev/null &

    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}

# Run these explicitly after sourcing. Default: model seed 1; no HA or preview controls.
# train_mahdrl_as_fast_hscore_tl S Tight
# train_mahdrl_as_fast_hscore_tl S Medium
# train_mahdrl_as_fast_hscore_tl S Loose
# train_mahdrl_as_fast_hscore_tl M Tight
# train_mahdrl_as_fast_hscore_tl M Medium
# train_mahdrl_as_fast_hscore_tl M Loose
# train_mahdrl_as_fast_hscore_tl L Tight
# train_mahdrl_as_fast_hscore_tl L Medium
# train_mahdrl_as_fast_hscore_tl L Loose
#
# SS source: targets SS, MS, LS
# eval_mahdrl_as_fast_hscore_tl_all S Tight
# eval_mahdrl_as_fast_hscore_tl_all S Medium
# eval_mahdrl_as_fast_hscore_tl_all S Loose
# SM source: targets SM, MM, LM
# eval_mahdrl_as_fast_hscore_tl_all M Tight
# eval_mahdrl_as_fast_hscore_tl_all M Medium
# eval_mahdrl_as_fast_hscore_tl_all M Loose
# SL source: targets SL, ML, LL
# eval_mahdrl_as_fast_hscore_tl_all L Tight
# eval_mahdrl_as_fast_hscore_tl_all L Medium
# eval_mahdrl_as_fast_hscore_tl_all L Loose
