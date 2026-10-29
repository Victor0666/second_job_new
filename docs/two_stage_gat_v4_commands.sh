#!/usr/bin/env bash
# Source from the repository root. Loading this file starts no jobs.

# Required: resource deadline. Optional: model-seed (default 1).
train_two_stage_gat() {
    local res="$1" ddl="$2"
    local seed="${3:-1}"
    local prefix="outputs/server_train_logs/train_2sd_v4_S${res}_${ddl}_sr092_m${seed}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_train_logs || return 1

    CUDA_VISIBLE_DEVICES=${GPU:-0} nohup bash -c '
        res="$1"
        ddl="$2"
        seed="$3"

        echo "[START] train=S${res}/${ddl} model=${seed}"
        python -u launch.py \
            algorithms/two_stage_gat/train.py \
            --task S --resource "$res" --deadline "$ddl" \
            --episodes 600 --model-seed "$seed" \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            || exit 1
        echo "[DONE] train=S${res}/${ddl} model=${seed}"
    ' two_stage_gat_train "$res" "$ddl" "$seed" \
        > "${prefix}.log" 2>&1 < /dev/null &

    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}

# Required: resource deadline. Optional: targets (default S M L), model-seed.
eval_two_stage_gat_all() {
    local res="$1" ddl="$2"
    local targets="${3:-S M L}" seed="${4:-1}"
    local prefix="outputs/server_eval_logs/eval_2sd_v4_S${res}_${ddl}_targets${targets// /}_s700-729_m${seed}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_eval_logs || return 1

    CUDA_VISIBLE_DEVICES=${GPU:-0} nohup bash -c '
        res="$1"
        ddl="$2"
        targets="$3"
        seed="$4"

        for task in $targets; do
            echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"
            python -u launch.py \
                algorithms/two_stage_gat/evaluate.py \
                --task "$task" --resource "$res" --deadline "$ddl" \
                --train-task S --train-resource "$res" \
                --train-deadline "$ddl" \
                --seeds 700-729 --model-seed "$seed" \
                --experiment-protocol configs/f22_split_v1.json \
                --normalization-mode f19_shared_v1 --device cuda \
                || exit 1
            echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
        done
        echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
    ' two_stage_gat_eval "$res" "$ddl" "$targets" "$seed" \
        > "${prefix}.log" 2>&1 < /dev/null &

    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}

# Explicit calls after sourcing; each training call starts one 600-episode job.
# train_two_stage_gat S Tight
# train_two_stage_gat S Medium
# train_two_stage_gat S Loose
# train_two_stage_gat M Tight
# train_two_stage_gat M Medium
# train_two_stage_gat M Loose
# train_two_stage_gat L Tight
# train_two_stage_gat L Medium
# train_two_stage_gat L Loose
#
# SS source: SS, MS, LS.
# eval_two_stage_gat_all S Tight
# eval_two_stage_gat_all S Medium
# eval_two_stage_gat_all S Loose
# SM source: SM, MM, LM.
# eval_two_stage_gat_all M Tight
# eval_two_stage_gat_all M Medium
# eval_two_stage_gat_all M Loose
# SL source: SL, ML, LL.
# eval_two_stage_gat_all L Tight
# eval_two_stage_gat_all L Medium
# eval_two_stage_gat_all L Loose
