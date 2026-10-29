#!/usr/bin/env bash
# Source from the repository root. These functions define jobs; sourcing starts nothing.

train_drl_gnn_fast_idle() {
    local res="$1" ddl="$2"
    local prefix="outputs/server_train_logs/train_drl_idle_S${res}_${ddl}_v6_sr092_m1_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_train_logs

    CUDA_VISIBLE_DEVICES=0 nohup bash -c '
        res="$1"
        ddl="$2"

        echo "[START] train=S${res}/${ddl} v6 sr092 model1"

        python -u launch.py \
            algorithms/drl_gnn_fast_idle/train.py \
            --task S --resource "$res" --deadline "$ddl" \
            --episodes 600 --model-seed 1 \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --graphs-per-backward 4 \
            || exit 1

        echo "[DONE] train=S${res}/${ddl}"
    ' drl_gnn_fast_idle_train "$res" "$ddl" \
        > "${prefix}.log" 2>&1 < /dev/null &

    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}

eval_drl_gnn_fast_idle_all() {
    local res="$1" ddl="$2"
    local targets="${3:-S M L}"
    local prefix="outputs/server_eval_logs/eval_drl_idle_sourceS${res}_${ddl}_targets${targets// /}_v6_sr092_s700-729_m1_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_eval_logs

    CUDA_VISIBLE_DEVICES=0 nohup bash -c '
        res="$1"
        ddl="$2"
        targets="$3"

        for task in $targets; do
            echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"

            python -u launch.py \
                algorithms/drl_gnn_fast_idle/evaluate.py \
                --task "$task" --resource "$res" --deadline "$ddl" \
                --train-task S --train-resource "$res" \
                --train-deadline "$ddl" \
                --seeds 700-729 --model-seed 1 \
                --experiment-protocol configs/f22_split_v1.json \
                --normalization-mode f19_shared_v1 --device cuda \
                || exit 1

            echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
        done

        echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
    ' drl_gnn_fast_idle_eval "$res" "$ddl" "$targets" \
        > "${prefix}.log" 2>&1 < /dev/null &

    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}

# Train sources (run the desired lines after sourcing; do not start all by sourcing):
# train_drl_gnn_fast_idle S Tight
# train_drl_gnn_fast_idle S Medium
# train_drl_gnn_fast_idle S Loose
# train_drl_gnn_fast_idle M Tight
# train_drl_gnn_fast_idle M Medium
# train_drl_gnn_fast_idle M Loose
# train_drl_gnn_fast_idle L Tight
# train_drl_gnn_fast_idle L Medium
# train_drl_gnn_fast_idle L Loose

# SS sources -> SS, MS, LS:
# eval_drl_gnn_fast_idle_all S Tight
# eval_drl_gnn_fast_idle_all S Medium
# eval_drl_gnn_fast_idle_all S Loose
# SM sources -> SM, MM, LM:
# eval_drl_gnn_fast_idle_all M Tight
# eval_drl_gnn_fast_idle_all M Medium
# eval_drl_gnn_fast_idle_all M Loose
# SL sources -> SL, ML, LL:
# eval_drl_gnn_fast_idle_all L Tight
# eval_drl_gnn_fast_idle_all L Medium
# eval_drl_gnn_fast_idle_all L Loose
