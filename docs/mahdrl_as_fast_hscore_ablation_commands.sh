#!/usr/bin/env bash
# Source from the project root. Defines functions only; starts no jobs.
# GPU=0 by default; model seed 1. Existing full/TL functions retain their interface.
source docs/mahdrl_as_fast_hscore_commands.sh
source docs/mahdrl_as_fast_hscore_tl_commands.sh

_hscore_ablation_train() {
    local suffix="$1" res="$2" ddl="$3" seed="${4:-1}"
    local prefix="outputs/server_train_logs/hs_${suffix}_S${res}_${ddl}_srgt092_m${seed}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_train_logs
    CUDA_VISIBLE_DEVICES=${GPU:-0} nohup bash -c '
        suffix="$1"; res="$2"; ddl="$3"; seed="$4"
        echo "[START] variant=$suffix source=S${res}/${ddl}"
        python -u launch.py "algorithms/mahdrl_as_fast_hscore_${suffix}/train.py" \
            --task S --resource "$res" --deadline "$ddl" \
            --episodes 600 --model-seed "$seed" \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --diagnostics-every 100 --profile-mode sampled --profile-every 100 \
            || exit 1
        echo "[DONE] variant=$suffix source=S${res}/${ddl}"
    ' hscore_train "$suffix" "$res" "$ddl" "$seed" > "${prefix}.log" 2>&1 < /dev/null &
    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}

_hscore_ablation_eval() {
    local suffix="$1" res="$2" ddl="$3" targets="${4:-S M L}" seed="${5:-1}"
    local prefix="outputs/server_eval_logs/hs_${suffix}_S${res}_${ddl}_to${targets// /}_srgt092_s700-729_m${seed}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    mkdir -p outputs/server_eval_logs
    CUDA_VISIBLE_DEVICES=${GPU:-0} nohup bash -c '
        suffix="$1"; res="$2"; ddl="$3"; targets="$4"; seed="$5"
        for task in $targets; do
            echo "[START] variant=$suffix source=S${res}/${ddl} target=${task}${res}/${ddl}"
            python -u launch.py "algorithms/mahdrl_as_fast_hscore_${suffix}/evaluate.py" \
                --task "$task" --resource "$res" --deadline "$ddl" \
                --train-task S --train-resource "$res" --train-deadline "$ddl" \
                --seeds 700-729 --model-seed "$seed" \
                --experiment-protocol configs/f22_split_v1.json \
                --normalization-mode f19_shared_v1 --device cuda \
                --diagnostics-every 100 --profile-mode sampled --profile-every 100 \
                || exit 1
            echo "[DONE] variant=$suffix source=S${res}/${ddl} target=${task}${res}/${ddl}"
        done
        echo "[ALL_DONE] variant=$suffix source=S${res}/${ddl} targets=$targets"
    ' hscore_eval "$suffix" "$res" "$ddl" "$targets" "$seed" > "${prefix}.log" 2>&1 < /dev/null &
    echo $! > "${prefix}.pid"
    printf 'started: %s.log\n' "$prefix"
}

train_mahdrl_as_fast_hscore_nta() { _hscore_ablation_train nta "$@"; }
eval_mahdrl_as_fast_hscore_nta_all() { _hscore_ablation_eval nta "$@"; }

train_mahdrl_as_fast_hscore_nha() { _hscore_ablation_train nha "$@"; }
eval_mahdrl_as_fast_hscore_nha_all() { _hscore_ablation_eval nha "$@"; }

train_mahdrl_as_fast_hscore_nva() { _hscore_ablation_train nva "$@"; }
eval_mahdrl_as_fast_hscore_nva_all() { _hscore_ablation_eval nva "$@"; }

train_mahdrl_as_fast_hscore_sync_task() { _hscore_ablation_train sync_task "$@"; }
eval_mahdrl_as_fast_hscore_sync_task_all() { _hscore_ablation_eval sync_task "$@"; }

train_mahdrl_as_fast_hscore_sync_phase() { _hscore_ablation_train sync_phase "$@"; }
eval_mahdrl_as_fast_hscore_sync_phase_all() { _hscore_ablation_eval sync_phase "$@"; }

train_mahdrl_as_fast_hscore_ertf() { _hscore_ablation_train ertf "$@"; }
eval_mahdrl_as_fast_hscore_ertf_all() { _hscore_ablation_eval ertf "$@"; }

train_mahdrl_as_fast_hscore_edf() { _hscore_ablation_train edf "$@"; }
eval_mahdrl_as_fast_hscore_edf_all() { _hscore_ablation_eval edf "$@"; }

train_mahdrl_as_fast_hscore_cp() { _hscore_ablation_train cp "$@"; }
eval_mahdrl_as_fast_hscore_cp_all() { _hscore_ablation_eval cp "$@"; }

train_mahdrl_as_fast_hscore_r090() { _hscore_ablation_train r090 "$@"; }
eval_mahdrl_as_fast_hscore_r090_all() { _hscore_ablation_eval r090 "$@"; }

train_mahdrl_as_fast_hscore_r010_nopf() { _hscore_ablation_train r010_nopf "$@"; }
eval_mahdrl_as_fast_hscore_r010_nopf_all() { _hscore_ablation_eval r010_nopf "$@"; }
