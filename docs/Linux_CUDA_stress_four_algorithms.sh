#!/usr/bin/env bash
# Source this file from the repository root. Defining functions starts no jobs.

_stress_train() {
    local algorithm="$1" tag="$2" res="${3:-S}" ddl="${4:-Tight}"
    case "$res" in S|M|L) ;; *) echo 'resource must be S, M or L' >&2; return 2 ;; esac
    [[ "$ddl" == Tight ]] || { echo 'stress profile uses Tight source scales only' >&2; return 2; }
    local prefix="outputs/server_train_logs/train_${tag}_sourceS${res}_${ddl}_a1l006r003_sr092_e600_m1_$(date +%Y%m%d_%H%M%S)_${BASHPID}"
    mkdir -p outputs/server_train_logs || return

    CUDA_VISIBLE_DEVICES=0 nohup bash -c '
        algorithm="$1"
        res="$2"
        ddl="$3"
        extra=()
        if [[ "$algorithm" != marl ]]; then extra+=(--device cuda); fi
        if [[ "$algorithm" == mahdrl_oe_ge_apf_score_fast ]]; then
            extra+=(--diagnostics-every 100 --profile-mode sampled --profile-every 100)
        fi

        echo "[START] train=$algorithm source=S${res}/${ddl} workload=stress_a1_l006_ref003"
        python -u launch.py "algorithms/${algorithm}/train.py" \
            --task S --resource "$res" --deadline "$ddl" \
            --episodes 600 --model-seed 1 \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 \
            --workload-profile stress_a1_l006_ref003 \
            "${extra[@]}" || exit 1
        echo "[ALL_DONE] train=$algorithm source=S${res}/${ddl}"
    ' stress_train "$algorithm" "$res" "$ddl" > "${prefix}.log" 2>&1 &

    local pid=$!
    echo "$pid" > "${prefix}.pid"
    echo "[PID] $pid [LOG] ${prefix}.log"
}

_stress_eval_all() {
    local algorithm="$1" tag="$2" res="${3:-S}" ddl="${4:-Tight}" targets="${5:-S M L}"
    case "$res" in S|M|L) ;; *) echo 'resource must be S, M or L' >&2; return 2 ;; esac
    [[ "$ddl" == Tight ]] || { echo 'stress profile uses Tight source scales only' >&2; return 2; }
    local task
    for task in $targets; do
        case "$task" in S|M|L) ;; *) echo 'targets must contain only S, M or L' >&2; return 2 ;; esac
    done
    local prefix="outputs/server_eval_logs/eval_${tag}_sourceS${res}_${ddl}_targets${targets// /}_a1l006r003_sr092_s700-729_m1_$(date +%Y%m%d_%H%M%S)_${BASHPID}"
    mkdir -p outputs/server_eval_logs || return

    CUDA_VISIBLE_DEVICES=0 nohup bash -c '
        algorithm="$1"
        res="$2"
        ddl="$3"
        targets="$4"
        extra=(--device cuda)
        if [[ "$algorithm" == mahdrl_oe_ge_apf_score_fast ]]; then
            extra+=(--diagnostics-every 100 --profile-mode sampled --profile-every 100)
        fi

        for task in $targets; do
            echo "[START] eval=$algorithm source=S${res}/${ddl} target=${task}${res}/${ddl}"
            python -u launch.py "algorithms/${algorithm}/evaluate.py" \
                --task "$task" --resource "$res" --deadline "$ddl" \
                --train-task S --train-resource "$res" --train-deadline "$ddl" \
                --seeds 700-729 --model-seed 1 \
                --experiment-protocol configs/f22_split_v1.json \
                --normalization-mode f19_shared_v1 \
                --workload-profile stress_a1_l006_ref003 \
                "${extra[@]}" || exit 1
            echo "[DONE] eval=$algorithm source=S${res}/${ddl} target=${task}${res}/${ddl}"
        done
        echo "[ALL_DONE] eval=$algorithm source=S${res}/${ddl} targets=${targets}"
    ' stress_eval "$algorithm" "$res" "$ddl" "$targets" > "${prefix}.log" 2>&1 &

    local pid=$!
    echo "$pid" > "${prefix}.pid"
    echo "[PID] $pid [LOG] ${prefix}.log"
}

train_marl_q42_stress() { _stress_train marl marl_q42 "$@"; }
train_pd3qn_fe_stress() { _stress_train pd3qn_fe pd3qn_fe "$@"; }
train_irws_r_fe_idle_stress() { _stress_train irws_r_fe_idle irws_r_fe_idle "$@"; }
train_mahdrl_as_fast_noapf_v3_stress() { _stress_train mahdrl_oe_ge_apf_score_fast mahdrl_as_fast_noapf_v3 "$@"; }
# Historical function names follow the current NoAPF entry; APF V2 weights remain incompatible.
train_mahdrl_as_fast_v2_stress() { train_mahdrl_as_fast_noapf_v3_stress "$@"; }

eval_marl_q42_stress_all() { _stress_eval_all marl marl_q42 "$@"; }
eval_pd3qn_fe_stress_all() { _stress_eval_all pd3qn_fe pd3qn_fe "$@"; }
eval_irws_r_fe_idle_stress_all() { _stress_eval_all irws_r_fe_idle irws_r_fe_idle "$@"; }
eval_mahdrl_as_fast_noapf_v3_stress_all() { _stress_eval_all mahdrl_oe_ge_apf_score_fast mahdrl_as_fast_noapf_v3 "$@"; }
eval_mahdrl_as_fast_v2_stress_all() { eval_mahdrl_as_fast_noapf_v3_stress_all "$@"; }
