#!/usr/bin/env bash
# 在项目根目录激活 Python/CUDA 环境后：source docs/mahdrl_as_fast_ablations_evaluate.sh
# 本文件只定义函数，不自动启动任何训练或评测。
# 默认 GPU=0；可在调用前设置 GPU=1。每次函数调用立即后台启动一个作业。
# 训练9个源模型；每个评测作业依次运行对应 S/M/L 目标，共27个场景。
# 旧权重不能用于本次 noAPF/SR>0.92 配置；先完成相应源模型训练再评测。
# 日志/PID 每次调用独立命名；单次作业失败立即退出，不继续后续目标。
# 最终 CSV 位于 outputs/run/<变体短标识>/，场景直接写入 CSV 文件名。

# NTA：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
eval_mahdrl_as_fast_nta_all() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local targets="${3:-S M L}"
    local task
    for task in $targets; do
        case "$task" in S|M|L) ;; *) echo "targets 只能包含 S M L" >&2; return 2 ;; esac
    done
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_eval_logs/eval_mahdrl_as_fast_nta_sourceS${res}_${ddl}_targets${targets// /}_noapf_select_srgt092_minenergy_v1_s700-729_m1_${run_id}"
    mkdir -p outputs/server_eval_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    targets="$3"

    for task in $targets; do
        echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"

        python -u launch.py \
            algorithms/mahdrl_oe_ge_apf_score_fast_nta/evaluate.py \
            --task "$task" --resource "$res" --deadline "$ddl" \
            --train-task S --train-resource "$res" \
            --train-deadline "$ddl" \
            --seeds 700-729 --model-seed 1 \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --diagnostics-every 100 \
            --profile-mode sampled --profile-every 100 \
            || exit 1

        echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
    done

    echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
' mahdrl_as_fast_nta_eval "$res" "$ddl" "$targets" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：评测 SS、MS、LS
# eval_mahdrl_as_fast_nta_all S Tight
# eval_mahdrl_as_fast_nta_all S Medium
# eval_mahdrl_as_fast_nta_all S Loose
# SM源模型：评测 SM、MM、LM
# eval_mahdrl_as_fast_nta_all M Tight
# eval_mahdrl_as_fast_nta_all M Medium
# eval_mahdrl_as_fast_nta_all M Loose
# SL源模型：评测 SL、ML、LL
# eval_mahdrl_as_fast_nta_all L Tight
# eval_mahdrl_as_fast_nta_all L Medium
# eval_mahdrl_as_fast_nta_all L Loose


# NHA：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
eval_mahdrl_as_fast_nha_all() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local targets="${3:-S M L}"
    local task
    for task in $targets; do
        case "$task" in S|M|L) ;; *) echo "targets 只能包含 S M L" >&2; return 2 ;; esac
    done
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_eval_logs/eval_mahdrl_as_fast_nha_sourceS${res}_${ddl}_targets${targets// /}_noapf_select_srgt092_minenergy_v1_s700-729_m1_${run_id}"
    mkdir -p outputs/server_eval_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    targets="$3"

    for task in $targets; do
        echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"

        python -u launch.py \
            algorithms/mahdrl_oe_ge_apf_score_fast_nha/evaluate.py \
            --task "$task" --resource "$res" --deadline "$ddl" \
            --train-task S --train-resource "$res" \
            --train-deadline "$ddl" \
            --seeds 700-729 --model-seed 1 \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --diagnostics-every 100 \
            --profile-mode sampled --profile-every 100 \
            || exit 1

        echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
    done

    echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
' mahdrl_as_fast_nha_eval "$res" "$ddl" "$targets" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：评测 SS、MS、LS
# eval_mahdrl_as_fast_nha_all S Tight
# eval_mahdrl_as_fast_nha_all S Medium
# eval_mahdrl_as_fast_nha_all S Loose
# SM源模型：评测 SM、MM、LM
# eval_mahdrl_as_fast_nha_all M Tight
# eval_mahdrl_as_fast_nha_all M Medium
# eval_mahdrl_as_fast_nha_all M Loose
# SL源模型：评测 SL、ML、LL
# eval_mahdrl_as_fast_nha_all L Tight
# eval_mahdrl_as_fast_nha_all L Medium
# eval_mahdrl_as_fast_nha_all L Loose


# NVA：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
eval_mahdrl_as_fast_nva_all() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local targets="${3:-S M L}"
    local task
    for task in $targets; do
        case "$task" in S|M|L) ;; *) echo "targets 只能包含 S M L" >&2; return 2 ;; esac
    done
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_eval_logs/eval_mahdrl_as_fast_nva_sourceS${res}_${ddl}_targets${targets// /}_noapf_select_srgt092_minenergy_v1_s700-729_m1_${run_id}"
    mkdir -p outputs/server_eval_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    targets="$3"

    for task in $targets; do
        echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"

        python -u launch.py \
            algorithms/mahdrl_oe_ge_apf_score_fast_nva/evaluate.py \
            --task "$task" --resource "$res" --deadline "$ddl" \
            --train-task S --train-resource "$res" \
            --train-deadline "$ddl" \
            --seeds 700-729 --model-seed 1 \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --diagnostics-every 100 \
            --profile-mode sampled --profile-every 100 \
            || exit 1

        echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
    done

    echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
' mahdrl_as_fast_nva_eval "$res" "$ddl" "$targets" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：评测 SS、MS、LS
# eval_mahdrl_as_fast_nva_all S Tight
# eval_mahdrl_as_fast_nva_all S Medium
# eval_mahdrl_as_fast_nva_all S Loose
# SM源模型：评测 SM、MM、LM
# eval_mahdrl_as_fast_nva_all M Tight
# eval_mahdrl_as_fast_nva_all M Medium
# eval_mahdrl_as_fast_nva_all M Loose
# SL源模型：评测 SL、ML、LL
# eval_mahdrl_as_fast_nva_all L Tight
# eval_mahdrl_as_fast_nva_all L Medium
# eval_mahdrl_as_fast_nva_all L Loose


# TL：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
eval_mahdrl_as_fast_tl_all() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local targets="${3:-S M L}"
    local task
    for task in $targets; do
        case "$task" in S|M|L) ;; *) echo "targets 只能包含 S M L" >&2; return 2 ;; esac
    done
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_eval_logs/eval_mahdrl_as_fast_tl_sourceS${res}_${ddl}_targets${targets// /}_noapf_select_srgt092_minenergy_v1_s700-729_m1_${run_id}"
    mkdir -p outputs/server_eval_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    targets="$3"

    for task in $targets; do
        echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"

        python -u launch.py \
            algorithms/mahdrl_oe_ge_apf_score_fast_tl/evaluate.py \
            --task "$task" --resource "$res" --deadline "$ddl" \
            --train-task S --train-resource "$res" \
            --train-deadline "$ddl" \
            --seeds 700-729 --model-seed 1 \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --diagnostics-every 100 \
            --profile-mode sampled --profile-every 100 \
            || exit 1

        echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
    done

    echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
' mahdrl_as_fast_tl_eval "$res" "$ddl" "$targets" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：评测 SS、MS、LS
# eval_mahdrl_as_fast_tl_all S Tight
# eval_mahdrl_as_fast_tl_all S Medium
# eval_mahdrl_as_fast_tl_all S Loose
# SM源模型：评测 SM、MM、LM
# eval_mahdrl_as_fast_tl_all M Tight
# eval_mahdrl_as_fast_tl_all M Medium
# eval_mahdrl_as_fast_tl_all M Loose
# SL源模型：评测 SL、ML、LL
# eval_mahdrl_as_fast_tl_all L Tight
# eval_mahdrl_as_fast_tl_all L Medium
# eval_mahdrl_as_fast_tl_all L Loose


# SYNC_TASK：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
eval_mahdrl_as_fast_sync_task_all() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local targets="${3:-S M L}"
    local task
    for task in $targets; do
        case "$task" in S|M|L) ;; *) echo "targets 只能包含 S M L" >&2; return 2 ;; esac
    done
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_eval_logs/eval_mahdrl_as_fast_sync_task_sourceS${res}_${ddl}_targets${targets// /}_noapf_select_srgt092_minenergy_v1_s700-729_m1_${run_id}"
    mkdir -p outputs/server_eval_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    targets="$3"

    for task in $targets; do
        echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"

        python -u launch.py \
            algorithms/mahdrl_oe_ge_apf_score_fast_sync_task/evaluate.py \
            --task "$task" --resource "$res" --deadline "$ddl" \
            --train-task S --train-resource "$res" \
            --train-deadline "$ddl" \
            --seeds 700-729 --model-seed 1 \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --diagnostics-every 100 \
            --profile-mode sampled --profile-every 100 \
            || exit 1

        echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
    done

    echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
' mahdrl_as_fast_sync_task_eval "$res" "$ddl" "$targets" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：评测 SS、MS、LS
# eval_mahdrl_as_fast_sync_task_all S Tight
# eval_mahdrl_as_fast_sync_task_all S Medium
# eval_mahdrl_as_fast_sync_task_all S Loose
# SM源模型：评测 SM、MM、LM
# eval_mahdrl_as_fast_sync_task_all M Tight
# eval_mahdrl_as_fast_sync_task_all M Medium
# eval_mahdrl_as_fast_sync_task_all M Loose
# SL源模型：评测 SL、ML、LL
# eval_mahdrl_as_fast_sync_task_all L Tight
# eval_mahdrl_as_fast_sync_task_all L Medium
# eval_mahdrl_as_fast_sync_task_all L Loose


# SYNC_PHASE：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
eval_mahdrl_as_fast_sync_phase_all() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local targets="${3:-S M L}"
    local task
    for task in $targets; do
        case "$task" in S|M|L) ;; *) echo "targets 只能包含 S M L" >&2; return 2 ;; esac
    done
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_eval_logs/eval_mahdrl_as_fast_sync_phase_sourceS${res}_${ddl}_targets${targets// /}_noapf_select_srgt092_minenergy_v1_s700-729_m1_${run_id}"
    mkdir -p outputs/server_eval_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    targets="$3"

    for task in $targets; do
        echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"

        python -u launch.py \
            algorithms/mahdrl_oe_ge_apf_score_fast_sync_phase/evaluate.py \
            --task "$task" --resource "$res" --deadline "$ddl" \
            --train-task S --train-resource "$res" \
            --train-deadline "$ddl" \
            --seeds 700-729 --model-seed 1 \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --diagnostics-every 100 \
            --profile-mode sampled --profile-every 100 \
            || exit 1

        echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
    done

    echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
' mahdrl_as_fast_sync_phase_eval "$res" "$ddl" "$targets" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：评测 SS、MS、LS
# eval_mahdrl_as_fast_sync_phase_all S Tight
# eval_mahdrl_as_fast_sync_phase_all S Medium
# eval_mahdrl_as_fast_sync_phase_all S Loose
# SM源模型：评测 SM、MM、LM
# eval_mahdrl_as_fast_sync_phase_all M Tight
# eval_mahdrl_as_fast_sync_phase_all M Medium
# eval_mahdrl_as_fast_sync_phase_all M Loose
# SL源模型：评测 SL、ML、LL
# eval_mahdrl_as_fast_sync_phase_all L Tight
# eval_mahdrl_as_fast_sync_phase_all L Medium
# eval_mahdrl_as_fast_sync_phase_all L Loose


# ERTF：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
eval_mahdrl_as_fast_ertf_all() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local targets="${3:-S M L}"
    local task
    for task in $targets; do
        case "$task" in S|M|L) ;; *) echo "targets 只能包含 S M L" >&2; return 2 ;; esac
    done
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_eval_logs/eval_mahdrl_as_fast_ertf_sourceS${res}_${ddl}_targets${targets// /}_noapf_select_srgt092_minenergy_v1_s700-729_m1_${run_id}"
    mkdir -p outputs/server_eval_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    targets="$3"

    for task in $targets; do
        echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"

        python -u launch.py \
            algorithms/mahdrl_oe_ge_apf_score_fast_ertf/evaluate.py \
            --task "$task" --resource "$res" --deadline "$ddl" \
            --train-task S --train-resource "$res" \
            --train-deadline "$ddl" \
            --seeds 700-729 --model-seed 1 \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --diagnostics-every 100 \
            --profile-mode sampled --profile-every 100 \
            || exit 1

        echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
    done

    echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
' mahdrl_as_fast_ertf_eval "$res" "$ddl" "$targets" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：评测 SS、MS、LS
# eval_mahdrl_as_fast_ertf_all S Tight
# eval_mahdrl_as_fast_ertf_all S Medium
# eval_mahdrl_as_fast_ertf_all S Loose
# SM源模型：评测 SM、MM、LM
# eval_mahdrl_as_fast_ertf_all M Tight
# eval_mahdrl_as_fast_ertf_all M Medium
# eval_mahdrl_as_fast_ertf_all M Loose
# SL源模型：评测 SL、ML、LL
# eval_mahdrl_as_fast_ertf_all L Tight
# eval_mahdrl_as_fast_ertf_all L Medium
# eval_mahdrl_as_fast_ertf_all L Loose


# EDF：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
eval_mahdrl_as_fast_edf_all() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local targets="${3:-S M L}"
    local task
    for task in $targets; do
        case "$task" in S|M|L) ;; *) echo "targets 只能包含 S M L" >&2; return 2 ;; esac
    done
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_eval_logs/eval_mahdrl_as_fast_edf_sourceS${res}_${ddl}_targets${targets// /}_noapf_select_srgt092_minenergy_v1_s700-729_m1_${run_id}"
    mkdir -p outputs/server_eval_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    targets="$3"

    for task in $targets; do
        echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"

        python -u launch.py \
            algorithms/mahdrl_oe_ge_apf_score_fast_edf/evaluate.py \
            --task "$task" --resource "$res" --deadline "$ddl" \
            --train-task S --train-resource "$res" \
            --train-deadline "$ddl" \
            --seeds 700-729 --model-seed 1 \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --diagnostics-every 100 \
            --profile-mode sampled --profile-every 100 \
            || exit 1

        echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
    done

    echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
' mahdrl_as_fast_edf_eval "$res" "$ddl" "$targets" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：评测 SS、MS、LS
# eval_mahdrl_as_fast_edf_all S Tight
# eval_mahdrl_as_fast_edf_all S Medium
# eval_mahdrl_as_fast_edf_all S Loose
# SM源模型：评测 SM、MM、LM
# eval_mahdrl_as_fast_edf_all M Tight
# eval_mahdrl_as_fast_edf_all M Medium
# eval_mahdrl_as_fast_edf_all M Loose
# SL源模型：评测 SL、ML、LL
# eval_mahdrl_as_fast_edf_all L Tight
# eval_mahdrl_as_fast_edf_all L Medium
# eval_mahdrl_as_fast_edf_all L Loose


# CP：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
eval_mahdrl_as_fast_cp_all() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local targets="${3:-S M L}"
    local task
    for task in $targets; do
        case "$task" in S|M|L) ;; *) echo "targets 只能包含 S M L" >&2; return 2 ;; esac
    done
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_eval_logs/eval_mahdrl_as_fast_cp_sourceS${res}_${ddl}_targets${targets// /}_noapf_select_srgt092_minenergy_v1_s700-729_m1_${run_id}"
    mkdir -p outputs/server_eval_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    targets="$3"

    for task in $targets; do
        echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"

        python -u launch.py \
            algorithms/mahdrl_oe_ge_apf_score_fast_cp/evaluate.py \
            --task "$task" --resource "$res" --deadline "$ddl" \
            --train-task S --train-resource "$res" \
            --train-deadline "$ddl" \
            --seeds 700-729 --model-seed 1 \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --diagnostics-every 100 \
            --profile-mode sampled --profile-every 100 \
            || exit 1

        echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
    done

    echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
' mahdrl_as_fast_cp_eval "$res" "$ddl" "$targets" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：评测 SS、MS、LS
# eval_mahdrl_as_fast_cp_all S Tight
# eval_mahdrl_as_fast_cp_all S Medium
# eval_mahdrl_as_fast_cp_all S Loose
# SM源模型：评测 SM、MM、LM
# eval_mahdrl_as_fast_cp_all M Tight
# eval_mahdrl_as_fast_cp_all M Medium
# eval_mahdrl_as_fast_cp_all M Loose
# SL源模型：评测 SL、ML、LL
# eval_mahdrl_as_fast_cp_all L Tight
# eval_mahdrl_as_fast_cp_all L Medium
# eval_mahdrl_as_fast_cp_all L Loose


# R090：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
eval_mahdrl_as_fast_r090_all() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local targets="${3:-S M L}"
    local task
    for task in $targets; do
        case "$task" in S|M|L) ;; *) echo "targets 只能包含 S M L" >&2; return 2 ;; esac
    done
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_eval_logs/eval_mahdrl_as_fast_r090_sourceS${res}_${ddl}_targets${targets// /}_noapf_select_srgt092_minenergy_v1_s700-729_m1_${run_id}"
    mkdir -p outputs/server_eval_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    targets="$3"

    for task in $targets; do
        echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"

        python -u launch.py \
            algorithms/mahdrl_oe_ge_apf_score_fast_r090/evaluate.py \
            --task "$task" --resource "$res" --deadline "$ddl" \
            --train-task S --train-resource "$res" \
            --train-deadline "$ddl" \
            --seeds 700-729 --model-seed 1 \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --diagnostics-every 100 \
            --profile-mode sampled --profile-every 100 \
            || exit 1

        echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
    done

    echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
' mahdrl_as_fast_r090_eval "$res" "$ddl" "$targets" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：评测 SS、MS、LS
# eval_mahdrl_as_fast_r090_all S Tight
# eval_mahdrl_as_fast_r090_all S Medium
# eval_mahdrl_as_fast_r090_all S Loose
# SM源模型：评测 SM、MM、LM
# eval_mahdrl_as_fast_r090_all M Tight
# eval_mahdrl_as_fast_r090_all M Medium
# eval_mahdrl_as_fast_r090_all M Loose
# SL源模型：评测 SL、ML、LL
# eval_mahdrl_as_fast_r090_all L Tight
# eval_mahdrl_as_fast_r090_all L Medium
# eval_mahdrl_as_fast_r090_all L Loose


# R010_NOPF：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
eval_mahdrl_as_fast_r010_nopf_all() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local targets="${3:-S M L}"
    local task
    for task in $targets; do
        case "$task" in S|M|L) ;; *) echo "targets 只能包含 S M L" >&2; return 2 ;; esac
    done
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_eval_logs/eval_mahdrl_as_fast_r010_nopf_sourceS${res}_${ddl}_targets${targets// /}_noapf_select_srgt092_minenergy_v1_s700-729_m1_${run_id}"
    mkdir -p outputs/server_eval_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    targets="$3"

    for task in $targets; do
        echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"

        python -u launch.py \
            algorithms/mahdrl_oe_ge_score_fast_r010_nopf/evaluate.py \
            --task "$task" --resource "$res" --deadline "$ddl" \
            --train-task S --train-resource "$res" \
            --train-deadline "$ddl" \
            --seeds 700-729 --model-seed 1 \
            --experiment-protocol configs/f22_split_v1.json \
            --normalization-mode f19_shared_v1 --device cuda \
            --diagnostics-every 100 \
            --profile-mode sampled --profile-every 100 \
            || exit 1

        echo "[DONE] source=S${res}/${ddl} target=${task}${res}/${ddl}"
    done

    echo "[ALL_DONE] source=S${res}/${ddl} targets=${targets}"
' mahdrl_as_fast_r010_nopf_eval "$res" "$ddl" "$targets" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：评测 SS、MS、LS
# eval_mahdrl_as_fast_r010_nopf_all S Tight
# eval_mahdrl_as_fast_r010_nopf_all S Medium
# eval_mahdrl_as_fast_r010_nopf_all S Loose
# SM源模型：评测 SM、MM、LM
# eval_mahdrl_as_fast_r010_nopf_all M Tight
# eval_mahdrl_as_fast_r010_nopf_all M Medium
# eval_mahdrl_as_fast_r010_nopf_all M Loose
# SL源模型：评测 SL、ML、LL
# eval_mahdrl_as_fast_r010_nopf_all L Tight
# eval_mahdrl_as_fast_r010_nopf_all L Medium
# eval_mahdrl_as_fast_r010_nopf_all L Loose
