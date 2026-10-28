#!/usr/bin/env bash
# 在项目根目录激活 Python/CUDA 环境后：source docs/mahdrl_as_fast_ablations_train.sh
# 本文件只定义函数，不自动启动任何训练或评测。
# 默认 GPU=0；可在调用前设置 GPU=1。每次函数调用立即后台启动一个作业。
# 训练9个源模型；每个评测作业依次运行对应 S/M/L 目标，共27个场景。
# 旧权重不能用于本次 noAPF/SR>0.92 配置；先完成相应源模型训练再评测。
# 日志/PID 每次调用独立命名；单次作业失败立即退出，不继续后续目标。
# 最终 CSV 位于 outputs/run/<变体短标识>/，场景直接写入 CSV 文件名。

# NTA：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
train_mahdrl_as_fast_nta() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_train_logs/train_mahdrl_as_fast_nta_sourceS${res}_${ddl}_noapf_select_srgt092_minenergy_v1_e600_m1_${run_id}"
    mkdir -p outputs/server_train_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    echo "[START] training=S${res}/${ddl} episodes=600 model_seed=1"

    python -u launch.py \
        algorithms/mahdrl_oe_ge_apf_score_fast_nta/train.py \
        --task S --resource "$res" --deadline "$ddl" \
        --episodes 600 --model-seed 1 \
        --experiment-protocol configs/f22_split_v1.json \
        --normalization-mode f19_shared_v1 --device cuda \
        --diagnostics-every 100 \
        --profile-mode sampled --profile-every 100 \
        || exit 1

    echo "[DONE] training=S${res}/${ddl} episodes=600 model_seed=1"
    echo "[ALL_DONE] training=S${res}/${ddl}"
' mahdrl_as_fast_nta_train "$res" "$ddl" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：训练 SS
# train_mahdrl_as_fast_nta S Tight
# train_mahdrl_as_fast_nta S Medium
# train_mahdrl_as_fast_nta S Loose
# SM源模型：训练 SM
# train_mahdrl_as_fast_nta M Tight
# train_mahdrl_as_fast_nta M Medium
# train_mahdrl_as_fast_nta M Loose
# SL源模型：训练 SL
# train_mahdrl_as_fast_nta L Tight
# train_mahdrl_as_fast_nta L Medium
# train_mahdrl_as_fast_nta L Loose


# NHA：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
train_mahdrl_as_fast_nha() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_train_logs/train_mahdrl_as_fast_nha_sourceS${res}_${ddl}_noapf_select_srgt092_minenergy_v1_e600_m1_${run_id}"
    mkdir -p outputs/server_train_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    echo "[START] training=S${res}/${ddl} episodes=600 model_seed=1"

    python -u launch.py \
        algorithms/mahdrl_oe_ge_apf_score_fast_nha/train.py \
        --task S --resource "$res" --deadline "$ddl" \
        --episodes 600 --model-seed 1 \
        --experiment-protocol configs/f22_split_v1.json \
        --normalization-mode f19_shared_v1 --device cuda \
        --diagnostics-every 100 \
        --profile-mode sampled --profile-every 100 \
        || exit 1

    echo "[DONE] training=S${res}/${ddl} episodes=600 model_seed=1"
    echo "[ALL_DONE] training=S${res}/${ddl}"
' mahdrl_as_fast_nha_train "$res" "$ddl" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：训练 SS
# train_mahdrl_as_fast_nha S Tight
# train_mahdrl_as_fast_nha S Medium
# train_mahdrl_as_fast_nha S Loose
# SM源模型：训练 SM
# train_mahdrl_as_fast_nha M Tight
# train_mahdrl_as_fast_nha M Medium
# train_mahdrl_as_fast_nha M Loose
# SL源模型：训练 SL
# train_mahdrl_as_fast_nha L Tight
# train_mahdrl_as_fast_nha L Medium
# train_mahdrl_as_fast_nha L Loose


# NVA：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
train_mahdrl_as_fast_nva() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_train_logs/train_mahdrl_as_fast_nva_sourceS${res}_${ddl}_noapf_select_srgt092_minenergy_v1_e600_m1_${run_id}"
    mkdir -p outputs/server_train_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    echo "[START] training=S${res}/${ddl} episodes=600 model_seed=1"

    python -u launch.py \
        algorithms/mahdrl_oe_ge_apf_score_fast_nva/train.py \
        --task S --resource "$res" --deadline "$ddl" \
        --episodes 600 --model-seed 1 \
        --experiment-protocol configs/f22_split_v1.json \
        --normalization-mode f19_shared_v1 --device cuda \
        --diagnostics-every 100 \
        --profile-mode sampled --profile-every 100 \
        || exit 1

    echo "[DONE] training=S${res}/${ddl} episodes=600 model_seed=1"
    echo "[ALL_DONE] training=S${res}/${ddl}"
' mahdrl_as_fast_nva_train "$res" "$ddl" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：训练 SS
# train_mahdrl_as_fast_nva S Tight
# train_mahdrl_as_fast_nva S Medium
# train_mahdrl_as_fast_nva S Loose
# SM源模型：训练 SM
# train_mahdrl_as_fast_nva M Tight
# train_mahdrl_as_fast_nva M Medium
# train_mahdrl_as_fast_nva M Loose
# SL源模型：训练 SL
# train_mahdrl_as_fast_nva L Tight
# train_mahdrl_as_fast_nva L Medium
# train_mahdrl_as_fast_nva L Loose


# TL：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
train_mahdrl_as_fast_tl() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_train_logs/train_mahdrl_as_fast_tl_sourceS${res}_${ddl}_noapf_select_srgt092_minenergy_v1_e600_m1_${run_id}"
    mkdir -p outputs/server_train_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    echo "[START] training=S${res}/${ddl} episodes=600 model_seed=1"

    python -u launch.py \
        algorithms/mahdrl_oe_ge_apf_score_fast_tl/train.py \
        --task S --resource "$res" --deadline "$ddl" \
        --episodes 600 --model-seed 1 \
        --experiment-protocol configs/f22_split_v1.json \
        --normalization-mode f19_shared_v1 --device cuda \
        --diagnostics-every 100 \
        --profile-mode sampled --profile-every 100 \
        || exit 1

    echo "[DONE] training=S${res}/${ddl} episodes=600 model_seed=1"
    echo "[ALL_DONE] training=S${res}/${ddl}"
' mahdrl_as_fast_tl_train "$res" "$ddl" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：训练 SS
# train_mahdrl_as_fast_tl S Tight
# train_mahdrl_as_fast_tl S Medium
# train_mahdrl_as_fast_tl S Loose
# SM源模型：训练 SM
# train_mahdrl_as_fast_tl M Tight
# train_mahdrl_as_fast_tl M Medium
# train_mahdrl_as_fast_tl M Loose
# SL源模型：训练 SL
# train_mahdrl_as_fast_tl L Tight
# train_mahdrl_as_fast_tl L Medium
# train_mahdrl_as_fast_tl L Loose


# SYNC_TASK：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
train_mahdrl_as_fast_sync_task() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_train_logs/train_mahdrl_as_fast_sync_task_sourceS${res}_${ddl}_noapf_select_srgt092_minenergy_v1_e600_m1_${run_id}"
    mkdir -p outputs/server_train_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    echo "[START] training=S${res}/${ddl} episodes=600 model_seed=1"

    python -u launch.py \
        algorithms/mahdrl_oe_ge_apf_score_fast_sync_task/train.py \
        --task S --resource "$res" --deadline "$ddl" \
        --episodes 600 --model-seed 1 \
        --experiment-protocol configs/f22_split_v1.json \
        --normalization-mode f19_shared_v1 --device cuda \
        --diagnostics-every 100 \
        --profile-mode sampled --profile-every 100 \
        || exit 1

    echo "[DONE] training=S${res}/${ddl} episodes=600 model_seed=1"
    echo "[ALL_DONE] training=S${res}/${ddl}"
' mahdrl_as_fast_sync_task_train "$res" "$ddl" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：训练 SS
# train_mahdrl_as_fast_sync_task S Tight
# train_mahdrl_as_fast_sync_task S Medium
# train_mahdrl_as_fast_sync_task S Loose
# SM源模型：训练 SM
# train_mahdrl_as_fast_sync_task M Tight
# train_mahdrl_as_fast_sync_task M Medium
# train_mahdrl_as_fast_sync_task M Loose
# SL源模型：训练 SL
# train_mahdrl_as_fast_sync_task L Tight
# train_mahdrl_as_fast_sync_task L Medium
# train_mahdrl_as_fast_sync_task L Loose


# SYNC_PHASE：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
train_mahdrl_as_fast_sync_phase() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_train_logs/train_mahdrl_as_fast_sync_phase_sourceS${res}_${ddl}_noapf_select_srgt092_minenergy_v1_e600_m1_${run_id}"
    mkdir -p outputs/server_train_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    echo "[START] training=S${res}/${ddl} episodes=600 model_seed=1"

    python -u launch.py \
        algorithms/mahdrl_oe_ge_apf_score_fast_sync_phase/train.py \
        --task S --resource "$res" --deadline "$ddl" \
        --episodes 600 --model-seed 1 \
        --experiment-protocol configs/f22_split_v1.json \
        --normalization-mode f19_shared_v1 --device cuda \
        --diagnostics-every 100 \
        --profile-mode sampled --profile-every 100 \
        || exit 1

    echo "[DONE] training=S${res}/${ddl} episodes=600 model_seed=1"
    echo "[ALL_DONE] training=S${res}/${ddl}"
' mahdrl_as_fast_sync_phase_train "$res" "$ddl" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：训练 SS
# train_mahdrl_as_fast_sync_phase S Tight
# train_mahdrl_as_fast_sync_phase S Medium
# train_mahdrl_as_fast_sync_phase S Loose
# SM源模型：训练 SM
# train_mahdrl_as_fast_sync_phase M Tight
# train_mahdrl_as_fast_sync_phase M Medium
# train_mahdrl_as_fast_sync_phase M Loose
# SL源模型：训练 SL
# train_mahdrl_as_fast_sync_phase L Tight
# train_mahdrl_as_fast_sync_phase L Medium
# train_mahdrl_as_fast_sync_phase L Loose


# ERTF：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
train_mahdrl_as_fast_ertf() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_train_logs/train_mahdrl_as_fast_ertf_sourceS${res}_${ddl}_noapf_select_srgt092_minenergy_v1_e600_m1_${run_id}"
    mkdir -p outputs/server_train_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    echo "[START] training=S${res}/${ddl} episodes=600 model_seed=1"

    python -u launch.py \
        algorithms/mahdrl_oe_ge_apf_score_fast_ertf/train.py \
        --task S --resource "$res" --deadline "$ddl" \
        --episodes 600 --model-seed 1 \
        --experiment-protocol configs/f22_split_v1.json \
        --normalization-mode f19_shared_v1 --device cuda \
        --diagnostics-every 100 \
        --profile-mode sampled --profile-every 100 \
        || exit 1

    echo "[DONE] training=S${res}/${ddl} episodes=600 model_seed=1"
    echo "[ALL_DONE] training=S${res}/${ddl}"
' mahdrl_as_fast_ertf_train "$res" "$ddl" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：训练 SS
# train_mahdrl_as_fast_ertf S Tight
# train_mahdrl_as_fast_ertf S Medium
# train_mahdrl_as_fast_ertf S Loose
# SM源模型：训练 SM
# train_mahdrl_as_fast_ertf M Tight
# train_mahdrl_as_fast_ertf M Medium
# train_mahdrl_as_fast_ertf M Loose
# SL源模型：训练 SL
# train_mahdrl_as_fast_ertf L Tight
# train_mahdrl_as_fast_ertf L Medium
# train_mahdrl_as_fast_ertf L Loose


# EDF：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
train_mahdrl_as_fast_edf() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_train_logs/train_mahdrl_as_fast_edf_sourceS${res}_${ddl}_noapf_select_srgt092_minenergy_v1_e600_m1_${run_id}"
    mkdir -p outputs/server_train_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    echo "[START] training=S${res}/${ddl} episodes=600 model_seed=1"

    python -u launch.py \
        algorithms/mahdrl_oe_ge_apf_score_fast_edf/train.py \
        --task S --resource "$res" --deadline "$ddl" \
        --episodes 600 --model-seed 1 \
        --experiment-protocol configs/f22_split_v1.json \
        --normalization-mode f19_shared_v1 --device cuda \
        --diagnostics-every 100 \
        --profile-mode sampled --profile-every 100 \
        || exit 1

    echo "[DONE] training=S${res}/${ddl} episodes=600 model_seed=1"
    echo "[ALL_DONE] training=S${res}/${ddl}"
' mahdrl_as_fast_edf_train "$res" "$ddl" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：训练 SS
# train_mahdrl_as_fast_edf S Tight
# train_mahdrl_as_fast_edf S Medium
# train_mahdrl_as_fast_edf S Loose
# SM源模型：训练 SM
# train_mahdrl_as_fast_edf M Tight
# train_mahdrl_as_fast_edf M Medium
# train_mahdrl_as_fast_edf M Loose
# SL源模型：训练 SL
# train_mahdrl_as_fast_edf L Tight
# train_mahdrl_as_fast_edf L Medium
# train_mahdrl_as_fast_edf L Loose


# CP：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
train_mahdrl_as_fast_cp() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_train_logs/train_mahdrl_as_fast_cp_sourceS${res}_${ddl}_noapf_select_srgt092_minenergy_v1_e600_m1_${run_id}"
    mkdir -p outputs/server_train_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    echo "[START] training=S${res}/${ddl} episodes=600 model_seed=1"

    python -u launch.py \
        algorithms/mahdrl_oe_ge_apf_score_fast_cp/train.py \
        --task S --resource "$res" --deadline "$ddl" \
        --episodes 600 --model-seed 1 \
        --experiment-protocol configs/f22_split_v1.json \
        --normalization-mode f19_shared_v1 --device cuda \
        --diagnostics-every 100 \
        --profile-mode sampled --profile-every 100 \
        || exit 1

    echo "[DONE] training=S${res}/${ddl} episodes=600 model_seed=1"
    echo "[ALL_DONE] training=S${res}/${ddl}"
' mahdrl_as_fast_cp_train "$res" "$ddl" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：训练 SS
# train_mahdrl_as_fast_cp S Tight
# train_mahdrl_as_fast_cp S Medium
# train_mahdrl_as_fast_cp S Loose
# SM源模型：训练 SM
# train_mahdrl_as_fast_cp M Tight
# train_mahdrl_as_fast_cp M Medium
# train_mahdrl_as_fast_cp M Loose
# SL源模型：训练 SL
# train_mahdrl_as_fast_cp L Tight
# train_mahdrl_as_fast_cp L Medium
# train_mahdrl_as_fast_cp L Loose


# R090：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
train_mahdrl_as_fast_r090() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_train_logs/train_mahdrl_as_fast_r090_sourceS${res}_${ddl}_noapf_select_srgt092_minenergy_v1_e600_m1_${run_id}"
    mkdir -p outputs/server_train_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    echo "[START] training=S${res}/${ddl} episodes=600 model_seed=1"

    python -u launch.py \
        algorithms/mahdrl_oe_ge_apf_score_fast_r090/train.py \
        --task S --resource "$res" --deadline "$ddl" \
        --episodes 600 --model-seed 1 \
        --experiment-protocol configs/f22_split_v1.json \
        --normalization-mode f19_shared_v1 --device cuda \
        --diagnostics-every 100 \
        --profile-mode sampled --profile-every 100 \
        || exit 1

    echo "[DONE] training=S${res}/${ddl} episodes=600 model_seed=1"
    echo "[ALL_DONE] training=S${res}/${ddl}"
' mahdrl_as_fast_r090_train "$res" "$ddl" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：训练 SS
# train_mahdrl_as_fast_r090 S Tight
# train_mahdrl_as_fast_r090 S Medium
# train_mahdrl_as_fast_r090 S Loose
# SM源模型：训练 SM
# train_mahdrl_as_fast_r090 M Tight
# train_mahdrl_as_fast_r090 M Medium
# train_mahdrl_as_fast_r090 M Loose
# SL源模型：训练 SL
# train_mahdrl_as_fast_r090 L Tight
# train_mahdrl_as_fast_r090 L Medium
# train_mahdrl_as_fast_r090 L Loose


# R010_NOPF：无主机/VM APF，严格 SR > 0.92；源模型均为 task=S。
train_mahdrl_as_fast_r010_nopf() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_train_logs/train_mahdrl_as_fast_r010_nopf_sourceS${res}_${ddl}_noapf_select_srgt092_minenergy_v1_e600_m1_${run_id}"
    mkdir -p outputs/server_train_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    echo "[START] training=S${res}/${ddl} episodes=600 model_seed=1"

    python -u launch.py \
        algorithms/mahdrl_oe_ge_score_fast_r010_nopf/train.py \
        --task S --resource "$res" --deadline "$ddl" \
        --episodes 600 --model-seed 1 \
        --experiment-protocol configs/f22_split_v1.json \
        --normalization-mode f19_shared_v1 --device cuda \
        --diagnostics-every 100 \
        --profile-mode sampled --profile-every 100 \
        || exit 1

    echo "[DONE] training=S${res}/${ddl} episodes=600 model_seed=1"
    echo "[ALL_DONE] training=S${res}/${ddl}"
' mahdrl_as_fast_r010_nopf_train "$res" "$ddl" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；以下注释不会自动启动作业。
# SS源模型：训练 SS
# train_mahdrl_as_fast_r010_nopf S Tight
# train_mahdrl_as_fast_r010_nopf S Medium
# train_mahdrl_as_fast_r010_nopf S Loose
# SM源模型：训练 SM
# train_mahdrl_as_fast_r010_nopf M Tight
# train_mahdrl_as_fast_r010_nopf M Medium
# train_mahdrl_as_fast_r010_nopf M Loose
# SL源模型：训练 SL
# train_mahdrl_as_fast_r010_nopf L Tight
# train_mahdrl_as_fast_r010_nopf L Medium
# train_mahdrl_as_fast_r010_nopf L Loose
