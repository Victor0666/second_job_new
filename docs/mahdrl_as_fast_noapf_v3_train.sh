#!/usr/bin/env bash
# 在项目根目录激活 Python/CUDA 环境后 source 本文件；只定义函数，不自动启动作业。
# 默认 GPU=0，可在调用前设置 GPU=1；每次调用后台训练一个源模型。
# NoAPF V3：SR >= 0.92 后最小能耗，规则 select_sr092_minenergy_v1。
# 模型/日志：outputs/{ckpts,logs}/mahdrl_as_fast_noapf_v3/<源场景>_m1_c<8位摘要>/。

train_mahdrl_as_fast_noapf_v3() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_train_logs/train_noapf_v3_S${res}_${ddl}_ep600_sr092_m1_${run_id}"
    mkdir -p outputs/server_train_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"

    echo "[START] source=S${res}/${ddl} episodes=600"

    python -u launch.py \
        algorithms/mahdrl_oe_ge_apf_score_fast/train.py \
        --task S --resource "$res" --deadline "$ddl" \
        --episodes 600 --model-seed 1 \
        --experiment-protocol configs/f22_split_v1.json \
        --normalization-mode f19_shared_v1 --device cuda \
        --diagnostics-every 100 \
        --profile-mode sampled --profile-every 100 \
        || exit 1

    echo "[ALL_DONE] source=S${res}/${ddl} episodes=600"
' mahdrl_as_fast_noapf_v3_train "$res" "$ddl" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# 复制所需调用到终端执行；按显存分批运行，测量耗时避免同 GPU 争用。
# train_mahdrl_as_fast_noapf_v3 S Tight
# train_mahdrl_as_fast_noapf_v3 S Medium
# train_mahdrl_as_fast_noapf_v3 S Loose
# train_mahdrl_as_fast_noapf_v3 M Tight
# train_mahdrl_as_fast_noapf_v3 M Medium
# train_mahdrl_as_fast_noapf_v3 M Loose
# train_mahdrl_as_fast_noapf_v3 L Tight
# train_mahdrl_as_fast_noapf_v3 L Medium
# train_mahdrl_as_fast_noapf_v3 L Loose
