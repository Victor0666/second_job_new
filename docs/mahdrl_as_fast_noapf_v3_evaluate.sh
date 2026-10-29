#!/usr/bin/env bash
# 在项目根目录激活 Python/CUDA 环境后 source 本文件；只定义函数，不自动启动作业。
# 默认 GPU=0，可在调用前设置 GPU=1；先完成对应源模型训练再评测。
# 九个源模型，每个作业顺序评测 S/M/L 任务规模，共27个目标场景、810次实例评测。
# 最终 CSV 平铺在 outputs/run/mahdrl_oe_ge_apf_score_fast/，文件名包含目标/源场景。
# SR >= 0.92，规则 select_sr092_minenergy_v1；旧 APF V2 权重不能加载。

eval_mahdrl_as_fast_noapf_v3_all() {
    local res="${1:-}" ddl="${2:-}"
    case "$res" in S|M|L) ;; *) echo "resource 必须是 S/M/L" >&2; return 2 ;; esac
    case "$ddl" in Tight|Medium|Loose) ;; *) echo "deadline 必须是 Tight/Medium/Loose" >&2; return 2 ;; esac
    local targets="${3:-S M L}"
    local task
    for task in $targets; do
        case "$task" in S|M|L) ;; *) echo "targets 只能包含 S M L" >&2; return 2 ;; esac
    done
    local run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    local prefix="outputs/server_eval_logs/eval_mahdrl_as_fast_noapf_v3_sourceS${res}_${ddl}_targets${targets// /}_select_sr092_minenergy_v1_seeds700-729_model1_${run_id}"
    mkdir -p outputs/server_eval_logs || return

    CUDA_VISIBLE_DEVICES="${GPU:-0}" nohup bash -c '
    res="$1"
    ddl="$2"
    targets="$3"

    for task in $targets; do
        echo "[START] source=S${res}/${ddl} target=${task}${res}/${ddl}"

        python -u launch.py \
            algorithms/mahdrl_oe_ge_apf_score_fast/evaluate.py \
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
' mahdrl_as_fast_noapf_v3_eval "$res" "$ddl" "$targets" \
        > "${prefix}.log" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "$pid" > "${prefix}.pid"
    printf '[LAUNCHED] pid=%s log=%s.log\n' "$pid" "$prefix"
}

# SS源模型：评测 SS、MS、LS；以下注释不会自动启动作业。
# eval_mahdrl_as_fast_noapf_v3_all S Tight
# eval_mahdrl_as_fast_noapf_v3_all S Medium
# eval_mahdrl_as_fast_noapf_v3_all S Loose
# SM源模型：评测 SM、MM、LM
# eval_mahdrl_as_fast_noapf_v3_all M Tight
# eval_mahdrl_as_fast_noapf_v3_all M Medium
# eval_mahdrl_as_fast_noapf_v3_all M Loose
# SL源模型：评测 SL、ML、LL
# eval_mahdrl_as_fast_noapf_v3_all L Tight
# eval_mahdrl_as_fast_noapf_v3_all L Medium
# eval_mahdrl_as_fast_noapf_v3_all L Loose
