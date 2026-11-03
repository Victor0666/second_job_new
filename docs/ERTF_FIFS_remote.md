# ERTF-H-FIFS：远程运行标准、突发与压力实验

在服务器项目根目录激活已有 Python 环境。AutoDL 当前环境可使用 `/root/autodl-tmp/envs/second_job/bin/python`；以下 `python` 均指该环境。

## 更新与检查

```bash
git pull --ff-only origin codex/f11-real-transitions
python -m algorithms.ertf_fifs.evaluate_suite --help
python launch.py common/f22_cache.py --all
mkdir -p outputs/server_eval_logs
```

检查通过后选择需要的实验。三个命令均使用 CPU、测试种子 700–729、每回合 50 个工作流；`--workers 4` 控制独立场景的并发数。首次运行的输出目录必须尚不存在。

## 标准主试验

S/M/L 任务 × S/M/L 资源 × Tight/Medium/Loose，共 27 场景、810 回合。到达率 0.03，沿用各截止期配置的 alpha=2/3 混合和冻结参考缓存。

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup python -X utf8 -B -u -m algorithms.ertf_fifs.evaluate_suite \
  --profiles standard --seeds 700-729 --workers 4 \
  --out outputs/run/ertf_fifs_main_20261103 \
  > outputs/server_eval_logs/ertf_fifs_main_20261103.log 2>&1 < /dev/null &
```

## 突发到达实验

共 27 场景、810 回合。固定观察窗口为 50/0.03 秒，低负载覆盖 0%–40% 和 60%–100%，到达率为 1/60；峰值覆盖 40%–60%，到达率为 1/12。使用窗口内恰好 50 次到达的条件泊松分布，沿用模板、载荷、资源及原相对截止期预算。

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup python -X utf8 -B -u -m algorithms.ertf_fifs.evaluate_suite \
  --profiles bursty_p20_r5 --seeds 700-729 --workers 4 \
  --out outputs/run/ertf_fifs_bursty_20261103 \
  > outputs/server_eval_logs/ertf_fifs_bursty_20261103.log 2>&1 < /dev/null &
```

## 两组截止期压力实验

到达率均为 0.06，截止期 alpha 分别固定为 1 和 1.5。每组仅运行 S/M/L 任务 × S/M/L 资源，共 9 场景、270 回合；两组合计 18 场景、540 回合。`Tight` 仅为场景配置标签，实际 alpha 由 profile 固定。保留原到达率 0.03 的参考 makespan，避免随资源竞争放宽截止期预算。

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup python -X utf8 -B -u -m algorithms.ertf_fifs.evaluate_suite \
  --profiles stress_a1_l006_ref003 stress_a15_l006_ref003 \
  --seeds 700-729 --workers 4 \
  --out outputs/run/ertf_fifs_stress_20261103 \
  > outputs/server_eval_logs/ertf_fifs_stress_20261103.log 2>&1 < /dev/null &
```

## 奖励权重实验中的 ERTF 基线

ERTF 按固定的任务 FCFS、最早空闲主机及 FIFS VM 规则选择动作；返回的奖励仅用于记录，不更新调度策略。相同实例下改变奖励权重不会改变物理调度结果。因此，奖励权重 0.10、0.90 和原设置可共同使用上述标准主试验的 ERTF 固定基线，并在结果中注明复用关系。

## 进度、结果与续跑

```bash
tail -f outputs/server_eval_logs/ertf_fifs_stress_20261103.log
```

按实验替换日志文件名。正式完成后输出 `ALL_DONE`，结果目录中的 `summary.json` 为 `status=complete`。`all_episodes.csv` 保存逐种子结果；`summary_45_scenarios.csv` 保存能耗、Makespan、工作流成功率、平均超期量的均值、样本标准差、均值的 95% t 区间，以及 ESP。该汇总文件名沿用旧入口，实际行数由请求的场景数决定；含突发配置时额外写 `summary_scenarios.csv`。

ESP 为每场景 30 个种子中，工作流按时完成率达到 0.92（至少 46/50 个按时完成）的回合比例。工作流成功率和 ESP 以 0–1 保存；平均超期量包含未超期工作流的零值。

中断后确认旧进程退出，再给相同命令添加 `--resume`，并将日志重定向 `>` 改为 `>>`。续跑会核验源码、协议、缓存、场景及种子，并跳过已经写入场景 CSV 的种子。正在使用严格续跑的批次应保持原源码；更新 Git 后新增加的 Python 文件也会影响 manifest 的源码清单。

默认不传 `--profiles` 时，会运行标准 27 场景及两组压力各 9 场景，共 45 场景、1350 回合。显式选择 profile 可独立执行三类实验。旧 `docs/audit_ertf_results.py` 只适用于其原有的 45 场景协议，不能直接核验单独的 27 或 18 场景批次。
