# ERTF-H-FIFS：低—高—低突发到达实验（远程 Linux）

本入口使用现有 `common.bursty_workload` 生成器，与其他算法的 `bursty_p20_r5` 评测配对。启发式的任务 FCFS、最早空闲主机及 FIFS VM 选择规则保持一致。

每回合固定 50 个工作流，到达观察窗口为 50/0.03≈1666.67 秒。低负载占窗口的 0%–40% 和 60%–100%，到达率为 1/60；峰值占 40%–60%，到达率为 1/12。到达时刻来自三段非齐次泊松过程在窗口内恰好 50 次到达的条件分布。沿用原工作流模板、载荷、资源和冻结 FCFS 参考缓存，按实际到达时刻加原有相对截止期预算。

S/M/L 任务 × S/M/L 资源 × Tight/Medium/Loose，共 27 个场景。每场景种子 700–729，共 810 回合、40500 个工作流。ERTF-H-FIFS 无训练和模型加载步骤。

## 同步与检查

在服务器项目根目录激活现有实验环境。上传 `ertf_fifs_bursty_remote_20261103.zip` 后执行：

```bash
unzip -o ertf_fifs_bursty_remote_20261103.zip
python -m algorithms.ertf_fifs.evaluate_suite --help
python launch.py common/f22_cache.py --all
```

补丁包含 `algorithms/ertf_fifs/evaluate.py`、`algorithms/ertf_fifs/evaluate_suite.py`、`common/scenarios.py`、`common/bursty_workload.py`、`environments/mahdrl_heft.py` 和本文档。它在现有同版本项目上更新到达配置及 ERTF 入口，不携带模型、缓存和其他算法源码。

## 完整回合预检（可选）

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -X utf8 -B -u -m algorithms.ertf_fifs.evaluate_suite \
  --profiles bursty_p20_r5 --tasks S --resources S --seeds 700 --workers 1 \
  --out outputs/run/ertf_fifs_bursty_preflight_20261103
```

运行 SS 的三个截止期，各 1 回合，共 150 个工作流。预检输出与正式结果分开。

## 正式后台评测

```bash
mkdir -p outputs/server_eval_logs

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup python -X utf8 -B -u -m algorithms.ertf_fifs.evaluate_suite \
  --profiles bursty_p20_r5 --seeds 700-729 --workers 4 \
  --out outputs/run/ertf_fifs_bursty_20261103 \
  > outputs/server_eval_logs/ertf_fifs_bursty_20261103.log 2>&1 < /dev/null &

echo $! > outputs/server_eval_logs/ertf_fifs_bursty_20261103.pid
```

首次运行要求输出目录尚不存在。`--workers 4` 只控制独立场景的 CPU 并发。

```bash
tail -f outputs/server_eval_logs/ertf_fifs_bursty_20261103.log
```

每个场景的逐种子日志与进度分别保存在输出目录的 `logs/` 和 `_details/<场景>/progress.json`。顶层进度按完成的场景更新。正式完成时输出 `ALL_DONE scenes=27 episodes=810`，`summary.json` 为 `status=complete`。

## 中断续跑

确认旧进程退出后执行下列命令。保持源码、缓存、协议、场景和种子不变；续跑会核对 manifest，并跳过已经保存到场景 CSV 的种子。

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup python -X utf8 -B -u -m algorithms.ertf_fifs.evaluate_suite \
  --profiles bursty_p20_r5 --seeds 700-729 --workers 4 \
  --out outputs/run/ertf_fifs_bursty_20261103 --resume \
  >> outputs/server_eval_logs/ertf_fifs_bursty_20261103.log 2>&1 < /dev/null &

echo $! > outputs/server_eval_logs/ertf_fifs_bursty_20261103.pid
```

## 指标与结果

`summary_scenarios.csv`：每场景的能耗、Makespan、工作流成功率、平均超期量，以及它们的样本标准差和均值的 95% t 区间；另含 `esp` 和 `esp_threshold`。`all_episodes.csv`：810 行逐种子结果，包括工作流总数与成功数。`_details/`：逐回合、工作流与决策明细。兼容旧入口还会写同内容的 `summary_45_scenarios.csv`，该文件名不代表实际场景数。

ESP 沿用现有突发实验报告的定义：每场景 30 个种子中，工作流按时完成率达到 0.92（50 个中至少 46 个按时完成）的回合比例。成功率和 ESP 均以 0–1 保存，转为百分数时乘 100。平均超期量包含未超期工作流的零值。

每回合核对 50 个工作流全部完成、三段到达时刻与公共生成器一致、相对截止期预算、公共指标与原入口指标一致，以及独立能耗积分一致。并发仿真及省略未使用的神经观测仅用于加速运行，相关耗时不用于跨算法推理效率比较。旧 `docs/audit_ertf_results.py` 写死了原 45 场景，不能直接用于本次突发评测。
