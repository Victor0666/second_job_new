# 新增压力变式：alpha=1.5，lambda=0.06，参考 lambda=0.03

新增 `--workload-profile stress_a15_l006_ref003`，四算法共用。原 `standard` 和 `stress_a1_l006_ref003` 保持原有行为、模型路径和结果命名。

| 参数 | 本变式 |
| --- | --- |
| 固定截止期系数 alpha | 1.5 |
| 实际到达率 lambda | 0.06 |
| 截止期公式 | `deadline_i = actual_arrival_i + 1.5 * reference_duration_i` |
| 参考时长 | 原 lambda=0.03 的 FCFS_task + VM_first_idle 缓存 `wf_makespans[i]` |
| 到达序列 | 四算法均调用公共 `poisson_arrival_times(0.06, n_max=50, seed=episode_seed)` |
| 每个 episode 的工作流数 | 50 |
| 训练/验证/测试实例种子 | 0–599 / 600 / 700–729 |
| 选模 | 沿用 SR≥92% 后最小能耗；无可行模型时最大 SR，再最小能耗 |
| 归一化 | 复用原 Tight 场景的固定训练尺度 |

无需生成新 deadline cache 或重新校准。alpha 作用于缓存中的参考时长，不作用于到达时刻；该系数与奖励权重无关。相对于 alpha=1 变式，使用相同实例种子时到达序列、任务负载和参考调度轨迹一致，可用时间预算增加 50%。实际成功率和能耗的变化需由实验确定。

四个入口为 MARL-Q4.2、PD3QN-FE、IRWS-R-FE-IDLE，以及当前仓库的 MAHDRL-AS-Fast-NoAPF v3。MAHDRL 继续使用 `algorithms/mahdrl_oe_ge_apf_score_fast/` 路径，此路径已在之前的改动中升级为 NoAPF v3。本次仅增加负载配置，不修改算法网络、奖励或调度规则。

## 启用参数

在这四个算法的训练和评测命令中使用：

```text
--deadline Tight --workload-profile stress_a15_l006_ref003
```

评测时同时保留 `--train-deadline Tight`。Tight 在压力变式中用于指定冻结归一化尺度，实际 alpha 固定为 1.5；不要重复运行 Medium/Loose。

训练应在该变式下重新进行，再使用对应的验证与测试实例。三类场景的有效协议与模型路径分开，加载时拒绝混用 standard、alpha=1 与 alpha=1.5 的模型，包括 IRWS 断点恢复。每个训练目录名不超过 32 字符，最终 CSV 仍平铺在相应算法的统一目录。新 CSV 名称例如：

```text
eval_pd3qn_fe_MS_Tight_fromSS_Tight_m1_s700-729_c12345678_a15l006r003.csv
```

末尾 `a15l006r003` 表示 alpha=1.5、实际 lambda=0.06、参考 lambda=0.03，区别于原 alpha=1 的 `a1l006r003`。

## Linux CUDA 后台训练

在项目根目录、已激活的 Python 环境中加载更新后的函数：

```bash
source docs/Linux_CUDA_stress_four_algorithms.sh
```

新函数名称中增加 `_a15`。每条命令独立启动一个 `CUDA_VISIBLE_DEVICES=0 nohup bash -c` 后台作业，默认训练 600 个 episode，模型种子 1；连续粘贴多条会并发启动，应根据 GPU 容量安排。

```bash
# MARL-Q4.2：SS、SM、SL 三个源模型
train_marl_q42_stress_a15 S Tight
train_marl_q42_stress_a15 M Tight
train_marl_q42_stress_a15 L Tight

# PD3QN-FE
train_pd3qn_fe_stress_a15 S Tight
train_pd3qn_fe_stress_a15 M Tight
train_pd3qn_fe_stress_a15 L Tight

# IRWS-R-FE-IDLE
train_irws_r_fe_idle_stress_a15 S Tight
train_irws_r_fe_idle_stress_a15 M Tight
train_irws_r_fe_idle_stress_a15 L Tight

# MAHDRL-AS-Fast-NoAPF v3
train_mahdrl_as_fast_noapf_v3_stress_a15 S Tight
train_mahdrl_as_fast_noapf_v3_stress_a15 M Tight
train_mahdrl_as_fast_noapf_v3_stress_a15 L Tight
```

## Linux CUDA 后台评测

相应源模型训练完成后再启动评测。每条命令按任务 S、M、L 顺序评测三个目标，使用测试种子 700–729；任何目标失败即退出。

```bash
# SS 源模型：SS、MS、LS
eval_marl_q42_stress_a15_all S Tight
eval_pd3qn_fe_stress_a15_all S Tight
eval_irws_r_fe_idle_stress_a15_all S Tight
eval_mahdrl_as_fast_noapf_v3_stress_a15_all S Tight

# SM 源模型：SM、MM、LM
eval_marl_q42_stress_a15_all M Tight
eval_pd3qn_fe_stress_a15_all M Tight
eval_irws_r_fe_idle_stress_a15_all M Tight
eval_mahdrl_as_fast_noapf_v3_stress_a15_all M Tight

# SL 源模型：SL、ML、LL
eval_marl_q42_stress_a15_all L Tight
eval_pd3qn_fe_stress_a15_all L Tight
eval_irws_r_fe_idle_stress_a15_all L Tight
eval_mahdrl_as_fast_noapf_v3_stress_a15_all L Tight
```

第三个参数可限制目标集合，例如 `eval_pd3qn_fe_stress_a15_all S Tight "M L"`。日志和 PID 文件名带 `a15l006r003`，分别存于 `outputs/server_train_logs/` 和 `outputs/server_eval_logs/`。

不带 `_a15` 的既有 `*_stress`、`*_stress_all` 函数仍运行 alpha=1，不受本次新增影响。标准场景仍通过省略 `--workload-profile` 运行。
