# 四算法新增压力场景：alpha=1，lambda=0.06，参考负载=0.03

另有 alpha=1.5 变式，参数为 `--workload-profile stress_a15_l006_ref003`，使用说明与新后台函数见 [Stress_a15_l006_ref003.md](Stress_a15_l006_ref003.md)。本文的 alpha=1 参数与原后台函数行为不变。

新增参数：`--workload-profile stress_a1_l006_ref003`。省略参数或使用 `--workload-profile standard` 时，继续运行原场景的混合 alpha 和训练/评测逻辑。

适用入口：MARL-Q4.2、PD3QN-FE、IRWS-R-FE-IDLE、MAHDRL-AS-Fast-V2。其他算法未接入此参数。

## 实验规则

| 项目 | 新压力场景 |
| --- | --- |
| 实际工作流到达率 | 0.06 |
| 到达序列 | 四算法使用 `common.read_xml_opt_Tsize.poisson_arrival_times(0.06, n_max=50, seed=episode_seed)` |
| 工作流数量 | 每个 episode 固定 50 个 |
| 截止时间 | `actual_arrival_i + 1.0 * reference_duration_i` |
| 参考时长 | 原有 lambda=0.03 的 FCFS_task + VM_first_idle 缓存中的 `wf_makespans[i]` |
| 训练/验证/测试实例种子 | F22：0–599 / 600 / 700–729 |
| 归一化 | 复用相应原 Tight 训练场景的固定尺度，不重新校准 |
| 选模 | 保留各算法现有 SR≥92% 后最小能耗规则；无可行模型时优先最大 SR，再最小能耗 |

`reference_duration_i` 是参考调度轨迹中工作流的完成时刻减到达时刻，包含该轨迹中的等待；不是单个工作流的独立运行最短时间。实际到达时间使用 0.06 重新生成，参考时长仍取 0.03 的记录，不减半、不放大，也不在 0.06 下重新计算。模板顺序、工作流负载和缓存实例哈希仍严格对应原记录。

MARL 在新场景保留 `cache_fcfs` 截止时间模式，只将到达序列来源切换为公共生成器；原场景仍读取缓存的到达时刻。其余三个算法使用原有公共生成器入口，传入 0.06。

新场景使用 `--deadline Tight`（默认值），评测源场景使用 `--train-deadline Tight`。这里的 Tight 用于选择原有固定归一化尺度；实际 alpha 由 workload profile 固定为 1。不要再运行 Medium/Loose 来复制相同的压力场景。底层双 alpha 接口对应 choices=(1,1)、probs=(1,0)，所有工作流实际 alpha 都为 1。

## 前置文件

无需生成新 deadline cache 或归一化配置。继续使用：

- `data/deadlines/dynamic_full_parent_output_v1/ddl_fcfs_task{S|M|L}_res{S|M|L}_seed0-1000.json`
- `outputs/f18_normalization_calibration/f18_shared_normalization_v1.json`
- `configs/f22_split_v1.json`

可先审计原参考缓存覆盖范围：

```powershell
python -m common.f22_cache --all
```

训练和测试入口会检查所用缓存与原 0.03 参考配置匹配。不要将参考缓存的 metadata 中 arrival_lambda 改成 0.06，也不要覆盖缓存中的 arrival_times。若文件缺失，按原 0.03 预计算流程补齐参考缓存。

## 训练命令

以下命令在项目根目录运行，使用已安装项目依赖的 Python。示例为 S 任务、S 资源训练，600 个 episode。Windows 本机可将 `python` 替换为 `.\.venv\Scripts\python.exe`；Linux 使用相应虚拟环境的 `python`。

```powershell
python -m algorithms.marl.train --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --workload-profile stress_a1_l006_ref003
python -m algorithms.pd3qn_fe.train --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --workload-profile stress_a1_l006_ref003
python -m algorithms.irws_r_fe_idle.train --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --workload-profile stress_a1_l006_ref003
python -m algorithms.mahdrl_oe_ge_apf_score_fast.train --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --workload-profile stress_a1_l006_ref003
```

在任一命令末尾添加 `--dry-run` 可仅查看配置和输出路径。IRWS 默认 CPU，可在有 CUDA 的机器上添加 `--device cuda`；PD3QN-FE、MAHDRL-AS-Fast-V2 也支持该参数。MARL 训练沿用原来的 CUDA 自动选择逻辑。

需要分别训练 M/L 资源时，修改 `--resource`，各配置会生成独立模型。不要将原场景模型当作新场景训练结果。新场景采用独立输出目录；已有非空训练/评测输出仍会阻止覆盖。

## 测试命令

先完成相应新场景模型的训练，然后运行：

```powershell
python -m algorithms.marl.evaluate --task S --resource S --deadline Tight --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 --workload-profile stress_a1_l006_ref003
python -m algorithms.pd3qn_fe.evaluate --task S --resource S --deadline Tight --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 --workload-profile stress_a1_l006_ref003
python -m algorithms.irws_r_fe_idle.evaluate --task S --resource S --deadline Tight --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 --workload-profile stress_a1_l006_ref003
python -m algorithms.mahdrl_oe_ge_apf_score_fast.evaluate --task S --resource S --deadline Tight --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 --workload-profile stress_a1_l006_ref003
```

对 M/L 任务做迁移评测时修改 `--task M` 或 `--task L`，保持 `--train-task S`，并使 `--train-resource` 对应所选训练模型。每次测试使用目标任务/资源对应的原 0.03 参考缓存。IRWS 的 VM 数量维度约束保持不变。

新旧场景加载检查会核对有效协议中的 workload 字段，包括实际到达率、参考到达率和 alpha；即使显式指定对方模型路径，也会拒绝混用。标准场景仍接受现有不含 workload 字段的合法模型。IRWS 断点恢复也执行该检查，其余算法原有的恢复支持范围不变。

## 结果与验证

原有评测输出继续包含 `wf_success_rate`、`wf_avg_tardiness`、能耗和 makespan；各 seed 的 `measurements/workflows/` CSV 包含工作流层面的记录，可用来统计超时分布。新场景元数据记录实际 lambda=0.06、参考 lambda=0.03、alpha=1，便于区分物理负载与归一化来源。

四算法统一使用短训练目录名（不超过 32 字符），例如 `SS_Tight_m1_c12345678`。算法、源场景、归一化配置内容、选模和 workload 参数进入配置哈希；完整参数仍保存在运行元数据中。

MARL 和 IRWS 的原标准训练路径不变。PD3QN-FE 和 MAHDRL 新训练改用短目录；评测优先选择新目录，新目录不存在时自动读取原版本目录，也可用 `--checkpoint-dir` 明确指定已有模型目录。不会移动或覆盖旧模型。若新目录已存在但不完整，会报错而不会静默回退旧模型。

最终 CSV 平铺到每个算法的统一目录：MARL 为 `outputs/run/marl_q_v4_2/`，PD3QN-FE 为 `outputs/run/pd3qn_fe/`，IRWS 为 `outputs/run/irws_r_fe_idle/`，MAHDRL 为 `outputs/run/mahdrl_oe_ge_apf_score_fast/`。例如：

```text
eval_mahdrl_as_fast_v2_MS_Tight_fromSS_Tight_m1_s700-729_c12345678_a1l006r003.csv
```

文件名中的 `MS` 是目标场景，`fromSS` 是源模型场景；`m1` 是模型种子，`s700-729` 是测试实例范围，`a1l006r003` 表示 alpha=1、实际 lambda=0.06、参考 lambda=0.03。标准场景省略该 workload 后缀，用 Tight/Medium/Loose 区分原有三种混合截止期。逐 seed 明细、工作流 CSV、summary JSON 放在 `_details/<16位哈希>/` 下。不同场景不会因统一 CSV 父目录已有文件而互相阻止，同名 CSV 或已有非空明细目录会阻止覆盖。

## Linux CUDA 后台训练与评测

先在仓库根目录、已激活的 Python 环境中加载函数：

```bash
source docs/Linux_CUDA_stress_four_algorithms.sh
```

脚本沿用 `CUDA_VISIBLE_DEVICES=0 nohup bash -c` 格式，后台进程的 `.log` 和 `.pid` 分别放入 `outputs/server_train_logs/`、`outputs/server_eval_logs/`。日志名包含源模型、目标集合、workload、SR92、模型种子及时间；启动函数会打印日志位置。加载脚本不会启动实验。

下面列出各算法的三种源资源训练命令。每条命令启动一个后台作业，请按 GPU 显存容量安排，默认逐条运行、等上一组结束后再启动下一组：

```bash
train_marl_q42_stress S Tight
train_marl_q42_stress M Tight
train_marl_q42_stress L Tight

train_pd3qn_fe_stress S Tight
train_pd3qn_fe_stress M Tight
train_pd3qn_fe_stress L Tight

train_irws_r_fe_idle_stress S Tight
train_irws_r_fe_idle_stress M Tight
train_irws_r_fe_idle_stress L Tight

train_mahdrl_as_fast_v2_stress S Tight
train_mahdrl_as_fast_v2_stress M Tight
train_mahdrl_as_fast_v2_stress L Tight
```

对应新场景模型训练完成后评测。每条命令在同一后台进程中按 S、M、L 任务顺序评测三个目标场景；任一目标失败立即退出：

```bash
# SS 源模型：SS、MS、LS
eval_marl_q42_stress_all S Tight
eval_pd3qn_fe_stress_all S Tight
eval_irws_r_fe_idle_stress_all S Tight
eval_mahdrl_as_fast_v2_stress_all S Tight

# SM 源模型：SM、MM、LM
eval_marl_q42_stress_all M Tight
eval_pd3qn_fe_stress_all M Tight
eval_irws_r_fe_idle_stress_all M Tight
eval_mahdrl_as_fast_v2_stress_all M Tight

# SL 源模型：SL、ML、LL
eval_marl_q42_stress_all L Tight
eval_pd3qn_fe_stress_all L Tight
eval_irws_r_fe_idle_stress_all L Tight
eval_mahdrl_as_fast_v2_stress_all L Tight
```

仅评测部分目标时可增加第三个参数，例如 `eval_mahdrl_as_fast_v2_stress_all S Tight "M L"`。本次函数固定使用新压力场景，不需要重复 Medium/Loose，也没有切换到 R090 算法或 SR99 选模规则。

回归验证命令：

```powershell
python -m unittest common.test_primary_output_paths common.test_stress_workload common.test_scenarios common.test_paired_instances common.test_marl_v4_2 common.test_marl_paths common.test_pd3qn_fe common.test_irws_r_fe_idle common.test_mahdrl_as_fast common.test_r010_paths common.test_f22 -v
```

小规模 CPU 测试用于验证训练、验证选模、保存、加载、测试和场景隔离，不替代正式训练，也不能据此判断压力场景下哪种算法性能更好。
