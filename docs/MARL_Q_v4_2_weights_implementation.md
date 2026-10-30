# MARL_Q4.2 奖励权重变式

两个变式均直接基于当前 `algorithms/marl/` 中的 MARL-Q4.2 实现，原版入口和默认系数 0.75 保持不变。

| 变式 | 延迟系数 alpha | 能耗系数 1-alpha | 独立代码目录 |
|---|---:|---:|---|
| MARL_Q4.2_010 | 0.1 | 0.9 | `algorithms/marl_q_v4_2_010/` |
| MARL_Q4.2_090 | 0.9 | 0.1 | `algorithms/marl_q_v4_2_090/` |

Global 和 Local 同时使用相应系数：

```text
r_global = (1-alpha) * r_energy_total - alpha * delay_norm
r_local  = (1-alpha) * r_energy_host  - alpha * delay_norm
```

Global 保留全局区间能耗，Local 保留选中 Host 的区间能耗；能耗缩放 0.001、延迟归一化分母 300 保持不变。环境初始化和每步执行核验奖励参数，检查点和配置指纹记录完整奖励设置，拒绝原版及另一权重变式的模型。

## 保留的 MARL-Q4.2 行为

- 只向空闲 VM 分配；任务在全局 Ready 集合按 FCFS 等待，处理实际到达和完成事件推进时间。
- 保留原观测、动作掩码、任务物理过程、通信和能耗计算。
- 各 Host 的 Local Actor/Critic 独立；Global Critic 读取实际 Local 动作；一步 TD 复用实际后继动作，先更新 Critic 再更新 Actor。
- 隐藏层为 1024/1024/512/512/256，学习率 0.0003，gamma=0.95，不启用熵正则和梯度裁剪。
- 保留 F22 划分、F19 冻结归一化和验证隔离。选模仍使用成功率至少 **92%** 后最小能耗规则，并成套保存 Global/Local。

## 输出路径

训练检查点和日志分别位于：

```text
outputs/ckpts/marl_q_v4_2_010/SS_Tight_m1_cXXXXXXXX/
outputs/logs/marl_q_v4_2_010/SS_Tight_m1_cXXXXXXXX/
outputs/ckpts/marl_q_v4_2_090/SS_Tight_m1_cXXXXXXXX/
outputs/logs/marl_q_v4_2_090/SS_Tight_m1_cXXXXXXXX/
```

配置指纹由实际协议、奖励、选模、观测、调度、归一化校准内容和能耗口径计算，不应写死。不同源场景和模型种子使用独立短目录。

最终 CSV 在各算法目录中平铺，例如：

```text
outputs/run/marl_q_v4_2_010/eval_marl_q_v4_2_010_LS_Tight_fromSS_Tight_m1_s700-729_cXXXXXXXX.csv
outputs/run/marl_q_v4_2_090/eval_marl_q_v4_2_090_LS_Tight_fromSS_Tight_m1_s700-729_cXXXXXXXX.csv
```

文件名包含目标场景、源场景、模型种子、测试种子和配置指纹；27 个场景的最终 CSV 不依靠父目录区分。summary 和逐种子明细放在同目录的 `_details/<文件名哈希>/`。显式 `--out` 仍生效，已有 CSV 或对应明细存在时拒绝覆盖。评测测量记录包含实际加载的两个检查点路径和哈希，包括自定义检查点路径。

## 训练和评测命令

在项目根目录、已激活 Python 环境的 Bash 中使用以下文件。每个训练文件提供 9 个源场景、600 回合的后台训练函数；每个评测文件按 9 个源模型顺序评测各自 S/M/L 目标，共 27 个场景，测试种子为 700–729、模型种子为 1。评测函数第三个参数可限制目标，例如 `"M L"`。

| 变式 | 训练 | 评测 |
|---|---|---|
| 010 | [600回合训练指令](Linux_CUDA_MARL_Q_v4_2_010_600代训练指令.txt) | [27场景评测指令](Linux_CUDA_MARL_Q_v4_2_010_27场景评测指令.txt) |
| 090 | [600回合训练指令](Linux_CUDA_MARL_Q_v4_2_090_600代训练指令.txt) | [27场景评测指令](Linux_CUDA_MARL_Q_v4_2_090_27场景评测指令.txt) |

Python 入口为各独立目录下的 `train.py` 和 `evaluate.py`，通过 `python -u launch.py <入口> ...` 调用。原版及其他权重的检查点不能直接用于这两个变式，需要分别训练。

## 验证

```bash
python -B -m unittest common.test_marl_q42_weights common.test_marl_v4_2 common.test_marl_v4 common.test_marl_q -v
```

测试覆盖相同行为下物理轨迹不变、Global/Local 奖励范围、系数固定、学习更新保留、跨版本模型拒绝、成套选模、短训练路径、27 个平铺 CSV，以及小网络真实训练、验证、保存重载和冻结评测。实际执行结果及命令校验记录见 [验证报告](MARL_Q_v4_2_weights_validation.json)。原版源文件快照保存在两个目录的 `source_snapshot_sha256.json` 中。

本次实现未运行 600 回合正式训练，冒烟测试不用于判断收敛或算法性能。

2026-10-31 路径复核：7 项路径测试通过；四份命令文件通过 Bash 语法检查，展开后的 18 条训练和 54 条评测命令通过实际参数解析，四个 `launch.py --dry-run` 入口通过。训练场景目录名最长 22 个字符；每个变式的 27 个最终 CSV 位于同一算法目录，文件名包含目标场景、源场景、模型种子、测试种子和配置指纹。
