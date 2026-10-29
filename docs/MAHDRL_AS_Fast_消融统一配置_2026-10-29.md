# AS-Fast 消融：无 APF、严格 SR > 0.92、紧凑输出

2026-10-29 更新。以下 11 个独立变体均使用本页规则；训练与评测入口名称保持不变。

| 变体 | algorithms/ 下的入口目录 | 短输出标识 |
| --- | --- | --- |
| NTA | mahdrl_oe_ge_apf_score_fast_nta | mahdrl_nta |
| NHA | mahdrl_oe_ge_apf_score_fast_nha | mahdrl_nha |
| NVA | mahdrl_oe_ge_apf_score_fast_nva | mahdrl_nva |
| TL | mahdrl_oe_ge_apf_score_fast_tl | mahdrl_tl |
| Sync-Task | mahdrl_oe_ge_apf_score_fast_sync_task | mahdrl_sync_task |
| Sync-Phase | mahdrl_oe_ge_apf_score_fast_sync_phase | mahdrl_sync_phase |
| ERTF | mahdrl_oe_ge_apf_score_fast_ertf | mahdrl_ertf |
| EDF | mahdrl_oe_ge_apf_score_fast_edf | mahdrl_edf |
| CP | mahdrl_oe_ge_apf_score_fast_cp | mahdrl_cp |
| R090 | mahdrl_oe_ge_apf_score_fast_r090 | mahdrl_r090 |
| R010-NoPF | mahdrl_oe_ge_score_fast_r010_nopf | mahdrl_r010_nopf |

## 候选资源与学习

训练、验证、评测统一关闭主机和 VM 两级 PF/APF。不执行速度阈值、自适应阈值或按时最低能耗候选保留。主机候选为所有至少拥有一个合法空闲 VM 的主机，VM 候选为所选主机内所有合法空闲 VM；TL 直接在全部主机的合法空闲 VM 中学习选择。

忙碌 VM、填充槽位和物理非法动作继续屏蔽。速度慢或预计超期不再构成排除条件。候选预测仍用于观测和原 GE 探索，不改变实际执行、通信、OE 能耗结算和公共 Host 任务窗口计量。

各变体保留自己的随机/固定排序、保留学习角色及更新时钟。R090 的时延/能耗权重仍为 0.9/0.1，R010-NoPF 仍为 0.1/0.9；其余变体仍为 0.75/0.25。学习率、网络形状、回放、训练预算、数据划分和源场景冻结尺度保持原设置。

## 模型选择

规则版本为 `select_srgt092_minenergy_v1`。总体 SR = 全部验证实例的按时成功工作流数之和 / 工作流数之和；能耗 E 为各验证实例总能耗的算术平均。

- 严格 `SR > 0.92` 才可行，等于 0.92 不可行。
- 可行模型优先，先比较更低 E，再比较更高 SR。
- 无可行模型时，先比较更高 SR，再比较更低 E。
- 完全并列保留较早的检查点。

默认验证 seed600 含 50 个工作流，46/50=0.92 不可行，47/50=0.94 可行。同步变体的训练稳定性统计也使用严格大于 0.92。

best/final 的证据校验、成套角色与保存组核验继续执行。新版本身份与旧 APF/SR099 配置隔离，旧模型不能改名当作新模型，需要按新设置重新训练。本页 11 个变体严格使用 `SR > 0.92`；完整版 MAHDRL-AS-Fast-V2 已另行改为 `SR >= 0.92`，并保留两级 APF。独立对比算法不随此修改。

## 训练输出

模型和日志分别采用：

```text
outputs/ckpts/<短输出标识>/SS_L_m1_cXXXXXXXX/
outputs/logs/<短输出标识>/SS_L_m1_cXXXXXXXX/
```

`SS_L` 表示任务 S、资源 S、Loose；截止期缩写 T/M/L 分别表示 Tight/Medium/Loose。`m1` 为模型种子，`cXXXXXXXX` 为 8 位配置摘要。默认场景、模型种子1的最后一级目录为 17 个字符。配置摘要绑定算法版本、奖励/机制、选模协议、归一化模式和校准内容；完整设置继续保存到运行及检查点元数据。移动路径但校准内容不变时，名称不变。

## 评测输出

各变体所有场景的最终 CSV 平铺在同一个目录，不再依靠场景文件夹区分同名 CSV：

```text
outputs/run/mahdrl_nta/
  eval_mahdrl_nta_noapf_srgt092_SS_L_from_SS_L_m1_s700-729_cXXXXXXXX.csv
  eval_mahdrl_nta_noapf_srgt092_SS_M_from_SS_M_m1_s700-729_cYYYYYYYY.csv
  eval_mahdrl_nta_noapf_srgt092_MS_L_from_SS_L_m1_s700-729_cXXXXXXXX.csv
  _details/<16位运行摘要>/...
```

文件名携带算法、无 APF/选模设置、目标场景、源训练场景、模型种子、测试种子和配置码。非连续种子集合额外携带集合摘要。S 源模型评测 M/L 任务时仍定位同一源检查点，不用目标场景重选模型或校准尺度。

逐工作流明细、测量和 summary 放入 `_details` 下的独立摘要目录。最终 CSV 行也记录目标/源场景及两级 APF 关闭标识。已有 CSV 或运行明细拒绝覆盖，同一变体的不同场景可以共享 CSV 父目录。完成提示输出最终 CSV 的绝对路径；`--dry-run` 同时输出模型、日志、CSV 和明细路径。

## 使用

按 Bash 函数、nohup、独立日志和 PID 格式整理的完整指令分别位于
[训练指令](mahdrl_as_fast_ablations_train.sh)和[评测指令](mahdrl_as_fast_ablations_evaluate.sh)。
两份文件各包含 11 个独立变体的函数及每个变体的 9 个源场景调用示例；source 只加载函数，不启动作业。

```bash
source docs/mahdrl_as_fast_ablations_train.sh
source docs/mahdrl_as_fast_ablations_evaluate.sh

# 训练示例：后台训练 R090 的 SS/Tight 源模型。
train_mahdrl_as_fast_r090 S Tight

# 训练完成后：后台依次评测 SS、MS、LS。
eval_mahdrl_as_fast_r090_all S Tight

# 只评测 M/L 任务目标时显式指定第三个参数。
eval_mahdrl_as_fast_r090_all S Tight "M L"
```

默认 `GPU=0`，可在调用前设置 `GPU=1`。每次调用立即后台启动一个作业；评测作业内部依次处理目标 S/M/L，失败即停止后续目标。日志/PID 位于 `outputs/server_train_logs/` 和 `outputs/server_eval_logs/`，带时间及随机后缀防止调用覆盖。同一卡的多个作业会叠加显存需求，应按设备容量选择并发数量。

在项目根目录激活 Python/CUDA 环境，示例为 NTA 的 SS/Loose：

```bash
python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast_nta/train.py --task S --resource S --deadline Loose --episodes 600 --model-seed 1 --normalization-mode f19_shared_v1 --device cuda
python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast_nta/evaluate.py --task S --resource S --deadline Loose --train-task S --train-resource S --train-deadline Loose --seeds 700-729 --model-seed 1 --normalization-mode f19_shared_v1 --device cuda
```

其余变体替换上表入口目录即可；加 `--dry-run` 查看当前路径。共享 FCFS 截止期缓存和 F19 校准可继续复用。默认仍为训练实例0–599、验证600、测试700–729；每模型600回合，每回合50个工作流，9个源场景与27个目标场景。

验证入口为 `common.test_mahdrl_ablation_settings`，并配合原有各消融专项、`common.test_r010_paths`、主算法与公共场景/通信回归。检查覆盖严格阈值、11个环境无筛选、297个唯一CSV名称、紧凑路径、真实短程训练/选模/保存/重载/评测及重复输出保护；不替代正式600回合实验。

本轮验证：**133项测试全部通过，0失败、0错误、0跳过**，用时151.246秒，包含CPU/CUDA更新；22个公开训练/评测入口dry-run通过。11×27=297个标准场景CSV名称均唯一；模型种子1下训练目录末级名称均为17字符，本工作区最长CSV绝对路径159字符、最长测量文件绝对路径142字符。

完整[验证报告](MAHDRL_AS_Fast_ablation_validation_2026-10-29.json)与[测试记录](MAHDRL_AS_Fast_ablation_tests_2026-10-29.txt)已保存。未执行正式600回合重训或完整27场景实验。

训练/评测 Bash 文件通过语法检查及24次替身调用，核验全部22个函数的命令参数、日志/PID、目标覆盖和失败立即停止；未启动真实训练。记录见[指令验证](MAHDRL_AS_Fast_ablation_commands_validation_2026-10-29.json)。
