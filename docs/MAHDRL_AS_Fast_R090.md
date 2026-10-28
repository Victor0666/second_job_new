> **2026-10-29 当前版本更新：**本页对应变体现已统一关闭主机和 VM 两级 APF，采用严格 `SR > 0.92` 选模、短训练目录和按场景命名的平铺最终 CSV。当前定义、版本隔离与路径示例见[统一说明](MAHDRL_AS_Fast_消融统一配置_2026-10-29.md)。原命令入口仍有效；下文旧阈值、APF 和路径描述为历史实现记录，以统一说明为准。

# MAHDRL-AS-Fast-R090

基于 MAHDRL-AS-Fast v2 的独立奖励系数变式，不属于 IRWS。

| 项目 | 值 |
| --- | --- |
| 算法 ID / 入口目录名 | `mahdrl_oe_ge_apf_score_fast_r090` |
| 显示名称 | `MAHDRL-AS-Fast-R090` |
| 模型版本 | `mahdrl_as_fast_r090_v1` |
| 奖励协议 | `mahdrl_as_three_layer_delay090_energy010_v1` |
| 选模规则 | `select_sr099_minenergy_v1` |

## 奖励变化

TA、HA、VA 的时延权重统一从 0.75 改成 0.9，对应能耗权重从 0.25 改成 0.1。
保留 `energy_reward_scale=0.001`、`task_baseline_norm=300`。

TA 继续使用阶段增量能耗与阶段累计拖期，不新增截断：

```
r_TA = -(0.9 * phase_delay_cost + (1 - 0.9) * phase_energy_cost)
phase_delay_cost = phase_tardiness_sum / 300
phase_energy_cost = phase_energy * 0.001
```

HA、VA 继续使用 OE 账本的逐分配增量能耗：

```
r_HA = r_VA = (1 - 0.9) * (-0.001 * assignment_energy)
              - 0.9 * clip(task_tardiness / 300, 0, 1)
```

强制分配的补结算能耗仍由原补结算函数按 `1-alpha` 加入，R090 使用 0.1 的
能耗权重。浮点运算沿用原来的 `1-alpha` 计算顺序。
环境构造、分配、阶段推进和运行入口检查实际奖励参数，避免运行时被覆盖为 0.75。

## 保留的机制与隔离

直接继承 v2 的 TA/HA/VA learner，复用网络、打分头、GE、APF、不可变回放、
NumPy 拼装、单决策缓存、采样诊断和计时。学习率、gamma、batch、容量、warmup、
epsilon、梯度裁剪、目标网络更新与训练数据划分保持一致。
相同动作下，观测、候选掩码、执行时长、物理能耗和 OE 账本应与原版本一致。

每回合验证后仍按总成功工作流数/总工作流数判断 SR>=0.99，达标后选最低平均能耗，
未达标时按成功率最高、能耗最低回退，完全并列保留较早模型。默认 seed600 有
50 个工作流，需 50/50 成功才达标；该条件不保证测试集成功率。

原 MAHDRL-AS-Fast v2、其他算法、公共注册表和公共物理代码不修改。
模型和日志使用 `mahdrl_as_fast_r090_v1_...`，评测使用
`outputs/run/mahdrl_oe_ge_apf_score_fast_r090/mahdrl_as_fast_r090_v1_.../`。
所有正式入口拒绝覆盖非空结果目录。

完整奖励设置写入 `algorithm_variant`、`implementation`、运行元数据、训练/评测总结，
以及 checkpoint 的 `normalization.mahdrl_as_fast_r090`。模型旁和评测目录还写入
`reward_settings.json`。加载严格检查奖励设置、算法/模型版本、网络结构、角色、
源归一化、实验协议、验证证据及三层保存组；R075 与 R090 权重互相拒绝加载。
R090 从头训练，不提供借旧权重绕过奖励身份校验的选项，也不新增断点续训。

原截止期缓存和源归一化尺度仍可使用，因为参考生成器、物理模型与数据划分未改。
泛化评测继续固定 `--train-task S`、源资源和源截止期，改变目标 `--task`。

## 运行示例

在实际项目根目录、已激活 Python/CUDA 环境后执行：

```bash
python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast_r090/train.py \
  --task S --resource S --deadline Tight --episodes 600 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json \
  --normalization-mode f19_shared_v1 --device cuda \
  --diagnostics-every 100 --profile-mode sampled --profile-every 100

python -u launch.py algorithms/mahdrl_oe_ge_apf_score_fast_r090/evaluate.py \
  --task M --resource S --deadline Tight \
  --train-task S --train-resource S --train-deadline Tight \
  --seeds 700-729 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json \
  --normalization-mode f19_shared_v1 --device cuda \
  --diagnostics-every 100 --profile-mode sampled --profile-every 100
```

`--dry-run` 预览配置和路径；`--profile-mode off` 关闭阶段计时；
`--diagnostics-every 1` 逐次采集详细数值诊断。相应开关不改变奖励配置。
自定义拓扑和 deadline precompute 仍通过本目录的 `precompute.py`、
`--resource-config`、`--deadline-cache` 使用原参考生成器。

九组后台训练见 `Linux_CUDA_MAHDRL_AS_Fast_R090_600代训练指令.txt`。
每个源模型一次调用完成 S/M/L 三个目标的指令见
`Linux_CUDA_MAHDRL_AS_Fast_R090_27场景共用评测指令.txt`。
函数按当前项目定位权重，不写死服务器根路径；每次调用启动一个后台进程。

## 验证与适用范围

```bash
python -X utf8 -B -m unittest common.test_mahdrl_as_fast_r090 common.test_mahdrl_as_fast common.test_mahdrl_oe_ge_apf_score common.test_pd3qn_dag_fe.SelectionTests -v
```

测试覆盖三层奖励及强制分配补结算；同动作的观测/掩码/时长/能耗/账本一致性；
正式网络宽度的 CPU/CUDA 固定经验更新等价性；0.99 选择规则复用；奖励配置篡改、
跨版本/角色/保存组拒绝；真实短训练→保存→加载→冻结评测；九组源模型和
27 个目标的路径映射、诊断/计时及原输出保留。

本机 47 项联合测试全部通过，包括 CPU/CUDA 正式网络宽度的固定经验更新检查。

测试中的短训练采用小网络、两个工作流，不能代替九场景 600 回合正式实验。
正式实验应与 R075 使用相同种子和预算，对比能耗、工作流成功率、拖期和完成时间。
提高时延权重不保证测试成功率提高，也不保证能耗降低。
