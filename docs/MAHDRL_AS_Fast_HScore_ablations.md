# HScore 全部消融（严格 SR > 0.92）

以 `mahdrl_as_fast_hscore` 为共同基线，补齐以下 10 个独立变体；已实现的 TL 继续使用全局分组 VA。共 11 个消融，加上完整版为 12 个算法入口。

| 目录后缀 | 学习代理 | 变化 |
|---|---|---|
| nta | HA、VA | 每阶段从合法 TA 增量中均匀随机选取 |
| nha | TA、VA | 在有合法空闲 VM 的主机中均匀随机选主机 |
| nva | TA、HA | 在选定主机的合法空闲 VM 中均匀随机选择 |
| tl | TA、VA | 删除 HA；共享固定 93→9 VA 对所有主机分组前向，在全局合法 VM 中选择 |
| sync_task | TA、HA、VA | 每次真实分配前作 TA 决策；分配完成后更新 |
| sync_phase | TA、HA、VA | 保持阶段级 TA 决策，所有更新积攒到物理阶段边界 |
| ertf | HA、VA | 按 Ready 时间升序、全局任务 ID 升序排序 |
| edf | HA、VA | 按任务绝对子截止期、Ready 时间、全局任务 ID 升序排序 |
| cp | HA、VA | 按 upward rank 降序、Ready 时间和任务 ID 升序排序 |
| r090 | TA、HA、VA | 延迟/能耗权重 0.9/0.1，三层一致 |
| r010_nopf | TA、HA、VA | 延迟/能耗权重 0.1/0.9，三层一致 |

入口格式：`algorithms/mahdrl_as_fast_hscore_<后缀>/{train,evaluate}.py`。十个新增变体复用公共执行代码，删除的代理不会创建网络、优化器、回放或检查点。固定排序变体也不创建 TA。TL 保留独立的全局动作与全局 Double DQN 后继逻辑。

所有变体关闭主机、VM 两级 APF，掩码只依据物理合法性和空闲状态。速度慢、预计超期、能耗高本身不会过滤候选。GE 引导探索、OE 账本仍保留；“去掉 APF”不表示去掉一切性能信息。HA 保留共享打分，保留的 VA 均为固定输入 93、输出 9，每主机最多 9 个真实 VM。

NVA 的 HA 预览前三项按实际均匀随机 VM 策略计算，第四项 learned-Q 明确置零；不创建隐藏的 VA 网络。随机策略使用独立于物理环境与网络抽样的随机数生成器，验证、评测仍执行该随机策略。NHA 的随机主机策略与 TL 的全局选 VM 策略定义不同。

同步变体都保留真实后继、冻结历史观测和更新预算。HA 默认每 4 条达到 warmup 的真实经验更新一次，探索时钟仍按这些经验计数。Sync-Task 同时钟内中间 TA 奖励为零，阶段物理奖励记给最后一个待结算 TA 动作；Sync-Phase 延迟执行已产生的更新额度，不将每阶段多条经验合并成一次更新。VA 预览与实际分配之间不更新网络。

选模统一为所有验证工作流聚合成功率严格 `SR > 0.92`，可行模型中最低平均总能耗；没有可行模型时先选最高 SR 再比能耗。因此 46/50 不可行、47/50 可行。完整版和 TL 也已更换选模版本与模型版本，需要重新训练；旧版 `SR >= 0.92` 权重不能直接复用。各变体身份、角色、参数形状、选模证据、机制和保存组均严格校验。

训练短路径示例：`outputs/ckpts/hs_nva/SS_T_m1_cXXXXXXXX/`，日志使用同名目录。标准场景模型种子 1 的末级目录为 17 字符，散列包含算法、机制、源场景和协议。最终 CSV 平铺在 `outputs/run/hs_nva/`，示例：

```text
eval_hs_nva_noapf_srgt092_p1h4_SS_T_from_SS_T_m1_s700-729_cXXXXXXXX.csv
eval_hs_nva_noapf_srgt092_p1h4_MS_T_from_SS_T_m1_s700-729_cXXXXXXXX.csv
eval_hs_nva_noapf_srgt092_p1h4_LS_T_from_SS_T_m1_s700-729_cXXXXXXXX.csv
```

`from_` 前为目标场景，后为源模型；T/M/L 表示 Tight/Medium/Loose。逐分配和审计明细才使用 `_details/<hash>/`。已有训练结果或最终 CSV 拒绝覆盖。

在服务器项目根目录加载命令，按 GPU 资源依次启动：

```bash
source docs/mahdrl_as_fast_hscore_ablation_commands.sh
train_mahdrl_as_fast_hscore_nva S Tight
# 训练完成后：同一个 SS/Tight 模型评测 SS、MS、LS
eval_mahdrl_as_fast_hscore_nva_all S Tight
# 只评测两个目标，模型种子 2，GPU 1
GPU=1 eval_mahdrl_as_fast_hscore_nva_all M Medium "S L" 2
```

新变体训练函数参数为资源、截止期、可选模型种子；评测为资源、截止期、可选目标任务列表、可选模型种子。完整版与 TL 使用各自已有函数接口。后台命令均使用 nohup、独立日志和 PID，并在子评测失败时停止。训练与评测全部九源场景调用分别见 `Linux_CUDA_HScore_全部消融_600代训练指令.txt` 和 `Linux_CUDA_HScore_全部消融_九源场景评测指令.txt`。这些文件只提供指令，未启动正式长训练。

```bash
python -X utf8 -B -m unittest common.test_mahdrl_as_fast_hscore_ablation common.test_mahdrl_as_fast_hscore common.test_mahdrl_as_fast_hscore_tl -q
```

短程回归用于验证实现、能耗结算和文件组织；不代表 600 回合正式实验或性能结论。

2026-11-01 验证：联合回归 63 项通过，独立导出的待提交代码回归 32 项通过，均无失败或跳过，包含 CPU/CUDA 真实更新。36 个公开入口 dry-run 通过；108 个训练路径和 324 个 CSV 文件名唯一，训练相对路径最长 48 字符。9 份 Bash 文件语法通过，432 次替身命令调用及失败停止检查通过。机器可读结果见 [验证报告](MAHDRL_AS_Fast_HScore_ablations_validation.json)。
