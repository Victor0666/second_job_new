# TL_2：去掉 HA，VA 使用全局能耗奖励

TL_2 是在原 TL 上增加的独立消融版本，标识为 `ablation_tl2`，不会替代原 `ablation_nohost`。

## 实验定义

| 设置 | TL | TL_2 |
| --- | --- | --- |
| 结构 | TA → 全局 VA，无 HA | 相同 |
| TA 状态、动作及 phase 奖励 | 沿用 MAHDRL | 相同 |
| VA 动作 | 在所有空闲 VM 中选择，全局 VM 索引 | 相同 |
| VA 观测 | 22 + 8N（222 / 422 / 622） | 相同 |
| VA 能耗项 | 所选主机的能耗奖励 | 所有主机的全局能耗奖励 |
| VA 延迟项、网络和训练配置 | 原 TL 配置 | 相同 |
| 实例、截止期、数据划分与归一化 | 公共协议 | 相同 |

TL_2 的 VA 奖励：

`r_VA = (1 - alpha_delay_vm) * r_energy_total + alpha_delay_vm * r_delay`

其中 `alpha_delay_vm=0.75`；`r_energy_total` 为当前推进区间内全部主机能耗乘以负的能耗缩放系数（默认 0.001），不是总能耗的累计值。延迟项保持原 TL 的任务超期惩罚、300 秒归一化和裁剪。没有把原 HA 奖励整体叠加到 VA，避免重复延迟惩罚。

同一时刻仍可继续分配时，能耗项为零；时间推进后再结算区间能耗。若分配已返回能耗，phase 结束不再重复加入；若 phase 另外推进时间，其全局能耗补给最近一次尚待结算的 VA 决策。训练奖励审计也按全局口径核对。验证和评测的 VA 奖励指标保留 phase 均值定义，并将延迟能耗补回最近分配所在 phase 的均值。

## 运行

在项目根目录执行（已有公共截止期缓存、归一化配置和数据划分配置应先准备好）：

```powershell
# 训练 SS；将 resource 改为 M/L，可分别训练 SM/SL。
.venv/Scripts/python.exe -m algorithms.ablation_tl2.train --task S --resource S --deadline Tight

# MS、LS 使用 SS 模型。资源规模相同；默认 train-task 为 S。
.venv/Scripts/python.exe -m algorithms.ablation_tl2.evaluate --task M --resource S --train-task S --deadline Tight
.venv/Scripts/python.exe -m algorithms.ablation_tl2.evaluate --task L --resource S --train-task S --deadline Tight

# 只查看实际配置和模型路径。
.venv/Scripts/python.exe -m algorithms.ablation_tl2.train --task S --resource M --deadline Medium --dry-run
```

完整泛化对应为 SS→SS/MS/LS、SM→SM/MM/LM、SL→SL/ML/LL。截止期等级分别训练和评测；默认仍为 600 回合及公共训练/验证/测试划分。

模型名以 `tl2_` 开头并含 `globalEnergy`；评测结果使用独立的 `hrl_tl_2_ddlMix` 目录与 `run_hrl_TL_2_...csv` 文件。训练入口复用原 TL 的流程，评测入口复用原 TL 的评测，避免复制整套代码。模型目录的 `reward_protocol.json` 标记 VA 能耗口径；TL_2 评测会拒绝缺少全局口径标记的模型，防止误用原 TL 权重。旧 TL 模型仍可按原 TL 入口评测。

## 结论范围和验证

TL 与 TL_2 的对比用于观察 VA 局部/全局能耗目标的影响。MAHDRL 与 TL_2 的对比仍包含移除 HA 后必然出现的输入维度、动作空间和参数量变化，不能声称是完全相同网络仅删除一个 HA。全局奖励也不保证效果更好，需要重跑后判断。

```powershell
.venv/Scripts/python.exe -B -m unittest common.test_tl2 -v
```

测试包括相同动作下物理轨迹和 TA 奖励一致、VA 奖励公式和总能耗守恒、尾部能耗不重复结算、模型口径隔离与泛化路径，以及临时目录中的训练→保存→加载→评测。冒烟测试使用缩小的网络和两工作流回合，以验证接线及实际更新；它不代表正式 600 回合实验结果。
