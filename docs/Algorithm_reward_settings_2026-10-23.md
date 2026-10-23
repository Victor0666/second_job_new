# 当前各算法实际奖励设置

日期：2026-10-23。只读核对训练入口注入的参数、实际环境和学习回报，不依据环境构造函数默认值推断。本次没有修改任何算法奖励。

## 共同量与结算方式

令任务完成时刻为 C_i、公共任务子截止期为 d_i：

```text
L_i = max(0, C_i-d_i)
D_i = clip(L_i/300, 0, 1)
R(E,D) = -0.25 * 0.001 * E - 0.75 * D
```

下表各学习算法实际训练配置均使用延迟权重0.75、能耗权重0.25、能耗缩放0.001、延迟分母300秒；TA 使用阶段超期和，IRWS 在外部奖励上另有融合。E 的单位为 J。物理能耗以每台实际使用 Host 自己的任务窗口计，包含窗口内空闲区间，不是从仿真0时刻给全部机器一直计空闲功率。

- `E_assignment`：新分配引起的精确 Host 任务窗口能耗增量，计入窗口扩展产生的内部空闲能耗。同次分配只向一台 Host 增加任务，故正常有效分配的全局增量与该 Host 增量相同。
- `E_global_interval`：实际仿真时间推进区间的全局能耗。同一时刻连续分配时可能暂为0，到推进时集中结算。
- `E_host_interval`：对应推进区间内本次所选 Host 的能耗。
- `E_phase`：整个管理阶段的物理能耗。

## 算法对照

| 算法/层级 | 实际回报 | 说明 |
|---|---|---|
| PD3QN | `R(E_assignment,D_i)` | 每次分配结算；末步总能耗重算可能产生微小浮点修正 |
| DRL-GNN | `R(E_assignment,D_i)` | 继承 IRWS 物理环境，直接使用外部奖励 |
| MAHDRL HA | `R(E_global_interval,D_i)` | 全局区间能耗 |
| MAHDRL VA | `R(E_host_interval,D_i)` | 所选 Host 区间能耗 |
| MAHDRL TA | `R(E_phase,sum_phase L_i/300)` | 对阶段内新完成任务累计超期；不截断到1 |
| MARL Global | `R(E_global_interval,D_i)` | 全局区间能耗 |
| MARL Local | `R(E_host_interval,D_i)` | 所选 Host 区间能耗；没有 TA |
| ERTF/FIFS | 无训练回报 | 规则选动作；环境仍产生 MAHDRL 形式的奖励供日志统计 |

MAHDRL 及其消融的学习回报还经过 F11 阶段尾结算：尚未返回的推进能耗会补入上一条 pending 决策，HA 使用全局、VA 使用所选 Host。不能只读取某次 `vm_assign` 的返回值来计算整条训练回报。

## MAHDRL 变式与消融

| 分支 | 相对原版奖励的变化 |
|---|---|
| GE、PF、TS、UB | 奖励公式保持原版 |
| OE、OE_GE、OE_PF、OE_GE_PF | HA/VA 均改为 `R(E_assignment,D_i)`；TA 保持原版 |
| ablation_hostrand / vmrand / mgrrand | 环境奖励保持原版，对应随机层不学习 |
| ablation_nohost（TL） | 去掉 HA；VA 全局选 VM，奖励仍使用所选 Host 区间能耗；TA 保持原版 |
| ablation_tl2（TL_2） | 去掉 HA；VA 改用全局区间能耗；TA 保持原版 |

GE 改探索，PF 改候选，TS/UB 改学习设置，均不意味着奖励发生变化。OE 改为逐分配能耗归属；能耗物理计算仍使用同一 Host 任务窗口模型。

## IRWS：外部奖励和实际策略奖励需要分开

三个分支的外部奖励均为 `r_ext=R(E_assignment,D_i)`，但真正进入策略更新的融合奖励不同。内在奖励 `r_int` 是 GRU 历史网络的线性输出，没有奖励值 clip；PPO 的 ratio clip 不是奖励 clip。

| 分支 | 策略使用的奖励 | 权重规则 |
|---|---|---|
| irws | `r_ext + 0.1*r_int` | 固定0.1 |
| irws_paper | `r_ext + w*r_int` | `w=t_ref/(t_ref+rank_down)`；两者均0取0；w 在[0,1] |
| irws_restored | `w*r_int + (1-w)*r_ext` | `w=t_ref/rank_down`；分母0且分子>0取1，否则取0；不做[0,1]截断 |

`t_ref` 为任务参考执行/输入传输时长，`rank_down` 为 DAG downward rank，单位均为秒。`irws_restored` 给队列中任务预订确定的未来服务区间，外部奖励使用该分配的窗口增量和预计完成时间。

需要明确的当前实现事实：

- `irws_restored` 的 w 可大于1，此时外部奖励系数 `1-w` 为负；其融合不等价于“共同外部奖励加一个小幅内在项”。
- `irws_paper` 和 `irws_restored` 的元学习外部成本均为 `outer_cost=-r_ext`，据此更新内在网络；融合形式仍不同。
- 旧 `irws` 入口在 `torch.no_grad()` 下调用随机初始化 GRU，未找到对内在网络的优化器或更新；实际训练的是 VM PPO。因此旧分支的内在项不能称为学习得到的内在奖励。

## 代码依据

- PD3QN：[step 奖励](../environments/pd3qn.py)，`step` 中的 `r_delay`、`dE_total`、`reward`；常数来自 [train.py](../algorithms/pd3qn/train.py)。
- MAHDRL：实际配置 [train.py](../algorithms/mahdrl/train.py) 第175–181、259–264行；下层公式 [mahdrl_heft.py](../environments/mahdrl_heft.py) 第599–623行；TA [mahdrl_fcfs.py](../environments/mahdrl_fcfs.py) 第199–240行；尾结算 [f11.py](../common/f11.py) 第193–203行。环境默认 HA=.35、VA=.60 被入口覆盖，不能用于本次比较。
- OE：[mahdrl_oe.py](../environments/mahdrl_oe.py) 第233–259行；TL [ablation_nohost.py](../environments/ablation_nohost.py) 第195–220行；TL_2 [ablation_tl2.py](../environments/ablation_tl2.py) 第6–7行。
- MARL：[train.py](../algorithms/marl/train.py) 第160–162行；[marl.py](../environments/marl.py) 第371–385行及698–722行。
- DRL-GNN：[config.py](../algorithms/drl_gnn/config.py) 和 [environment.py](../algorithms/drl_gnn/environment.py) 第118行转交 IRWS。
- 共同 IRWS 外部奖励：[irws.py](../environments/irws.py) 第591–604、741–754行。旧融合 [train.py](../algorithms/irws/train.py) 第264–280行；paper 融合 [learner.py](../algorithms/irws_paper/learner.py) 第27–30行、权重 [environment.py](../algorithms/irws_paper/environment.py) 第34–40行；restored 融合 [learner.py](../algorithms/irws_restored/learner.py) 第22–24行、权重 [environment.py](../algorithms/irws_restored/environment.py) 第34–40行。
- 规则基线：[ertf_fifs/evaluate.py](../algorithms/ertf_fifs/evaluate.py) 第19、275–281、349–396行；奖励仅作记录，不训练策略。

DRL-GNN、TwoStageGAT、MARL 的训练和评测复用相同奖励配置。评测比较统一物理能耗、超期等指标，并不因为某个算法内部训练回报数值较高就判定它更好。共同权重不能证明逐步训练目标相同，能耗结算范围/时点、TA 的累计惩罚及 IRWS 的融合均需在实验中说明。
