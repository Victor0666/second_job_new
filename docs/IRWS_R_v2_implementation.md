> 历史批量版已保存在 `algorithms/irws_r_v2/`；当前入口的加速修订见 [新实现说明](IRWS_R_v3_implementation.md)。本页保留原验收记录。

# IRWS-R v2 批量更新与最小机制补全

日期：2026-10-25。算法ID仍为`irws_r`，显示名IRWS-R，模型版本`irws_r_v2_batch`。这是依据用户确认方案的实质修订，不宣称与论文作者源码等价。

## 保留范围

- 修改前的逐样本实现及入口保存在`algorithms/irws_r_v1/`，副本文件哈希见该目录`snapshot_sha256.json`。
- 原`irws`、`irws_paper`、`irws_restored`及公共仿真物理计算不修改。
- 外部奖励保持公共能耗增量与任务超期软惩罚；原式融合`w*r_in+(1-w)*r_ext`及零rank边界不变。
- FIFO忙VM队列、容量10、策略周期1秒、精确物理事件推进不变。
- Q继续固定随机初始化权重，由模型种子控制；上游图嵌入训练。不添加任务概率联合PPO、熵项或额外截止期观测。
- gamma=.2、clip=.2、batch上限64、四组学习率、网络宽度、trajectory=16、float64不变。

## 1. 图聚合的最小补全

每个方向按拓扑顺序对前驱已计算状态取均值，无前驱用零向量。再计算`softmax(Wself(previous[v])+Wneighbor(mean_predecessors),dim=-1)`，直接作为式(17)的聚合向量。保留既有双向两token GRU和方向均值池化。

均值归约、特征维Softmax与无前驱零向量是式(16)未完整规定处的工程约定，不是作者明确参数。该实现消除v1中自身注意力权重被Softmax平移抵消的退化。

## 2. 有限区间批量元梯度

一个策略区间收集完成后，无放回随机抽取`min(64,n)`条；默认S=1。每轮使用一批平均损失更新Critic和Actor各一次，然后更新内在奖励与嵌入各一次。没有改为跨区间凑满64条。

行为优势、旧动作log概率和成本在本轮更新前固定。每轮将所需DAG编码一次，按真实时间顺序重建RNN轨迹；每16步以已记录的行为hidden作为截断起点。内外层Actor均使用同一轮重新编码后的观测快照。旧log概率始终保留实际行为值。

外层使用实际批量Adam更新后的Actor，按式(35)/(38)的概率比、裁剪条件和优势构造整个已观测区间的平均score梯度。本区间内原式区间折扣因子为1；未观察的后续区间不补造成本。该截断不等价于完整长期目标。

对每个选中样本j，先计算其更新前Actor score与外层梯度的内积，再乘该样本融合权重和直接内在奖励偏导，最后对样本梯度取平均。内在奖励参数按公共成本下降，嵌入参数按聚合奖励上升。局部Jacobian沿用v1的直接内在奖励偏导解释；不穿透Adam全部历史，不增加虚拟SGD代理。不将“各因子分别平均后的乘积”作为样本梯度平均。

Algorithm 1把更新放在逐样本循环内，表III及损失公式又使用mini-batch/期望。v2明确选择批平均计算：与v1连续多次单样本Adam不等价，属于训练实现适配。

## 3. 区间边界与实际RNN历史

当前区间结束后，更新前只读获取下一Ready集合，不调用`env.select_task`，不采样任务，也不消耗随机数。按旧Q分布计算候选任务Critic值的期望，作为本区间末步bootstrap。

末步内在奖励也对候选任务分别计算RNN输出后取概率加权期望，不把平均状态代入非线性RNN冒充这一期望。该记录显式保存候选集合，`done=False`、实际`next_decision=None`。这是一项论文未写全处的边界约定，不伪造终止或未来轨迹。

完成本区间网络更新并清空缓存后，才重新编码并选择下一实际任务。RNN的跨区间hidden随后用该真实任务及更新后的RNN重算末条token，再继续收集。训练中使用的边界奖励期望与更新后的真实历史延续明确分开；不把估值候选任务当成已经执行的后继。

## 输出与旧模型

新输出进入`outputs/{ckpts,run}/irws_r/irws_r_v2_batch/<配置标识>/model<seed>/`。Checkpoint、训练结果及评测汇总记录`implementation`，包含聚合、更新、元梯度及边界规则。原v1目录和文件不移动、不覆盖。

v2检查版本和实现标识，拒绝加载v1及其他IRWS模型。新训练须从头开始。v1模型继续通过`algorithms/irws_r_v1/evaluate.py`评测；续训通过该副本的`train.py --resume-checkpoint ...`。

训练、评测命令及公共配置参数沿用原入口，例如：

```bash
python launch.py algorithms/irws_r/train.py --task S --resource S --deadline Tight --model-seed 1 --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 --episodes 600 --queue-capacity 10 --slot-duration 1 --device cuda
python launch.py algorithms/irws_r/evaluate.py --task S --resource S --deadline Tight --model-seed 1 --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 --seeds 700-729 --queue-capacity 10 --slot-duration 1 --device cuda
```

外部nohup控制台日志由shell重定向管理，建议另用`irws_r_v2_...log`及`.pid`，以免旧命令中的`>`覆盖旧控制台日志；算法内部版本目录保护不能保护用户自定义的shell重定向路径。

## 验证

专项入口：`python -B -m unittest common.test_irws_r common.test_irws_r_v1 common.test_irws_restored -v`。

最终31项全部通过，包含80条区间经验无放回抽64条、超过16步的RNN截断重放，以及保存加载后继续训练与不中断训练逐参数一致。v1/v2的训练和评测CLI预览均已核对。

真实DAX闭环：`python -B -m common.smoke_irws_r --out docs/IRWS_R_v2_smoke.json`，仅使用临时缓存、两个工作流和一个训练回合，不启动正式600回合训练。

验证覆盖图自身分支的有效梯度、逐样本元梯度汇总的有限差分、Actor批平均Adam参照、内外编码一致、边界期望与实际历史、更新后再选择、实际短batch预算、固定Q、公共物理执行、保存加载评测及旧输出保护。测试通过仅证明这些实现性质，不证明长期梯度等价或调度性能优于v1。

真实DAX短程完成54次分配、39次排队，执行时长最大误差约5.68e-14，能耗再积分误差0；任务依赖和VM无重叠成立。四个可学习网络组均改变，Q变化为0，54条经验汇总成18次批量更新，保存加载后的确定性评测一致。

## 本地CUDA速度对照

报告见`IRWS_R_v2_performance.json`。同一台本地GPU、CPU线程1、数值诊断开启，5个真实S档DAX工作流，每版本两次交替测试。每次均完成138次分配，使用临时固定尺度和无截止期缓存的诊断环境；策略和日程允许不同，不属于正式S/S/Tight性能或收敛实验。

v1平均15.60秒，v2平均7.24秒，约2.15倍速度。每个优化器的step由138次降为49次，图编码调用由297次降为178次。保留区间更新并计算边界候选期望后，不会等同于旧`irws_paper`的跨区间凑64条更新频率；该短测不能外推为5090六进程或600回合的相同加速倍数。
