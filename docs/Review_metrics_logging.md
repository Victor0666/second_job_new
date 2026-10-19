# 审稿实验记录补充（review_metrics_v1）

## 是否改变实验结果

新增记录不调整奖励、动作、更新时机、学习率、截止期、随机种子或选模标准。记录器不进行额外动作选择或优化器更新，不使用随机数。CPU真实短程对照覆盖MAHDRL、PD3QN、MARL、四种原消融、TL_2及历史IRWS，记录开关前后最终参数哈希和物理指标完全一致；新版IRWS另有CPU/CUDA权重、损失、随机状态对照。

记录与CUDA同步会增加实际运行开销，不能把有记录与无记录的耗时混作同一计时协议。跨算法效率比较应使用相同硬件、同一记录模式，并避免多个训练进程争用GPU。GPU天然的非确定性不因这次记录而获得普遍确定性保证。

评测CSV的Makespan是唯一有意改变的旧输出数值口径（旧训练CSV原样保留，训练标准明细使用统一口径）：统一使用最后完成时刻减首次到达时刻。没有改变物理完成时刻；标准明细另存legacy_environment_makespan。已有能耗、成功率、超期量和各算法原CSV继续保留。

## 训练：自动生效，不需要新增参数

各训练日志目录新增：

- measurement_metadata.json：模型/源训练场景、模型种子、Python/PyTorch/CUDA、设备、网络结构、优化器参数量及实际设置、环境设置。
- training_measurements.jsonl：每个完成的episode一行。train_seconds为回合训练控制循环及有计时的reset时间，包含普通日志/记录开销；初始模型建立和维度探测不包括在内。validation_seconds和checkpoint_seconds分开记录，另有累计值、reset_seconds、assignment_calls、phase_advance_calls、policy_calls、各优化器实际step次数、optimizer_sample_presentations（含重用的样本呈现次数）、learning_samples和IRWS rollout_transition_presentations。
- metrics和validation：统一能耗、工作流总数/完成数/按时数/超期数、成功率、平均超期及响应时间，以及任务级指标。验证只记录原有验证实例和频率；不额外验证，不改变原有按验证能耗选模规则。

learning_samples沿用F11计数：transitions是记录的交互，sampled_targets是参与目标计算的样本呈现次数（包含重用，不是独立新实例数）。若显式禁用F11诊断，相应样本计数不会完整；optimizer_steps和optimizer_sample_presentations仍独立计数；DQN使用实际批量大小，MARL使用单样本，PPO按各轮实际小批量含末尾不足批次计数，新版IRWS各优化器分别记录本轮rollout样本数。IRWS按每轮rollout的transition呈现次数单独记录，不能简单把各算法的“一个update”当成同等计算量。

实际优化器次数与policy调用次数均是训练期计数，不包含验证调用。reset会结束上一回合并开启下一回合；末回合由close写出。只完成部分回合而进程被强制结束时，不会凭空补出完整回合时间。

## 评测：统一明细

MAHDRL、PD3QN、MARL、ERTF-H-FIFS、五种消融及历史IRWS的新增目录：

outputs/run/review_metrics_v1/<算法>/<scenario.output_version>/<任务资源_截止期_model模型种子>/

例如默认F22下PD3QN的SS/Tight：

outputs/run/review_metrics_v1/pd3qn/f19_shared_v1_protocol_64d2c570d2fd/SS_Tight_model1/

新版IRWS沿用其独立run目录，在各评测场景目录下新增measurements/。可以执行 `find outputs/run -path '*/episodes/seed700.json'` 定位。

每个测量目录包含：

- measurement_metadata.json：设备、模型结构、计时范围。
- episodes/seed700.json等：统一指标、模型种子/实例种子、源训练场景、实际best模型文件哈希、数据/缓存来源、平均/P95/P99决策延迟及策略调用次数。
- workflows/seed700.csv等：每个工作流的模板、到达时间、参考耗时、截止期系数、绝对截止期、完成时间、响应时间、lateness、tardiness、完成与成功标记、任务数。
- decisions/seed700.json等：逐次任务分配的决策路径秒数、状态/策略API耗时及各策略调用耗时。

决策延迟范围：从上一次任务分配调用返回，到下一次分配调用前的控制路径；扣除独立phase推进操作，排除任务运行仿真、初始reset及最终文件写入。step/phase内部提前构造缓存观测的可识别状态API成本记入下一次决策；嵌套状态API不重复计数。CUDA在计时边界同步。应在论文披露此范围，不把它解释为真实集群通信延迟。

同一模型和种子重新评测会更新对应seed文件。统计器要求指定清单内所有seed文件齐全，并拒绝混合模型、场景、checkpoint或配对时不同缓存。每个算法的模型目录必须保持固定，不要在一轮30实例评测中替换模型。

## 离线统计

```bash
python launch.py common/summarize_experiments.py \
  --directory "测量目录（包含episodes子目录）" \
  --seeds 700-729 --qos-threshold 0.99
```

输出statistical_summary.json：能耗/成功率/超期量/响应时间等的均值、样本标准差及按实例seed计算的t置信区间；还输出决策延迟分布。QoS事件明确定义为“该实例的工作流成功率>=阈值”，给出经验比例和Clopper-Pearson精确二项区间，不把一个实例中的50工作流当成50次独立试验。

增加 `--reference "参考算法测量目录"`，按相同seed配对计算差值、能耗改善百分比和未做多重比较校正的Wilcoxon p值。发表前需预先确定比较族并进行相应校正。统计基于一个训练模型的环境重复评测，不代表独立训练方差，也不保证真实概率达到0.99。

## 对已有数据及其他审稿要求的影响

冻结的九组缓存、两份相同尺度配置、来源清单和协议文件均未修改，不需要因为增加记录重新生成。冻结清单的代码指纹描述的是当时生成校准文件的源码，不将它伪装为此次日志代码版本。

已训练模型可以重新评测补齐明细，不必仅因加字段而重训；过去未记录的逐回合真实耗时无法准确事后还原。关闭新增记录可设置 `EXPERIMENT_METRICS=0`，但那次运行无法提供这些时间/预算/明细，不能和开启记录的耗时混合比较。

这次不新增更紧截止期、突发流量、权重扫描、固定排序或同步更新策略；这些仍需独立实验。也不改变现有模型选择规则。历史IRWS入口另修正了已有的save_dir未初始化即写元数据的问题，仅调整初始化顺序；正式IRWS仍使用irws_paper。

## 验证命令

```bash
python -m unittest common.test_experiment_measurements -q
python -m common.smoke_measurements --out docs/Review_metrics_training_smoke.json
python -m common.smoke_measurements --without-training-metrics --out outputs/review_metrics_control.json
python -m common.smoke_irws_paper --out docs/Review_metrics_IRWS_smoke.json
python -m unittest discover -s common -p 'test_*.py' -q
```

所有短程网络/工作流缩小仅用于验证，没有运行600episode正式训练或30seed性能实验。
