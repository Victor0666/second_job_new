# PD3QN-DAG-FE v2：精简观测、共享校准和单成功率门槛

算法ID仍为 `pd3qn_dag_fe`，入口仍为 `algorithms/pd3qn_dag_fe/train.py`、`evaluate.py`。本次直接更新该算法，不新建其他算法ID；模型版本为 `pd3qn_dag_fe_v2`。修改只涉及该算法目录、专属测试和本文及两份专属运行指令，公共学习器、环境、参数解析器以及PD3QN、PD3QN-DAG、PD3QN-FE保持不变。

## 网络与学习

- 共享网络保留五层 `1024/1024/512/512/256`，每层ReLU。
- V头直接 `Linear(256,1)`，A头直接 `Linear(256,N_VM)`，取消两个头原有的额外256维隐层，保留 `V+A-mean(A)` 合成。
- Adam、学习率0.01、折扣0.95、batch128、回放容量10000、无权MSE、无IS校正、无梯度裁剪。
- PER指数0.6、正数项1e-6，新经验按自身TD误差入池；回放达到128条后每决策更新。
- 探索率从1到0.05，按80000次训练决策衰减，验证和测试不计数。这里不是MAHDRL按成功更新次数衰减的时钟。
- 目标按原论文式(29)取target网络下一状态最大合法Q；终止或空后继动作集不bootstrap。目标网络硬复制周期60，回合内决策时钟重置。

局部 `config.py` 明确固定论文学习设置，避免切换共享归一化后公共配置选到历史学习模式。`agent.py`继承公共学习器，仅替换本算法实例的两个输出头，并重建Adam使其绑定新参数；保留原网络对象上的数值诊断hook。SS在线网络参数为2,055,962。

## 精简观测与共享归一化

布局版本 `task3_vm3_shared_v1`，顺序为：

1. 任务工作量、总输入数据量、任务子截止期减当前时间的Slack。
2. 按VM顺序展平的剩余占用时间、算力、带宽。

维度为 `3+3*N_VM`，资源S/M/L分别为78/153/228维。删除Ready数、独立当前时间、单独输出数据量、逐VM父数据分布和每Host的使用标记/历史空闲间隔。

局部环境子类只更改观测构造；内部当前时间、输出数据、父任务位置和Host历史能耗账本继续用于原物理计算。FCFS/ERTF、空闲VM合法动作、父任务真实完成事件、同VM父子数据免传输、截止期、Host任务窗口能耗及奖励均保持。删去这些输入会减少信息，不能视为无损压缩或完整MDP观测。

仅允许 `f19_shared_v1`，与MAHDRL使用同一源训练场景的冻结共享校准文件。MI、输入bits、Slack/占用时间、PC、BW分别除对应参考尺度；保留负Slack及大于1的合法值。不使用整体L2、旧固定单位或在线拟合。

默认配置路径仍为 `outputs/f18_normalization_calibration/f18_shared_normalization_v1.json`，文件名中的F18表示兼容的尺度文件格式，输入编码版本为F19。正式入口校验场景、单位、文件hash及F22训练分区绑定。已完成的公共校准直接复用；缺失时先用既有校准工具准备，不生成临时替代值。泛化到M/L任务规模仍使用S源训练场景尺度，禁止用目标验证/测试实例重拟合。

## 单成功率选模

规则版本 `select_sr099_minenergy_v1`。每训练回合结束后确定性评估固定验证集：

- 总体成功率 = 所有验证实例按时完成工作流数之和 / 工作流总数之和，未完成计失败。
- 总体成功率不低于99%时，当前模型可行。不计算ESP，不要求每个实例分别达到99%。
- 可行模型优于不可行模型；可行模型按平均实例总能耗更低、总体成功率更高排序。
- 尚无可行模型时，按总体成功率更高、平均实例总能耗更低排序。
- 只有严格更优才替换，完全平局保留较早检查点。

当前F22验证只有seed600的一个实例、50个工作流，因此可行实际要求50/50按时完成，当前验证集下新旧选模排序一致；多实例时两种规则可不同。经验成功率不构成总体概率保证。

默认训练0–599、验证600、测试700–729，600回合、模型种子1不变。验证恢复Python/NumPy/Torch CPU/CUDA随机状态与网络模式，不更新权重、回放和训练时钟。

## 输出、版本及兼容性

检查点和日志目录以 `pd3qn_dag_fe_v2_` 开头，含源场景、模型种子、F19编码、协议hash、能耗版本和新选模规则；与旧DAG-FE目录隔离。评测目录继续区分源模型与目标场景。

- `best_pd3qn_dag_fe.pth`：最佳可行模型，或尚无可行模型时的最佳备用模型。
- `pd3qn_dag_fe_final.pth`：最终回合模型。
- `train_metrics_pd3qn_dag_fe_cost.csv`：训练指标、验证能耗/成功率/可行标记及选中记录，不再含eval_ESP。
- `validation_selection.jsonl`：逐实例验证明细与可复算的选模记录。
- `training_summary.json`：最佳回合、实际规则及feasible_model_found。
- 主评测CSV保留场景化文件名，例如 `eval_pd3qn_dag_fe_taskS_resS_Loose_model1_seeds700-729.csv`。

复用公共序列化，基础学习步骤标记 `pd3qn_implementation` 仍为 `pd3qn_dag_v1`；当前模型版本、观测布局、线性头及选择规则分别写入normalization与专属元数据。加载前校验这些信息、源场景、模型种子、校准hash及网络维度，拒绝旧DAG-FE、其他算法与错误配置。输入和输出头结构已变，必须重新训练，不支持把旧权重改标签复用或恢复训练。训练/评测非空目录继续拒绝覆盖。

## 运行与验证

[九源场景600回合训练指令](Linux_CUDA_PD3QN_DAG_FE_600代训练指令.txt)和[27场景分组评测指令](Linux_CUDA_PD3QN_DAG_FE_27场景共用评测指令.txt)已同步为v2/F19/单成功率门槛；每条分组评测依次获得S/M/L三个结果。

Linux/CUDA单场景示例：

```bash
python -u launch.py algorithms/pd3qn_dag_fe/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --normalization-mode f19_shared_v1 --device cuda
python -u launch.py algorithms/pd3qn_dag_fe/evaluate.py --task S --resource S --deadline Tight --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 --normalization-mode f19_shared_v1 --device cuda
```

两入口支持 `--dry-run` 预览新配置与路径；正式运行需公共截止期缓存及源场景校准就绪。显式CUDA不可用时退出。

```powershell
.\.venv\Scripts\python.exe -m unittest common.test_pd3qn_dag_fe -v
.\.venv\Scripts\python.exe -m unittest common.test_pd3qn common.test_pd3qn_paper common.test_pd3qn_fe common.test_scenarios -v
```

专属测试覆盖聚合99%边界、与原FE多实例选模差异、加权成功率、能耗/平局排序、共享归一化与论文学习参数解耦、78维观测、固定动作物理轨迹一致、完整五层线性头梯度更新、128条预热、真实两回合训练保存加载、旧版本拒绝及S/M/L任务泛化。测试校准仅为临时夹具，不写公共正式校准；两回合闭环使用16/16共享层，另有完整生产网络的一次实际更新。未启动正式600回合或CUDA长程实验。

本次验证：16项专属测试、50项原PD3QN/PD3QN-DAG/PD3QN-FE及公共场景回归全部通过，Python语法与补丁格式检查通过。修改前后270个其他算法、公共源文件及配置的SHA-256均一致。
