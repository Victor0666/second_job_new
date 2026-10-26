# PD3QN-DAG-FE：PD3QN-DAG 的可行性优先选模变式

PD3QN-DAG-FE（算法ID：`pd3qn_dag_fe`）在PD3QN-DAG基础上使用双99%选模规则。实现、测试和指令均为新增文件，不修改PD3QN、PD3QN-DAG、PD3QN-FE及公共环境、学习器或参数解析器。

## 学习过程

直接复用原PD3QN-DAG的环境、`D3QNAgent`和`agent_kwargs`，固定`paper_mode=True`。学习率0.01、折扣0.95、batch 128、回放容量10000；隐层1024/1024/512/512/256；PER无IS校正、MSE、无梯度裁剪，target最大合法Q构造目标。目标网络硬同步周期60，时钟每训练回合重置；warmup只要求回放达到batch 128，沿用原DAG的每决策更新顺序。

观测固定为`pd3qn_units_v1`，含父数据与Host历史窗口，SS为112维。拒绝历史PD3QN的legacy/f18/f19观测；不需要F19训练校准文件。物理能耗、奖励系数、截止期生成和每实例50个工作流均复用原DAG。

默认数据划分为F22：训练seed0–599、固定验证seed600、测试seed700–729。默认600回合、模型初始化种子1。仅支持互不重叠的固定协议。

## 检查点选择

完全复用PD3QN-FE的纯选模函数，规则版本`select_sr099_esp099_v1`。每训练回合结束后确定性验证，不更新权重、回放或探索计数；验证前后恢复Python、NumPy、Torch CPU/CUDA随机状态和网络模式。

每实例成功率=按时完成工作流数/该实例工作流总数，未完成计失败。成功率不低于99%的实例占比为ESP；ESP不低于99%时模型可行。

- 可行模型始终优于不可行模型；可行模型按验证平均总能耗更低、总体工作流成功率更高排序。
- 尚无可行模型时，按ESP更高、总体成功率更高、平均总能耗更低排序。
- 只有严格更优才替换，完全平局保留较早检查点。

总体成功率按验证集总成功数除以总工作流数；平均能耗按各实例总能耗取算术平均。当前固定验证集只有seed600的50个工作流，因此双99%实际要求50/50按时完成，ESP只能为0或1；它是经验指标，不构成总体概率保证。

## 独立输出与加载

模型、日志、结果使用`pd3qn_dag_fe`专属目录，源训练场景、目标场景和模型种子分别标识。所有非空训练/评测目录均拒绝覆盖。

- `best_pd3qn_dag_fe.pth`：最优可行模型，或尚无可行模型时的最优备用模型。
- `pd3qn_dag_fe_final.pth`：最后回合模型。
- `train_metrics_pd3qn_dag_fe_cost.csv`：逐回合训练指标、验证能耗、成功率、ESP、是否可行、是否选中及最佳回合。
- `validation_selection.jsonl`：逐实例验证明细及完整选模记录，可独立复算。
- `training_summary.json`：规则、数据划分、最佳回合和`feasible_model_found`。

沿用原Agent保存机制，基础学习实现标记仍为`pd3qn_dag_v1`，新变式身份、规则、源场景、模型种子、检查点角色、回合及验证结果独立保存在`normalization.pd3qn_dag_fe`。专属加载器在载入权重前校验这些元数据，拒绝原PD3QN、PD3QN-DAG、PD3QN-FE和错误规则的模型。既有DAG模型无法补回未保存的逐回合候选，正式比较需要在新目录重训；不支持恢复训练。

评测结果位于`outputs/run/pd3qn_dag_fe/<源模型标识>/<目标场景与种子>/`，主CSV名包含目标场景、模型种子和测试种子，例如`eval_pd3qn_dag_fe_taskS_resS_Loose_model1_seeds700-729.csv`；同目录还保存`summary_pd3qn_dag_fe.json`及工作流/时间测量明细。关闭公共测量日志不影响选模。

## 运行

[九个源模型的600回合训练指令](Linux_CUDA_PD3QN_DAG_FE_600代训练指令.txt)包含9条独立后台命令；[27场景共用评测指令](Linux_CUDA_PD3QN_DAG_FE_27场景共用评测指令.txt)包含9条分组后台命令，每条依次评测S/M/L，得到1个原训练规模结果和2个泛化结果。每场景测试seed700–729，共27×30=810次实例评测。

Windows单场景示例（项目根目录）：

```powershell
.\run.cmd algorithms/pd3qn_dag_fe/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --normalization-mode pd3qn_units_v1
.\run.cmd algorithms/pd3qn_dag_fe/evaluate.py --task S --resource S --deadline Tight --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 --normalization-mode pd3qn_units_v1
```

Linux/CUDA单场景示例：

```bash
python -u launch.py algorithms/pd3qn_dag_fe/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --normalization-mode pd3qn_units_v1 --device cuda
python -u launch.py algorithms/pd3qn_dag_fe/evaluate.py --task S --resource S --deadline Tight --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 --normalization-mode pd3qn_units_v1 --device cuda
```

追加`--dry-run`只预览身份、学习配置、划分和路径。正式入口检查对应公共截止期缓存；无需新增依赖或校准。显式`--device cuda`在CUDA不可用时退出。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest common.test_pd3qn_dag_fe -v
```

测试覆盖与原DAG配置一致、历史模式拒绝、双99%纯函数复用、九源模型与27目标路径隔离、数据分区，以及真实环境中的两回合训练、梯度更新、检查点保存加载和S/M/L三种任务规模评测。烟雾检查只把隐层改为16/16，保留batch128、容量10000、学习率0.01、折扣0.95和同步周期60；正式入口使用完整原DAG网络。短训练验证流程，不证明600回合的收敛或可行性。

本次验证：新增专属9项测试全部通过；原PD3QN、PD3QN-DAG、PD3QN-FE及公共场景共49项回归测试全部通过。9条训练命令和9条S/M/L分组评测命令的36次参数展开、27个独立结果路径及Bash循环语法检查通过，9组源训练缓存预检查通过。实施前后所审计的390个既有算法、公共代码、配置和文档文件SHA-256全部一致；本次仅新增10个文件，未启动600回合正式实验。
