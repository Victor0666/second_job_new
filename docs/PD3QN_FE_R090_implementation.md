# PD3QN-FE-R090：0.9延迟权重与单成功率门槛

算法ID为`pd3qn_fe_r090`，模型版本为`pd3qn_fe_r090_v1`。独立入口为`algorithms/pd3qn_fe_r090/train.py`及`evaluate.py`。本次仅新增文件，不修改现有PD3QN、PD3QN-FE、PD3QN-DAG-FE或公共学习器、环境、参数解析器。

## 奖励与学习设置

复用PD3QN-FE的环境参数构造，再仅覆盖`alpha_delay_vm=0.9`。环境自动得到能耗权重`1-alpha_delay_vm`，即0.1。训练、验证与测试均走新变式自己的参数入口：

```text
r = -0.9 * clip(max(0, finish-task_subdeadline)/300, 0, 1)
    -0.1 * 0.001 * delta_host_window_energy
```

Host任务窗口能耗、通信、DAG完成事件、截止期与物理资源均保持。固定相同动作时物理结果不变，奖励改变；不同奖励训练得到的策略仍可能产生不同物理结果。

学习过程与原PD3QN-FE一致：历史PD3QN学习模式、lr=0.0003、gamma=0.99、batch256、容量120000、共享层1024/1024/512/512/256、原V/A头、Adam/ReLU、Huber、PER及IS校正、梯度裁剪10、tau=0.005软更新。预热为max(batch,容量3%)，探索率按成功更新次数衰减。没有采用DAG-FE v2的论文学习配置、78维观测或线性头。

默认使用`f19_shared_v1`，复用与原FE相同的源训练场景冻结校准；也保留原FE的显式F18/legacy入口，拒绝`pd3qn_units_v1`。SS默认81维观测不变。无需为奖励权重重新计算截止期缓存或拟合观测尺度。泛化始终使用源训练场景的校准。

## 选模规则

根据本次确认取消ESP门槛，复用DAG-FE v2的纯选模函数，规则版本`select_sr099_minenergy_v1`；仅复用此纯函数，不复用其学习配置。

- 总体工作流成功率为验证集按时完成数之和除以工作流总数之和；未完成计失败。
- 总体成功率不低于99%时可行，不要求每个实例分别达到99%，不计算ESP。
- 可行模型优于不可行模型；可行模型按平均实例总能耗更低、总体成功率更高排序。
- 尚无可行模型时按总体成功率更高、平均实例总能耗更低排序。
- 只有严格更优时替换，完全平局保留较早检查点。

原PD3QN-FE的双99%规则保持不变。默认F22训练0–599、验证600、测试700–729，600回合、模型种子1。当前单验证实例含50个工作流，99%实际要求50/50按时完成，不构成总体概率保证。

## 输出与检查点

模型和日志目录前缀为`pd3qn_fe_r090_v1_`，包含源场景、模型种子、观测模式、协议hash、能耗版本和选模规则。评测位于`outputs/run/pd3qn_fe_r090/<源模型>/<目标场景>/`。

- `best_pd3qn_fe_r090.pth`：最佳可行模型，或最佳备用模型。
- `pd3qn_fe_r090_final.pth`：最终回合。
- `train_metrics_pd3qn_fe_r090_cost.csv`：逐回合训练与验证指标，不含ESP。
- `validation_selection.jsonl`与`training_summary.json`：可复算选模记录及汇总。
- 主评测CSV例如`eval_pd3qn_fe_r090_taskS_resS_Loose_model1_seeds700-729.csv`。

场景元数据、训练/评测汇总及checkpoint同时记录模型版本和奖励配置。专属加载器在载入权重前检查算法、权重、缩放参数、规则、源场景、模型种子与验证记录，公共加载器继续校验网络与归一化。原0.75模型和其他算法检查点均拒绝混用，正式比较需重新训练。已有非空模型和评测目录拒绝覆盖，不支持恢复训练。

## 运行

在项目根目录执行。示例为SS/Tight，其他源场景使用resource=S/M/L与deadline=Tight/Medium/Loose组合。

[九源场景600回合训练指令](Linux_CUDA_PD3QN_FE_R090_600代训练指令.txt)提供9条独立nohup命令；[27场景九组三合一评测指令](Linux_CUDA_PD3QN_FE_R090_27场景共用评测指令.txt)每组依次运行S/M/L任务规模，各30个测试实例，共810次实例评测。

```bash
python -u launch.py algorithms/pd3qn_fe_r090/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --normalization-mode f19_shared_v1 --device cuda
python -u launch.py algorithms/pd3qn_fe_r090/evaluate.py --task S --resource S --deadline Tight --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 --normalization-mode f19_shared_v1 --device cuda
```

每个源模型依次评测S/M/L任务规模可得到原场景1个和泛化2个结果：

```bash
for task in S M L; do
    python -u launch.py algorithms/pd3qn_fe_r090/evaluate.py --task "$task" --resource S --deadline Tight --train-task S --train-resource S --train-deadline Tight --seeds 700-729 --model-seed 1 --normalization-mode f19_shared_v1 --device cuda || break
done
```

两入口支持`--dry-run`预览参数与路径；正式运行需要公共缓存和源场景共享校准就绪。可通过`--normalization-config`指定与原FE相同的冻结文件。显式CUDA不可用时退出。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest common.test_pd3qn_fe_r090 -v
```

专属测试覆盖仅alpha改变、原FE/DAG-FE仍为0.75、原FE双99%保留、新变式聚合SR单门槛、九源与27目标路径、固定动作物理结果相同且奖励改变、真实两回合训练、三阶段0.9权重、检查点保存加载与错误权重拒绝、S/M/L任务泛化及防覆盖。烟测使用临时校准与小网络，不写正式公共校准或结果；不证明600回合收敛或性能优劣。

本次验证：9项专属测试与66项PD3QN/PD3QN-FE/PD3QN-DAG-FE及公共场景回归全部通过。正式launch.py训练/评测入口dry-run及Python语法检查通过。修改前后486个既有源文件、配置和文档的SHA-256一致；实现、测试及说明共8个新增文件，发布时另增两份专属运行指令。未启动正式600回合或CUDA长程实验。
