# IRWS 输出文件与完成提示

当前正式入口为 `algorithms/irws_paper/train.py` 和 `algorithms/irws_paper/evaluate.py`。本次修改输出命名和控制台提示，训练目标、策略、参数更新、Checkpoint 格式及加载路径均保持原有规则；不需要因为文件命名修改而重新训练模型。

以 S 任务、S 资源、Tight、模型种子 1 为例：

| 产物 | 新文件名 |
| --- | --- |
| 训练逐回合日志 | `train_irws_paper_taskS_resS_Tight_model1.jsonl` |
| 30 种子评测主结果 | `eval_irws_paper_taskS_resS_30seeds_Tight_model1.csv` |
| 评测汇总 | `summary_irws_paper_taskS_resS_30seeds_Tight_model1.json` |

训练主日志保存在原模型目录，统一训练统计仍位于该目录的 `logs/`。评测主 CSV 和汇总仍位于原有目标场景目录，逐种子统一明细仍位于 `measurements/`。`best.pth`、`final.pth` 和所有现有目录不迁移，原服务器训练和评测命令仍然有效。

对 S 任务、M 资源、Loose 训练得到的模型，评测 L 任务、M 资源、Loose 时，主 CSV 为 `eval_irws_paper_taskL_resM_30seeds_Loose_model1_from_taskS_resM_Loose.csv`。目标场景、模型种子和不同于目标的源训练场景都能直接从文件名辨认，复制单个文件到本机后不会失去这些信息。实际评测种子数量替代示例中的 30。

训练启动时输出场景、总回合数、起始回合及模型/日志路径；成功完成后输出 `[IRWS train completed] 训练完成`，随后列出最佳模型、最终模型、主训练日志的绝对路径及训练统计目录。续训仍读取 `final.pth`，继续追加本场景的新名称训练日志。旧 `training.jsonl` 如存在则保留，不自动移动或合并，跨本次更新续训时历史和后续日志分别保存在旧、新文件中。

评测启动时输出模型路径及主 CSV 路径。每个种子结束立即写入并刷新 CSV，打印 `当前数量/总数量`；全部种子成功后关闭 CSV、写入汇总，再输出 `[IRWS eval completed] 评测完成`、平均能耗、CSV/JSON 绝对路径和逐种子明细目录。中途失败不生成本轮完成提示；已完成种子的 CSV 行保留。主 CSV 仍按原规则在一次新评测启动时重新写入，不新增断点续评语义。旧主结果文件不自动迁移或删除。

历史简化入口 `algorithms/irws` 同步采用 `train_irws_taskS_resS_Tight_model1.csv`、`eval_irws_taskS_resS_30seeds_Tight_model1.csv`，训练/评测成功后明确打印完成标识和绝对路径，汇总标题也按实际种子数量显示。

验证覆盖：27 个目标场景文件名可区分、模型种子及源训练场景可区分、第二个种子中断前第一个种子 CSV 已刷新、中断不误报完成、成功后主 CSV 和汇总可读取且控制台提供对应绝对路径；另运行实际短程训练、保存、加载和评测流程。没有运行正式 600 回合训练。
