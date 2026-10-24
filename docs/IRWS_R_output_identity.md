# IRWS-R 独立对比算法身份与输出隔离

2026-10-25更新：当前入口`algorithms/irws_r/`已升级为`irws_r_v2_batch`，新模型和结果进入该版本目录；旧逐样本实现保留在`algorithms/irws_r_v1/`，仍对应原`irws_r_v1`目录。v2拒绝加载v1 Checkpoint，需要重新训练。v2算法变化见[机制说明](IRWS_R_v2_implementation.md)。以下内容记录最初v1的命名与隔离方案，其中“本次仅改变身份”仅适用于当时的v1复制操作。

日期：2026-10-24（Asia/Shanghai）。此前修改的IRWS现在作为独立命名的对比算法运行，避免旧IRWS结果与新结果混淆。

| 项目 | 新标识 |
|---|---|
| 显示名称 / 算法ID | IRWS-R / irws_r |
| 代码入口 | algorithms/irws_r/train.py、evaluate.py |
| 模型版本 | irws_r_v1 |
| 最佳 / 最终模型 | best_irws_r.pth / final_irws_r.pth |
| 训练主日志 | train_irws_r_taskS_resS_Tight_model1.jsonl |
| 评测主文件 | eval_irws_r_taskS_resS_30seeds_Tight_model1.csv |
| 评测汇总 | summary_irws_r_taskS_resS_30seeds_Tight_model1.json |

新目录由`algorithms/irws_restored/`当前v2复制得到。旧`irws`、`irws_paper`、`irws_restored`及其模型、日志、评测结果不移动、不改名。复制源及公共文件哈希见`algorithms/irws_r/copied_from_sha256.json`。

本次改变运行身份、文件命名与输出隔离，没有为产生不同数值而改变奖励、调度、网络或学习参数。`environment.py`、`networks.py`、`learner.py`与复制源逐字节一致；相同初始化与实例的对照日程及物理指标一致。新名称不表示原文差异、融合权重或元梯度近似问题已经解决。

## 输出目录

```text
outputs/
  ckpts/irws_r/irws_r_v1/<源场景_协议_归一化_能耗_q_dt>/model1/
    best_irws_r.pth
    final_irws_r.pth
    train_irws_r_taskS_resS_Tight_model1.jsonl
    data_split.json
    logs/
      training_measurements.jsonl
      measurement_metadata.json
  run/irws_r/irws_r_v1/<同配置标识>/model1/SS_Tight/
    eval_irws_r_taskS_resS_30seeds_Tight_model1.csv
    summary_irws_r_taskS_resS_30seeds_Tight_model1.json
    measurements/
      episodes/seed700.json
      workflows/seed700.csv
      decisions/seed700.json
```

配置标识绑定源训练场景、实验协议、归一化指纹、能耗口径、队列容量与策略周期。不同模型种子使用不同子目录。附属文件即使沿用公共记录器的通用文件名，也只写入IRWS-R目录。

Checkpoint、训练主日志、评测CSV及汇总明确写入`algorithm=irws_r`、`display_name=IRWS-R`及模型版本。公共训练/逐种子测量记录写入`algorithm=irws_r`，测量器角色名为`IRWS-R`。Checkpoint验证算法身份、版本和模型种子。

## 覆盖与续训

- 新训练：目标训练目录存在任何内容即报错，保留已有文件。
- 续训：仅通过`--resume-checkpoint`显式指定本次目录的`final_irws_r.pth`；继续原训练，追加主日志并更新自己的模型。
- 新评测：评测入口遇到目标场景目录已有内容即报错，在打开CSV或写逐种子明细前拒绝覆盖。同一目录不同测试子集也不能静默混写。
- 旧结果：不会自动扫描、迁移或改名；旧IRWS目录非空不阻止IRWS-R首次运行。
- 旧权重：新入口查找新文件名并校验身份，不自动把旧`best.pth`或恢复版v2当作IRWS-R模型。

重复评测若要保留两次结果，应先把已有IRWS-R目标评测目录备份到另一个明确位置，再运行。更换模型种子时，先在对应种子独立训练，再以相同`--model-seed`评测。没有自动覆盖开关。

## 命令

```powershell
.\run.cmd algorithms/irws_r/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1 --queue-capacity 10 --slot-duration 1
.\run.cmd algorithms/irws_r/evaluate.py --task S --resource S --deadline Tight --seeds 700-729 --model-seed 1 --queue-capacity 10 --slot-duration 1
```

Linux使用`python launch.py algorithms/irws_r/train.py ...`或`evaluate.py ...`。公共实例、截止期缓存及归一化继续复用；训练0–599、验证600、测试700–729、每回合50工作流及原IRWS学习设置不变。跨场景评测仍以`--train-task`、`--train-resource`、`--train-deadline`指定源训练场景，加载时校验VM维度。

## 验证

专项：`.venv/Scripts/python.exe -X utf8 -B -m unittest common.test_irws_r -v`。

5项专项覆盖复制源保护、学习参数一致、同种子日程一致、输出与模型种子隔离、重复运行拒绝覆盖、Checkpoint身份与种子校验，以及训练/保存/加载/评测/续训调用链。

真实DAX短程：`.venv/Scripts/python.exe -X utf8 -B -m common.smoke_irws_r --out docs/IRWS_R_smoke.json`。缓存、模型和逐种子输出仅写临时目录，报告保留为上述JSON；不执行正式600回合或30种子性能实验。

本次验收：5项专项全部通过；训练/评测入口预览均为`IRWS-R / irws_r / irws_r_v1`。复制时32份文件的原始哈希保留；源算法与物理代码按哈希核对，允许Git检出时的LF/CRLF换行转换。共享场景注册表可随新算法扩展，单独核对其资源、截止期和数据划分保持原值。真实DAX检查完成54次分配、39次实际排队，执行时长最大误差4.619e-14、最终能耗再积分误差0，依赖及VM无重叠检查通过。训练、验证、新模型保存重载与确定性评测闭环通过；没有启动正式性能实验。
