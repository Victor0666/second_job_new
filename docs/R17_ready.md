# R17 正式实验前置文件已准备

默认协议为 configs/f22_split_v1.json，训练0–599、验证600、测试700–729。九组缓存各覆盖0–1000，共9009个实例记录；缓存仅为问题条件准备，不是网络训练结果。

正式文件已纳入Git：

- data/deadlines/dynamic_full_parent_output_v1/：九组新版截止期缓存。
- outputs/f18_normalization_calibration/f18_shared_normalization_v1.json：各学习算法默认加载的校准配置。
- outputs/f18_calibration_r17/：同一校准配置的源文件、来源清单及四张统计表。九个源训练场景均使用101–130，共270次校准全部成功。
- outputs/r17_preparation/frozen_manifest.json：前置文件SHA256、协议哈希、校准代码指纹。

这些输出文件是经过选择的正式前置材料，因此显式纳入Git；outputs下其他模型、训练日志、270份原始校准过程记录和本地任务监控文件不纳入本次提交。无需重跑校准即可使用已提交的冻结配置训练；重新校准应选择新目录。

.gitattributes对上述冻结JSON及CSV禁用换行转换，确保下载/检出后的文件哈希不被Git自动改写。来源清单中的本机绝对路径是历史溯源信息，实际运行以项目根目录和Scenario计算的路径为准。

检查缓存：

```powershell
.\run.cmd common/f22_cache.py --all
```

开始训练示例（不需要额外指定归一化文件）：

```powershell
.\run.cmd algorithms/mahdrl/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1
```

正式训练和后续测试需统一使用冻结文件。前置文件准备及上传不代表已完成算法训练或正式性能评测。
