# MAHDRL-TS 与 MAHDRL-UB 训练、评测指令

在项目根目录运行。以下示例使用 SM / Tight、模型种子 1、F22 数据划分和 F19 共享归一化；两个算法分别训练自己的模型。

## Windows PowerShell

```powershell
# MAHDRL-TS：600 回合训练
.\run.cmd algorithms/mahdrl_ts/train.py --task S --resource M --deadline Tight --episodes 600 --model-seed 1 --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1

# MAHDRL-TS：加载本变式 best 模型，评测 30 个实例
.\run.cmd algorithms/mahdrl_ts/evaluate.py --task S --resource M --deadline Tight --seeds 700-729 --model-seed 1 --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1

# MAHDRL-UB：600 回合训练
.\run.cmd algorithms/mahdrl_ub/train.py --task S --resource M --deadline Tight --episodes 600 --model-seed 1 --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1

# MAHDRL-UB：加载本变式 best 模型，评测 30 个实例
.\run.cmd algorithms/mahdrl_ub/evaluate.py --task S --resource M --deadline Tight --seeds 700-729 --model-seed 1 --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1
```

## Linux / 已激活的 Python 环境

```bash
# MAHDRL-TS
python -u launch.py algorithms/mahdrl_ts/train.py --task S --resource M --deadline Tight --episodes 600 --model-seed 1 --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1
python -u launch.py algorithms/mahdrl_ts/evaluate.py --task S --resource M --deadline Tight --seeds 700-729 --model-seed 1 --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1

# MAHDRL-UB
python -u launch.py algorithms/mahdrl_ub/train.py --task S --resource M --deadline Tight --episodes 600 --model-seed 1 --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1
python -u launch.py algorithms/mahdrl_ub/evaluate.py --task S --resource M --deadline Tight --seeds 700-729 --model-seed 1 --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1
```

GPU 训练前可运行 `python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"` 检查当前环境。Linux 可在命令前加 `CUDA_VISIBLE_DEVICES=0` 选择可见 GPU；这两个算法自动选择可用 CUDA。

## 切换场景与注意事项

- 九场景：固定 `--task S`，将 `--resource` 设为 `S/M/L`，与 `--deadline Tight/Medium/Loose` 组合；每个场景训练后使用相同资源和截止期评测。
- 泛化评测：将评测的 `--task` 改为 `M` 或 `L`，加 `--train-task S`，仍加载 S 任务训练的模型。
- 配置预览：在任一命令末尾加 `--dry-run`，打印模型路径与协议，不运行训练或评测。
- 短程筛查：把训练的 `--episodes 600` 改为 `50`。50 与 600 回合使用相同场景目录，不能直接接续运行；正式重训前先归档该变式自身的非空模型和日志目录。入口拒绝覆盖且不支持续训。
- 修改 `--model-seed` 时，训练和评测保持一致；不同模型种子在当前公共命名规则下不会自动获得独立输出目录，应分别归档或在独立工作目录运行。
- 模型、训练日志和评测输出均按 `mahdrl_ts` / `mahdrl_ub` 隔离。旧 MAHDRL 模型不能替代这两个变式各自重新训练的模型。
- 700–729 是当前协议中的 30 个评测实例，已经在前期比较中查看过；最终独立验证应统一使用未参与方案选择的实例。

实现细节见 [MAHDRL-TS](MAHDRL_TS.md) 和 [MAHDRL-UB](MAHDRL_UB.md)。
