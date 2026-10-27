# IRWS-R-FE_R090：超期奖励权重0.9的独立变体

2026-10-27。以IRWS-R-FE为基础复制，算法ID `irws_r_fe_r090`，
显示名 `IRWS-R-FE_R090`，模型版本 `irws_r_v3_fe_r090_v1`。
原FE（0.75）、V3、其他算法与公共物理/校准/实验协议文件保留。
源副本哈希见 `algorithms/irws_r_fe_r090/source_snapshot_sha256.json`。

## 奖励变化

局部 `COMMON_REWARD` 为 `alpha_delay_vm=.9, energy_reward_scale=.001, task_baseline_norm=300.`。
正式训练和评测入口通过原有env_kwargs显式传参，覆盖公共环境默认的0.75；无需修改公共环境。
外部奖励为 `-0.1 * 0.001 * delta_energy - 0.9 * clip(task_tardiness / 300, 0, 1)`。
外层代价仍为外部奖励的相反数，RNN历史接收新奖励，因此更新会响应新权重。
不改变内外奖励融合权重w、Rank零边界、PPO及元梯度的计算结构。

学习器、网络、环境、诊断和选模模块逐字节复制原FE。99%总体工作流成功率门槛、
可行后最低能耗、未可行时成功率优先、完全平局保留较早模型，以及验证隔离和续训历史保持。
V3加速配置和论文学习参数保持，包括默认rollout64、5秒调度周期、float32和四组学习率。
共享校准、任务实例、截止期、能耗计量及F22划分继续沿用。

## 输出与兼容

训练/评测输出独立进入 `outputs/{ckpts,run}/irws_r_fe_r090/irws_r_v3_fe_r090_v1/`。
best为 `best_irws_r_fe_r090.pth`，final/续训为 `final_irws_r_fe_r090.pth`。
控制台日志和指标文件也包含r090标识；CLI dry-run展示0.9权重及独立路径。
检查点保存奖励设置；版本、角色、奖励、校准、队列/周期/rollout/精度及验证记录均继续校验。
旧FE/V3和0.75检查点拒绝加载。同配置非空训练/评测目录拒绝覆盖；续训只使用本次运行的final。

## 运行

在仓库根目录执行，正式入口沿用已有公共缓存与冻结校准：

```bash
python -u launch.py algorithms/irws_r_fe_r090/train.py \
  --task S --resource M --deadline Tight --episodes 600 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 \
  --queue-capacity 10 --slot-duration 5 --rollout-size 64 --precision float32 --device cuda

python -u launch.py algorithms/irws_r_fe_r090/evaluate.py \
  --task L --resource M --deadline Tight \
  --train-task S --train-resource M --train-deadline Tight --seeds 700-729 --model-seed 1 \
  --experiment-protocol configs/f22_split_v1.json --normalization-mode f19_shared_v1 \
  --queue-capacity 10 --slot-duration 5 --rollout-size 64 --precision float32 --device cuda
```

[9源场景后台训练函数](Linux_CUDA_IRWS_R_FE_R090_600代训练指令.txt)和
[27目标场景后台评测函数](Linux_CUDA_IRWS_R_FE_R090_27场景共用评测指令.txt)为专属指令。
原FE入口仍运行原0.75版本。新变体从头训练，选定同一源best后固定用于S/M/L目标，不按测试结果重选。

## 验证范围

专项检查固定动作下公共物理轨迹不变、奖励及outer_cost正确改变、独立路径和旧权重拒绝，
以及训练/保存/续训/评测和CUDA短程更新。临时工作流与测试尺度仅用于短测，不写公共正式数据。

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest common.test_irws_r_fe_r090 common.test_irws_r_fe -v
```

未运行正式600回合或30测试种子实验；0.9权重不保证成功率提高。
当前融合公式w可能大于1的性质保持；这是奖励权重敏感性变体，不代表其他算法的公共奖励权重也已改变。

本次5项R090专项与13项原FE回归共18项通过，包含本地CUDA实际更新；
两份后台指令的Bash语法、9条训练和27条评测的实际CLI/源模型映射均通过。
302个既有代码/配置/环境文件的SHA-256保持，新增Python文件语法检查通过。
