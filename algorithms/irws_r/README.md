# IRWS-R

当前模型标识为 `irws_r_v3_rollout`，训练/评测入口仍在本目录。默认跨区间累计64条经验更新、5秒调度tick、float32神经网络计算；保留当前奖励、网络宽度、四组学习率、gamma=.2、clip=.2、历史截断16和固定任务Q头。

静态DAG缓存、拓扑层批处理、设备内观测/历史和状态版本资源缓存减少重复计算。新旧训练规则和数值精度不同，不声称算法结果等价。实现、验证和限制见 [加速实现说明](../../docs/IRWS_R_v3_implementation.md)。

```bash
python launch.py algorithms/irws_r/train.py --task S --resource S --deadline Tight --rollout-size 64 --slot-duration 5 --precision float32 --device cuda --dry-run
python -B -m unittest common.test_irws_r common.test_irws_r_speed common.test_irws_r_v2 -q
python -B -m common.smoke_irws_r --out docs/IRWS_R_v3_smoke.json
```

完整命令见 [训练与组合评测指令](../../docs/Linux_CUDA_IRWS_加速版_训练与组合评测指令.txt)。新配置需要重训，输出进入 `outputs/ckpts/irws_r/irws_r_v3_rollout/`，子目录绑定窗口长度、周期和精度。best/final文件名保持现有规则；同配置可用 `--resume-checkpoint` 指向自己的 `final_irws_r.pth` 续训。

旧批量版本保存于 `algorithms/irws_r_v2/`，通过该目录的train.py/evaluate.py继续使用原 `irws_r_v2_batch` 模型。旧逐样本版仍在 `algorithms/irws_r_v1/`。旧模型、旧结果不覆盖或自动加载。
