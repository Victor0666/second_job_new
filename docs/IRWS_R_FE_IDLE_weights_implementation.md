# IRWS-R-FE_IDLE_010 与 IRWS-R-FE_IDLE_090

2026-10-31。两个独立版本均直接复制自当前 `algorithms/irws_r_fe_idle`，
各自的 `source_snapshot_sha256.json` 保存源 Python 文件哈希。

| 项目 | IDLE_010 | IDLE_090 |
| --- | --- | --- |
| 显示名称 | IRWS-R-FE_IDLE_010 | IRWS-R-FE_IDLE_090 |
| 算法目录 | `algorithms/irws_r_fe_idle_010` | `algorithms/irws_r_fe_idle_090` |
| 延迟权重 | 0.1 | 0.9 |
| 能耗权重 | 0.9 | 0.1 |
| 模型版本 | `irws_r_v3_fe_idle_010_v1` | `irws_r_v3_fe_idle_090_v1` |

## 奖励与保留机制

```text
r_ext = -alpha * clip(task_tardiness / 300, 0, 1)
        -(1-alpha) * 0.001 * assignment_host_window_energy
```

仅改变外部奖励中的 alpha；内外奖励融合仍为 `w*r_int+(1-w)*r_ext`，
外层代价仍为 `-r_ext`。不改 PPO、元梯度、网络、V3 rollout64、图/资源缓存或 float32 默认设置。
融合权重 w 可能大于 1 的既有性质保留。

两个版本均采用 `idle_only_periodic_v1`：只向合法空闲 VM 分配，默认 5 秒调度 tick。
所有 VM 忙碌时任务留在全局 Ready 集合，不向忙碌 VM 预留或排队。
`--queue-capacity 10` 仅保留原观测布局，VM 等待队列始终为空。
物理完成时刻仍精确记账，后续分配遵循原周期边界。

**选模继承当前 IDLE 的 92% 门槛**（`select_sr092_minenergy_v1`）：
验证集总体工作流成功率达到 92% 后选最低平均能耗；未达标时优先成功率，再比较能耗，完全平局保留较早模型。
默认单验证实例含 50 个工作流，46/50 即达到门槛；不代表对测试集成功率的保证。

## 独立模型与输出

原 IDLE、排队版 R010/R090、已有 `irws_r_fe_r090_idle` 及公共代码均保留。
这里的 `IDLE_090` 与已有 `R090_IDLE` 使用不同的算法 ID、模型版本和独立输出，加载时不得互换。
两个新版本继承相同的当前 IDLE 选模协议，奖励系数以各自配置为准。

```text
outputs/ckpts/irws_r_fe_idle_010/SS_Tight_m1_cXXXXXXXX/
outputs/ckpts/irws_r_fe_idle_090/SS_Tight_m1_cXXXXXXXX/

outputs/run/irws_r_fe_idle_010/
  eval_irws_r_fe_idle_010_MS_Tight_fromSS_Tight_m1_s700-729_cXXXXXXXX.csv

outputs/run/irws_r_fe_idle_090/
  eval_irws_r_fe_idle_090_MS_Tight_fromSS_Tight_m1_s700-729_cXXXXXXXX.csv
```

训练目录使用短配置指纹；训练指标位于模型目录的 `train.jsonl`、`logs/`。
每个算法的最终 CSV 和汇总 JSON 平铺在同一目录，文件名包含目标场景、源模型场景、模型种子、
精确测试种子和配置指纹；明细使用 `_details/<16位标识>/`。
检查点分别为 `best_irws_r_fe_idle_010.pth` / `final_irws_r_fe_idle_010.pth`，
以及 `best_irws_r_fe_idle_090.pth` / `final_irws_r_fe_idle_090.pth`。
严格校验算法/模型版本、奖励、空闲调度协议、92%选模、校准与训练数据划分；拒绝跨版本混用。
续训仅使用同次运行的 final 与历史 best；既有非空模型及评测结果拒绝覆盖。

## 训练与评测入口

每个算法目录分别提供 `train.py` 和 `evaluate.py`，在项目根目录用 `python -u launch.py <入口>` 启动。
支持 `--dry-run` 查看实际配置和路径。沿用 F22/F19 及源场景冻结校准，工作负载选项与当前 IDLE 一致。
默认九源场景训练 600 回合；各源 best 固定评测 S/M/L 任务规模，共 27 个目标，测试种子 700–729。

| 版本 | 训练函数与九组命令 | 评测函数与九组命令 |
| --- | --- | --- |
| IDLE_010 | [600回合训练](Linux_CUDA_IRWS_R_FE_IDLE_010_600代训练指令.txt) | [27场景评测](Linux_CUDA_IRWS_R_FE_IDLE_010_27场景共用评测指令.txt) |
| IDLE_090 | [600回合训练](Linux_CUDA_IRWS_R_FE_IDLE_090_600代训练指令.txt) | [27场景评测](Linux_CUDA_IRWS_R_FE_IDLE_090_27场景共用评测指令.txt) |

每次函数调用启动一个后台进程；评测第三参数可用 `"M L"` 仅补泛化。

## 验证

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest common.test_irws_r_fe_idle_weights common.test_irws_r_fe_idle common.test_irws_r_fe_r090_idle -v
```

专项覆盖直接来源、空闲掩码与忙 VM 原子拒绝、5秒 tick、固定动作的物理轨迹一致性与新奖励，
真实短程训练/续训/保存/加载/冻结评测、错误奖励/调度/选模及跨版本模型拒绝、九源27目标命名和 CUDA 更新。
测试使用临时实例与校准，不替代正式 600 回合实验。本次未启动正式训练或评测。

验证结果：37 项新专项及原 IDLE / 既有 R090_IDLE 回归全部通过，无跳过，包含两个新版本的实际 CUDA 更新。
四个真实 `launch.py` 入口 dry-run、23 个新增 Python 文件语法、四份 Bash 语法检查通过；
函数展开的 18 条训练和 54 条评测命令全部通过真实 CLI 参数检查。
两个版本的源 IDLE 文件哈希均与创建时快照一致。
输出路径逐场景核查：每算法九个源训练目录末级长度为 21–22 个字符，
相对项目根目录最长 55 个字符；27 个最终 CSV 均在同一算法目录，且文件名全部不同。
后台日志名也已缩短，保留算法、源场景、模型种子及评测目标与种子范围。
详见 [验证报告](IRWS_R_FE_IDLE_weights_validation.json)。发布分支为 `codex/f11-real-transitions`，提交状态以 Git 历史为准。
