# 三个独立 R010 变式

日期：2026-10-28。延迟权重由原来的 0.75 设为 0.1，能耗权重由 0.25 设为 0.9。
沿用对应 R090 版本的训练与评测流程；既有 0.75、R090 算法、公共环境和结果均保留。

| 变式 | 算法 ID / 入口目录 |
| --- | --- |
| PD3QN-FE-R010 | `algorithms/pd3qn_fe_r010` |
| IRWS-R-FE_R010 | `algorithms/irws_r_fe_r010` |
| MAHDRL-AS-Fast-R010-NoPF | `algorithms/mahdrl_oe_ge_score_fast_r010_nopf` |

每个目录分别提供 `train.py`、`evaluate.py`，使用项目根目录的 `launch.py` 启动。
独立算法 ID、模型版本、奖励元数据和输出路径隔离全部模型与结果，旧权重不能用于新变式评测。

## 奖励

PD3QN、IRWS 外部奖励、MAHDRL HA/VA 的逐分配奖励为：

```text
L_i = max(0, task_finish_i - task_subdeadline_i)
r = -0.9 * 0.001 * assignment_energy - 0.1 * clip(L_i / 300, 0, 1)
```

Host 任务窗口能耗、通信规则、截止期缓存、F22 数据划分及源场景冻结归一化保持。
MAHDRL 三层时延系数均为 0.1，TA 使用阶段能耗和阶段内新完成任务的累计超期：

```text
r_TA = -0.9 * 0.001 * phase_energy - 0.1 * phase_tardiness_sum / 300
```

TA 超期项不截断；下层阶段尾及强制分配的能耗补结算同样使用 0.9。
IRWS 保留 `w*r_int + (1-w)*r_ext` 融合和 `outer_cost=-r_ext`，0.1/0.9 仅描述外部奖励。
当前融合权重可能大于 1 的性质保持，不能把最终策略回报解释为凸组合。

## MAHDRL 取消两级 PF/APF

保留 TA 学习排序、HA 学习选主机、VA 学习选 VM，以及原共享打分网络、OE 账本和 Fast 实现。
关闭主机和 VM 两级速度阈值筛选、自适应阈值及按时最低能耗候选保留通道。
HA 可选全部包含合法空闲 VM 的主机；VA 可选所选主机内全部合法空闲 VM。
忙碌、物理非法和填充动作仍被屏蔽，空闲判断沿用 `current_time + 1e-12`。

HA 摘要由本主机全部合法空闲 VM 计算，观测维度不变：HA 为 23 维上下文和 10 维候选，
VA 为 30 维上下文和 7 维候选。探索、贪心、Dueling 合法动作均值及 TD 后继统一使用新掩码。
GE 保留原 epsilon 规则与 0.5 引导概率，在全部合法候选内执行；验证和测试使用贪心动作。

元数据明确 `host_apf_enabled=false`、`vm_apf_enabled=false`，候选诊断写入
`candidates_episode*.json` / `candidates_seed*.json`，`r_H`、`r_V` 为 null。
训练、验证、评测不调用 PF/APF 算法。模型校验同时检查奖励、筛选状态、角色、保存组及验证证据。

## 训练与评测

三者均沿用 R090 的总体工作流成功率 99% 门槛：达标后优先最低平均能耗，
未达标时优先成功率、再比较能耗，完全平局保留较早模型。PD3QN 不引入 ESP 门槛。
默认九个源场景为 task=S、resource=S/M/L × deadline=Tight/Medium/Loose，每模型训练 600 回合。
每个源 best 模型固定用于 S/M/L 三个任务规模，共 27 个目标场景；测试种子 700–729，模型种子 1。
源资源、源截止期、归一化和协议保持一致，不按测试结果重选模型。

在项目根目录、已激活的 Python/CUDA 环境使用以下专属后台指令，按 GPU 容量分批启动：

| 算法 | 9 场景训练 | 27 场景评测 |
| --- | --- | --- |
| PD3QN-FE-R010 | [600 回合训练](Linux_CUDA_PD3QN_FE_R010_600代训练指令.txt) | [27 场景评测](Linux_CUDA_PD3QN_FE_R010_27场景共用评测指令.txt) |
| IRWS-R-FE_R010 | [600 回合训练](Linux_CUDA_IRWS_R_FE_R010_600代训练指令.txt) | [27 场景评测](Linux_CUDA_IRWS_R_FE_R010_27场景共用评测指令.txt) |
| MAHDRL-AS-Fast-R010-NoPF | [600 回合训练](Linux_CUDA_MAHDRL_AS_Fast_R010_NoPF_600代训练指令.txt) | [27 场景评测](Linux_CUDA_MAHDRL_AS_Fast_R010_NoPF_27场景共用评测指令.txt) |

入口支持 `--dry-run` 查看配置和实际输出路径；F22 缓存和 F19 校准复用既有对应场景文件。
所有正式入口拒绝覆盖非空输出目录。IRWS 沿用同次运行 final 的续训功能；PD3QN、MAHDRL 不新增续训。

## 紧凑输出路径（2026-10-29）

训练目录统一采用 `源场景_m模型种子_c8位配置码`，最后一级不超过 40 个字符。
配置码涵盖模型版本、实验协议、归一化模式与校准内容、奖励及相应学习设置；完整信息继续保存在元数据中。
相同源场景的训练和 S/M/L 泛化评测定位同一源模型。移动项目或校准文件且内容不变时，配置码不变。

以下 `cXXXXXXXX` 为配置码示意，实际值可通过 `--dry-run` 查看：

| 算法 | SS/Tight 模型目录 |
| --- | --- |
| PD3QN | `outputs/ckpts/pd3qn_fe_r010/SS_Tight_m1_cXXXXXXXX/` |
| IRWS | `outputs/ckpts/irws_r_fe_r010/SS_Tight_m1_cXXXXXXXX/` |
| MAHDRL | `outputs/ckpts/mahdrl_r010_nopf/SS_Tight_m1_cXXXXXXXX/` |

PD3QN、MAHDRL 的训练日志使用对应的 `outputs/logs/算法/同名训练目录/`；
IRWS 的统计日志沿用模型目录内的 `logs/`。配套 Bash 指令的控制台日志和 PID 文件名也已缩短。

最终评测 CSV 平铺保存在每个算法的同一目录中，文件名包括算法、目标场景、源场景、模型种子、
实际测试种子和配置码，不需要依赖父文件夹判断场景。例如：

```text
outputs/run/pd3qn_fe_r010/
  eval_pd3qn_fe_r010_SS_Tight_fromSS_Tight_m1_s700-729_cXXXXXXXX.csv
  eval_pd3qn_fe_r010_MS_Tight_fromSS_Tight_m1_s700-729_cXXXXXXXX.csv
  eval_pd3qn_fe_r010_LS_Tight_fromSS_Tight_m1_s700-729_cXXXXXXXX.csv

outputs/run/irws_r_fe_r010/
  eval_irws_r_fe_r010_SS_Loose_fromSS_Loose_m1_s700-729_cXXXXXXXX.csv

outputs/run/mahdrl_r010_nopf/
  eval_mahdrl_r010_nopf_SS_Medium_fromSS_Medium_m1_s700-729_cXXXXXXXX.csv
```

`SS/MS/LS` 分别表示任务规模 S/M/L、资源规模 S；截止期直接使用 Tight/Medium/Loose。
不同源模型、模型种子及测试种子集合均有不同文件名。连续种子用范围表示，离散种子用 `+` 连接；
很长的离散列表用数量和短摘要表示，完整列表保留在对应明细中。
诊断、汇总、逐种子记录存放在同一算法目录的 `_details/CSV同名主干/`，不影响直接批量下载根目录下的 CSV。
仅当本次 CSV 已存在或本次明细目录非空时拒绝覆盖，其他场景的结果不阻止本场景评测。

本次路径调整不搬移已有模型和结果；已有匹配模型可通过 `--checkpoint-dir` 指定原模型目录。

## 验证

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest common.test_r010_paths common.test_pd3qn_fe_r010 common.test_irws_r_fe_r010 common.test_mahdrl_as_fast_r010_nopf -v
```

覆盖固定动作的物理一致性与新奖励、NoPF 候选完整性、观测摘要、忙碌动作拒绝、原学习器复用、
CPU/CUDA 更新、实际短程训练/保存/加载/冻结评测、九源与 27 目标路径、跨版本/错奖励模型拒绝及防覆盖。
短程测试采用临时缓存、校准或小网络，不替代正式 600 回合实验。本次未启动正式实验。

变式首次实现时，25 项 R010 专项与 23 项原 R090 回归共 48 项通过，无跳过。
本次路径调整后，新增 3 项跨算法路径检查与 25 项 R010 专项共 28 项全部通过，包含 CPU/CUDA 更新。
路径检查覆盖每算法 27 个 CSV 同父目录且文件名唯一、短训练目录、源模型映射、精确种子标识及配置隔离。
六份 Bash 指令统一为独立函数；评测支持第三参数指定目标，例如 `"M L"`。
发布前在临时目录模拟后台执行，27 条训练和 81 条评测展开命令全部通过真实 CLI dry-run 参数检查。
结果记录见 [验证报告](R010_variants_validation.json)。发布分支为 `codex/f11-real-transitions`，提交记录以 Git 历史为准。
