# MAHDRL-OE-PF：OE＋PF，不加入 GE

算法 ID：`mahdrl_oe_pf`。独立训练、评估、模型、日志和结果目录；原 MAHDRL、OE/GE/PF/TS/UB 及此前的 OE＋GE、OE＋GE＋PF 保留。

## 行为

- **OE**：保留任务子截止期、主机身份/功率及候选成本观测；HA/VA 按每次任务分配的精确新增排程能耗即时获得奖励。最终物理计能、TA phase 奖励和异常补结算沿用 OE。
- **PF**：HA 按每台主机最快可用 VM 的预计时长筛主机，VA 按已选主机内最快 VM 的预计时长筛 VM；每层保留不超过该层最快值 1.25 倍的合法动作。训练、验证、独立评估及 TD 后继动作均使用筛选掩码。
- **没有 GE**：HA/VA 在 epsilon 分支内，对筛选后的合法候选均匀随机探索，其余按 Q 值贪心选择；没有“50% 选最快”的引导分支。TA 也使用原动作选择与更新。

OE 的候选摘要按筛选后仍可选的 VM 计算，最低能耗摘要不会指向已排除的慢 VM；被筛掉的主机追加摘要和 VM 追加候选特征置零。阈值、候选处理及物理环境与已有三合一版一致，两者的实际训练差别是是否加入 GE。

网络输入维度与 OE 相同，隐藏层、lr、gamma、batch、buffer、epsilon 退火、目标网络更新、延迟项及能耗缩放保持原设置。

## 实现与隔离

- 入口：`algorithms/mahdrl_oe_pf/train.py`、`algorithms/mahdrl_oe_pf/evaluate.py`，从 OE 建立，不传递性能引导参数。
- agent：`algorithms/mahdrl_oe_pf/agent.py`，直接继承原始 epsilon-greedy 和 D3QN 更新；继承链中没有 `GuidedExplorationAgent`。
- 环境：`environments/mahdrl_oe_pf.py`，只复用已验证的 OE＋PF 物理、奖励、筛选后观测逻辑。GE 位于 agent/trainer，复用环境不会加入 GE。
- phase 奖励结算直接复用 `algorithms.mahdrl_oe.reward`，没有重复发放能耗。

三层 checkpoint 均记录 `mahdrl_oe_pf_v1`、`components=["OE","PF"]`、`guided_probability=0`、`max_duration_ratio=1.25` 和角色。新入口拒绝原版、单变式、其他组合和错角色模型；此前组合也会拒绝本版权重。新模型需独立从头训练。

模型与日志目录以 `mahdrl_oe_pf_task...` 开头；主评测 CSV 以 `eval_mahdrl_oe_pf_task...` 开头，结果及统一测量分别使用 `mahdrl_oe_pf` 目录。`algorithm_variant.json` 保存组合身份和参数。

## 训练与评估

从项目根目录、已激活的 Python 环境执行，示例为 SM/Tight，默认 F22/F19，模型种子 1：

```bash
# 50代训练
python -u launch.py algorithms/mahdrl_oe_pf/train.py --task S --resource M --deadline Tight --episodes 50 --model-seed 1

# 600代训练
python -u launch.py algorithms/mahdrl_oe_pf/train.py --task S --resource M --deadline Tight --episodes 600 --model-seed 1

# 两种训练预算共用评测命令，加载当前本版best
python -u launch.py algorithms/mahdrl_oe_pf/evaluate.py --task S --resource M --deadline Tight --train-task S --seeds 700-729 --model-seed 1

# 泛化至MM/Tight，仍用SM/Tight训练模型
python -u launch.py algorithms/mahdrl_oe_pf/evaluate.py --task M --resource M --deadline Tight --train-task S --seeds 700-729 --model-seed 1
```

Windows PowerShell 将 `python -u launch.py` 替换为 `.\run.cmd`。任一命令追加 `--dry-run` 只预览配置。沿用现有公共缓存与归一化，无需为该组合重建缓存。

50 和 600 代使用同一模型、训练日志及默认评测目录。入口不支持续训、拒绝覆盖非空训练目录；若先跑 50 代，应先评测并归档该组合自己的模型、日志和结果，再从头训练 600 代。方案选择使用训练/验证材料；现行 700–729 已被查看，后续最终核验应统一预留未参与方案选择的实例。

## 验证

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest common.test_mahdrl_oe_pf common.test_mahdrl_combinations common.test_scenarios -q
```

测试包括：动作选择与原算法使用完全相同的随机抽样过程；无 GE 类/引导参数；相同动作下与原 OE 奖励、最终能耗及三合一环境的掩码/摘要一致；正常与异常能耗结算守恒；单变式和其他组合的双向模型拒绝；临时目录中真实小网络训练、三层梯度更新、保存加载、评估及原输出保护。

没有执行正式 50/600 代训练，不据测试宣称节能效果。

2026-10-23 验证：OE＋PF、已有组合及场景专项共 18 项通过；最终全量回归共 249 项通过（129.688 秒）。50/600 代训练与评估三条真实 CLI dry-run、语法和空白检查通过。新增前现有算法、独立变式、两个组合及共享学习代码共 136 个受保护 Python 文件 SHA256 一致；仅在公共配置和场景测试中增补 `mahdrl_oe_pf` 注册。

完整回归仍在 discovery 后为测试进程恢复 F11/F19/测量开关，处理既有未提交冻结 TA 工具的导入副作用，没有修改该工具。通过记录：`tmp/mahdrl_oe_pf_regression.txt`。上述专项命令可直接运行。
