# F22：统一实例划分与复现性验收

本次在提交 `13edf45` 基础上修改；没有运行正式训练或 30 种子正式测试。算法网络、动作、学习更新、奖励、截止期公式、通信和功耗规则保持原实现。环境只修正 `reset(seed=...)` 的种子传递。

## 验收结果

| 检查项目 | 预期结果 | 实际结果 |
|---|---|---|
| Train | 0–599 | 公共协议逐回合索引，首 0、末 599，无重复 |
| Validation | 600，只有 1 个 | 全部训练入口默认 `[600]`；协议拒绝多个验证种子 |
| Test | 700–729 | 学习算法、四消融和 ERTF-FIFS 均默认 30 个种子 |
| Train Episodes | 600 | 保持 600；显式短程测试预算仍可用 |
| MAHDRL | 统一划分 | 默认启用 F22，保留验证频率和最小验证能耗选模 |
| PD3QN | 统一划分 | 默认启用 F22，按公共序列推进 |
| MARL | 统一划分 | 默认启用 F22，不循环复用缓存种子 |
| IRWS | 统一划分 | `irws_paper` 及历史 `irws` 两入口均适配 |
| MAHDRL Ablations | 统一划分 | hostrand、mgrrand、vmrand、nohost 全部适配 |
| Deadline Cache | 正确定位、严格校验 | 代码通过测试；九组新版正式缓存均缺失，不能开始正式实验 |
| Instance Hash | 同场景、同种子跨算法一致 | 对 0/Tight、600/Medium、700/Loose 的实际 DAX 实例通过；原有 seed=2、50 工作流、三档截止期检查也通过 |
| Model Seed | 与 Instance Seed 分离 | 公共 CLI 独立参数；改变模型 RNG 不影响物理实例 |
| Checkpoint | 保存划分和复现信息 | normalization 元信息记录协议、模型 seed、源训练场景、Git commit、缓存信息、生成版本；新旧协议加载校验 |
| Smoke Tests | 通过 | T01–T14 通过；完整回归 146 项通过 |

完整测试记录：`docs/F22_tests.txt`（146 tests，119.319 s）。最终补充验证：`docs/F22_focused_tests.txt`（15 项通过）、`docs/F22_legacy_checkpoint_tests.txt`（2 项通过）。缓存只读审计：`docs/F22_cache_audit.json`。

## 实现与边界

唯一正式配置为 `configs/f22_split_v1.json`。所有默认入口经 `common/scenarios.py` 读取它；`common/experiment_protocol.py` 校验三个集合两两不交、单验证种子、正式版本的固定范围和预算。显式自定义协议必须满足不重叠、单验证种子等校验，不能将其他划分标记成 `f22_split_v1`。

正式评测保留 `--seeds` 显式接口；覆盖值不能与训练或验证集合重叠。明确的历史评测可使用 `--legacy-split --seeds 2-31`，输出标识为 Legacy，不标为 F22。`configs/paired_cloud_protocol_v1.json` 原样保留，供此前 IRWS 32–631/632/2–31 协议显式复现；它不再是默认配置，元信息仍保留其原版本。短程自动测试中的 2、32、632 是显式夹具，不是正式默认值。

`instance_seed` 通过环境原有 `random_seed`/MARL `episode_seed` 接口传入。DAX 抽样使用环境独立 RandomState，到达使用局部 default_rng，payload 使用局部 random.Random；Alpha 保持 F06 独立派生规则。没有改变生成分布。修复 MAHDRL/base、PD3QN、IRWS 的 `reset(seed=...)` 忽略参数问题，尤其确保 paper IRWS 连续训练复用同一环境时能够切换实例。MARL 原有活动 episode seed 接口保持不变。

`model_seed` 独立控制模型初始化和全局算法 RNG。CLI 默认初始化种子已统一为 1，覆盖 MAHDRL、PD3QN、MARL、新旧 IRWS 和全部消融；仍可通过 `--model-seed` 显式指定其他值。训练回合切换只更换实例 seed，不重建网络、优化器、经验回放或探索状态。消融原有独立随机选择器仍保持其既有规则，不消耗环境实例 RNG。

`common/f22_instances.py` 对已全部到达的实例记录规范模板、DAG/MI/输入输出文件内容哈希、到达、绝对截止期、全部主机/VM 配置；同时生成场景/种子完整标识。哈希不含排序、调度状态、Rank 或强化学习状态。每轮 F11 诊断/IRWS training.jsonl 记录实例标识；提前截断时标记未完整到达，不冒充完整实例。设置 `F11_DIAGNOSTICS=0` 会关闭现有 F11 逐回合诊断及其哈希记录，独立 `data_split.json` 和 checkpoint 协议元信息仍保存。

MAHDRL/四消融在第 600 回合完成、验证和保存后停止，不再额外 reset 并选择第 601 回合动作。其他更新和选模顺序保持原逻辑。已有动态验证函数以小型真实 DAX 运行，使用监测策略检查 deterministic/count_step、参数与回放不变；真实 paper IRWS 参数/优化器 checkpoint 及无学习评测另由原回归测试覆盖。

新 checkpoint/log 路径追加协议哈希，评测结果根目录也追加协议哈希，避免覆盖旧协议产物；原有输出保护继续有效。旧模型缺少 F22 协议标识，正式模式会拒绝加载；不能将旧模型改标签冒充新训练。历史模式需显式选择其原协议/归一化配置/模型目录。

## 硬编码旧种子的处理

- `common/scenarios.py`：评测默认 `2-31` 改为 `700-729`，训练起始 1/验证 1 的默认回退只在显式 Legacy 下保留。
- 七个已适配训练入口：继续经公共 first_training_seed/validation_seeds 读取；本次增加独立 model_seed 和公共逐回合索引。
- `algorithms/irws/train.py`：硬编码训练/验证 1 改为公共协议。
- `algorithms/irws/evaluate.py`：汇总行硬编码 `2-31` 改为实际评测参数。
- 评测环境构造器和维度探测：默认实例改为实际测试序列首个种子，模型 RNG 不再借用每个测试实例 seed。
- `common/f18_calibrate_normalization.py`：校准种子成员校验改为训练集合 0–599；原默认校准 101–130 仍属于训练集，百分位/尺度/采样方法不变。
- 不再存在生效的正式默认训练 1–600 或测试 2–31。历史文档、旧协议文件、Legacy 回退、明确的自动测试夹具保留原值。

## 缓存结论与补齐

当前 `data/deadlines/dynamic_full_parent_output_v1/` 下 SS、SM、SL、MS、MM、ML、LS、LM、LL 九组标准缓存均缺失；每组缺少正式所需的 631 个 seed，即 `0-600,700-729`。项目旧目录里的历史缓存不满足现有 F07 模型，不能静默替代。没有生成或覆盖正式缓存。

`common/f22_cache.py` 检查模型/通信版本、模板标识版本、任务/资源规模、DAX 混合权重、到达参数、工作流数、PC/BW、参考 FCFS 策略、唯一 seed 和记录完整性。正式 CLI 启动前审计全部三组，并纳入显式额外测试 seed；运行环境仍逐工作流校验模板和 payload hash。不允许静默选其他 seed。参考 makespan 缓存可跨 Tight/Medium/Loose 共用，最终绝对 deadline 仍按各自原有 Alpha 概率重新计算。

```powershell
# 只读审计；缺失/不兼容返回非零退出码
.\run.cmd common/f22_cache.py --all

# 当前文件缺失时：只生成该资源场景缺少的 631 个正式种子。
# 三种资源缓存覆盖九个 S 训练场景；不逐 Deadline 重算参考 makespan。
foreach ($resource in @('S','M','L')) {
    $out = "data/deadlines/dynamic_full_parent_output_v1/ddl_fcfs_taskS_res${resource}_seed0-1000.json"
    .\run.cmd common/f22_cache.py --task S --resource $resource --supplement-out $out
}
```

若标准路径已经有兼容的部分缓存，`--supplement-out` 必须指定另一个新文件（例如 `..._f22_supplemented.json`）；工具合并原记录，仅计算缺失 seed，拒绝覆盖已有文件。审阅新文件后再决定如何切换标准缓存路径。版本/配置不兼容会报错，不能合并不同模型。若要准备 M/L 任务泛化，只需将上述 `--task S` 及路径中的 taskS 对应改为 M 或 L，并逐资源执行。没有必要为了 F22 扩展到 0–1000 全部 seed。

缓存就绪后，保持原 F18 校准设计，运行其既有命令。当前校准配置 `scenes={}`，尚未就绪：

```powershell
.\run.cmd common/f18_calibrate_normalization.py --all-nine --seeds 101-130 --resume
```

## 正式命令模板（本次未执行）

在仓库根目录使用 PowerShell；建议先给单条命令添加 `--dry-run` 检查。下面使用公共默认 F22 协议，不必逐入口再写种子。

```powershell
# MAHDRL、PD3QN、MARL、正式 paper IRWS：各九种 S 任务训练配置
foreach ($algorithm in @('mahdrl','pd3qn','marl','irws_paper')) {
    $modelSeed = 1
    foreach ($resource in @('S','M','L')) {
        foreach ($deadline in @('Loose','Medium','Tight')) {
            .\run.cmd "algorithms/$algorithm/train.py" --task S --resource $resource --deadline $deadline --episodes 600 --model-seed $modelSeed
        }
    }
}

# 四个既有消融：只提供模板，不在本次执行
foreach ($algorithm in @('ablation_hostrand','ablation_mgrrand','ablation_vmrand','ablation_nohost')) {
    foreach ($resource in @('S','M','L')) {
        foreach ($deadline in @('Loose','Medium','Tight')) {
            .\run.cmd "algorithms/$algorithm/train.py" --task S --resource $resource --deadline $deadline --episodes 600 --model-seed 1
        }
    }
}

# 历史 IRWS 实现入口也使用 F22；如确需比较该旧实现，可显式使用此模板
.\run.cmd algorithms/irws/train.py --task S --resource S --deadline Tight --episodes 600 --model-seed 1
```

验证无须额外 CLI：上述所有训练入口内部均使用 `[600]`，保持各自原验证频率、确定性评测模式及原 checkpoint 选择规则。没有新增多 seed 验证，也没有新增消融。

```powershell
# 正式测试：训练模型完成后再执行；九种源模型场景使用相同 30 个 seed
foreach ($algorithm in @('mahdrl','pd3qn','marl','irws_paper','ablation_hostrand','ablation_mgrrand','ablation_vmrand','ablation_nohost')) {
    $modelSeed = 1
    foreach ($resource in @('S','M','L')) {
        foreach ($deadline in @('Loose','Medium','Tight')) {
            .\run.cmd "algorithms/$algorithm/evaluate.py" --task S --resource $resource --deadline $deadline --train-task S --seeds 700-729 --model-seed $modelSeed
        }
    }
}

# 示例：MS 泛化，继续使用原 S 任务、相同资源和截止期的源模型
.\run.cmd algorithms/mahdrl/evaluate.py --task M --resource S --deadline Tight --train-task S --seeds 700-729
# 示例：启发式基线，默认也使用 700–729
.\run.cmd algorithms/ertf_fifs/evaluate.py --task S --resource S --deadline Tight --seeds 700-729

# 历史模型示例：必须显式 Legacy，且使用该模型实际归一化方式
.\run.cmd algorithms/mahdrl/evaluate.py --legacy-split --seeds 2-31 --normalization-mode legacy --checkpoint-dir '你的历史模型目录'
# IRWS 上一版显式无重叠协议复现：保留原协议和原测试种子
.\run.cmd algorithms/irws_paper/evaluate.py --experiment-protocol configs/paired_cloud_protocol_v1.json --seeds 2-31 --normalization-mode f18_shared_v1 --checkpoint-dir '上一版IRWS模型目录'
```

旧 paper IRWS checkpoint 的协议 hash 仍匹配旧协议，但其早期连续环境 reset 未正确更换 seed 的训练结果不能视为通过 F22 验收；相关结果应按实际生成版本处理。所有 F22 前的模型均不符合新正式划分，需重新训练。

## 修改文件

公共配置与元信息：`configs/f22_split_v1.json`、`common/experiment_protocol.py`、`common/scenarios.py`、`common/f22_cache.py`、`common/f22_instances.py`、`common/f18_state_normalization.py`、`common/f18_calibrate_normalization.py`。

学习入口：MAHDRL、PD3QN、MARL、历史 IRWS、四消融的 `train.py`/`evaluate.py`；paper IRWS 的 `runner.py`；ERTF-FIFS 的 `evaluate.py`。

种子透传：`environments/base_fcfs.py`、`base_heft.py`、`mahdrl_fcfs.py`、`mahdrl_heft.py`、`pd3qn.py`、`irws.py`，每处仅增加 reset seed 写回。

测试和文档：`common/test_f22.py`、`common/test_scenarios.py`、`common/test_irws_paper.py`、`common/test_f18.py`、README 及本目录的 F22 报告/测试记录/审计。四个用户 Word 文件未修改。

测试范围：T01–T14、旧 132 项回归及末次聚焦检查；没有运行 600 Episode 训练、完整 30 seed 测试、正式缓存生成、正式校准或参数搜索。F22 默认数据划分不存在训练/验证/测试重叠；显式 Legacy 不提供该保证。正式重训须先完成上述缓存和校准准备。

F19 后续说明：学习入口默认编码已升级为 `f19_shared_v1`，F22 实例划分不变。需要加载旧 F18 模型时，应显式传入 `--normalization-mode f18_shared_v1`，不能仅切换种子协议。
