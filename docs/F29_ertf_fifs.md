# F29：修复 ERTF-H-FIFS 在 padding 主机上的 VM 选择

## 修改

`algorithms/ertf_fifs/evaluate.py` 的 `_pick_fifs_vm` 直接使用 `env.host_to_vm_indices[host_id]`，按环境 `vm_assign(slot)` 的真实本地 slot 顺序映射全局 VM。不再根据 mask 长度猜测动作类型，删除旧推断函数及最小合法编号的静默 fallback。

在 mask 合法的真实 slot 中，以 `(vm_available_at[global_index], local_slot)` 为排序键，返回本地 slot。padding 位不得标记为合法；空候选、主机越界、mask 形状错误、无效索引和非有限空闲时刻会明确报错。

主机选择、任务 FCFS 顺序、两阶段主机→VM 结构、截止期生成、物理模型及能耗计量均未修改。原有 9VM 主机正常分支和 8VM 主机现在使用同一规则；空闲时刻相同时选择较小本地 slot。

## 验证

执行：

```powershell
.venv/Scripts/python.exe -B -m unittest common.test_f29 -v
```

测试包含 8VM/9维 padding 的历史反例、9VM、与全局编号排序不同的真实映射顺序、忙碌 VM 排除、并列规则，以及异常数据显式报错。真实环境测试分别覆盖 S/M/L 三种资源规模，每回合两个工作流；逐次检查两阶段选择与独立全局 FIFS oracle 一致，并核对任务完成时刻、工作流 makespan 和全局能耗。

## 重跑与校准

需要用修复后的选择器重跑 ERTF-H-FIFS，旧指标不代表修复后的策略，也不能预判修复后指标一定改善。本次未运行正式 30 种子评测。

公共截止期预计算本来就使用正确的全局 FIFS，没有此次 padding 问题；本修复不要求因此重新生成截止期缓存。

`common/f18_calibrate_normalization.py` 复用 ERTF-H-FIFS 的选择器，后续校准会自然使用修复版本。如果已有归一化尺度是用旧选择器生成的，应重新校准并核对尺度与配置哈希；如尺度变化，各学习算法需共同使用新尺度重训，不能把旧模型当作使用新尺度训练的模型。本次不覆盖现有尺度、模型或实验输出。
