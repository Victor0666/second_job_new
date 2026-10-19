# R16 协议与校准绑定

默认划分不变：训练0–599，校准101–130，验证600，测试700–729。修改不改变奖励、网络、物理模型或缩放公式。

## 已补充的检查

- 公共Scenario检查实际评测种子属于所选协议的test清单，覆盖所有正式算法与启发式CLI；支持清单内子集调试。正式表格需使用统一完整名单，子集结果不能当作完整30实例结果。
- 校准器新增`--experiment-protocol`；单场景和九场景都传递实际协议，并支持自定义协议的回合预算。直接调用sample_episode同样核对训练种子。
- 校准清单及输出配置记录协议内容、文件SHA256、实际校准种子、源场景、代码指纹及缓存来源。续跑和仅汇总拒绝协议缺失或不匹配的清单，并重新检查全部场景的校准种子。
- 正式学习CLI及normalization_metadata在训练/加载模型时核对校准文件的协议和源训练场景种子。跨规模泛化仍检查源S场景；不根据测试场景重新校准。
- 预览参数、截止期缓存生成、显式legacy复现和无协议的底层物理测试保留原有用途；旧README正式评测命令已改用700–729。

## 使用方法

默认协议预览：

```powershell
.\run.cmd common/f18_calibrate_normalization.py --all-nine --seeds 101-130 --dry-run
```

正式校准选择新目录，避免与旧版清单混用：

```powershell
.\run.cmd common/f18_calibrate_normalization.py --all-nine --seeds 101-130 --out outputs/f18_calibration_r16
```

训练/评测通过`--normalization-config outputs/f18_calibration_r16/f18_shared_normalization_v1.json`使用新文件。需要提前准备完整且兼容的正式Deadline Cache。

自定义协议（文件内明确互不重叠的train/validation/test、episodes、selection以及独立version）：

```powershell
.\run.cmd common/f18_calibrate_normalization.py --all-nine --seeds 101-130 --experiment-protocol configs/paired_cloud_protocol_v1.json --out outputs/calibration_paired_r16
.\run.cmd common/f18_calibrate_normalization.py --summary-only --experiment-protocol configs/paired_cloud_protocol_v1.json --out outputs/calibration_paired_r16
```

训练、评测必须使用同一协议及该协议下生成的校准文件。自定义协议评测要显式指定其test名单，例如历史配对协议用`--seeds 2-31`。

## 旧文件与验证边界

未记录协议的旧校准文件现在不能直接用于正式运行，需重新生成；不会自动覆盖旧校准、模型和结果。只是增加检查不要求一律重训，但重新生成的校准文件若内容哈希或尺度发生变化，会触发既有checkpoint匹配检查，旧模型不能直接混用。

新增`common.test_r16`检查跨协议种子身份、各CLI对900的拒绝、九场景自定义协议、校准文件缺失/错误来源、泛化源场景、续跑/汇总隔离。真实临时DAX短程验证覆盖校准、IRWS训练、验证、保存、加载和评测；不是600回合训练或性能比较。
