# IRWS 历史批量版

模型版本为 `irws_r_v2_batch`，默认1秒调度周期、float64、单区间更新。此副本保留修改前的计算过程，入口仅重定向到本目录，仍使用原模型及输出目录。

训练：`python launch.py algorithms/irws_r_v2/train.py ...`；评测：`python launch.py algorithms/irws_r_v2/evaluate.py ...`。原有队列和场景参数保持一致。

副本创建时的文件哈希见 `snapshot_sha256.json`；README在本次说明时更新，不作为计算源码保护项。新加速版位于 `algorithms/irws_r`。
