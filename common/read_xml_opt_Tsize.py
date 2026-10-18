# -*- coding: utf-8 -*-
"""
read_xml_opt_Tsize.py
改动要点：
1) 任务工作量统一改为 MI 单位：Task.workload_mi（MI）。
2) 为每个子任务建立“来自各父任务的输入量”映射：
   - Task.parent_in_bits: {parent_task_id -> bits}
   - Task.from_parents_bits: float（合计）
   - Task.ext_in_bits: float（仅入口任务具有外部输入）
3) 每个子任务引用全部直接父输出文件；最终统一到 bits。
"""

import hashlib
import json
import random
import math
import networkx as nx
from xml.etree import ElementTree as ET
from dataclasses import dataclass, field


DAX_TEMPLATE_ID_SCHEME = "basename_casefold_v1"

DATA_MODEL = "dynamic_full_parent_output_v1"
PAYLOAD_GENERATOR_VERSION = "full_parent_outputs_rng_v1"
FILE_SIZE_MB_RANGE = (512, 1024)
FILE_COUNT_RANGE = (1, 3)
LENGTH_MI_RANGE = (60, 90)
FILE_SIZE_CONVERSION = "MB * 1024 * 1024 * 8.0"


def data_model_metadata():
    return dict(data_model=DATA_MODEL, file_size_mb_range=list(FILE_SIZE_MB_RANGE),
                output_file_count_range=list(FILE_COUNT_RANGE),
                entry_input_file_count_range=list(FILE_COUNT_RANGE),
                file_size_conversion=FILE_SIZE_CONVERSION,
                payload_generator_version=PAYLOAD_GENERATOR_VERSION,
                length_mi_range=list(LENGTH_MI_RANGE))


def workflow_instance_hash(graph, tasks):
    """Hash immutable instance data, scoped to this workflow; omit scheduling state."""
    def files(values):
        return sorted((f.name, f.size_bits, f.source_task_id) for f in values)
    payload = dict(nodes=sorted(graph.nodes), edges=sorted(graph.edges), tasks=[
        dict(id=t.task_id, workload=t.workload_mi, inputs=files(t.in_files),
             outputs=files(t.out_files), parents=sorted(t.parent_in_bits.items()),
             input_bits=t.in_file_size_sum_bits, output_bits=t.out_file_size_sum_bits,
             parent_bits=t.from_parents_bits, external_bits=t.ext_in_bits)
        for t in sorted(tasks, key=lambda t: t.task_id)])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def validate_cached_instance(cache, seed, workflow_id, graph, tasks):
    """Check generated payload against its reference record when hashes are supplied."""
    record = (cache or {}).get(int(seed), {})
    hashes = record.get("wf_instance_hashes")
    if hashes is not None:
        if workflow_id >= len(hashes) or hashes[workflow_id] != workflow_instance_hash(graph, tasks):
            raise ValueError(f"Deadline cache instance hash mismatch: seed={seed} workflow={workflow_id}")


def dax_template_id(path):
    """A portable, case-insensitive identifier for a configured DAX template."""
    return str(path).replace("\\", "/").rsplit("/", 1)[-1].casefold()


def validate_dax_template_name(cached_name, runtime_name, *, strict=True):
    if dax_template_id(cached_name) == dax_template_id(runtime_name):
        return
    message = f"[deadline-cache] DAX template mismatch: runtime={runtime_name} cache={cached_name}"
    if strict:
        raise ValueError(message)
    print("WARNING:", message)


# -----------------------------
# 基本数据结构
# -----------------------------

@dataclass(frozen=True)
class DataFile:
    name: str
    size_bits: float
    source_task_id: int | None  # 由哪个任务生成；若为外部输入则为 None


@dataclass
class Task:
    task_id: int
    parents: list = field(default_factory=list)     # 父任务编号
    children: list = field(default_factory=list)    # 子任务编号

    # 任务负载（统一用 MI）
    workload_mi: float = 0.0

    # 输入/输出文件
    in_files: list = field(default_factory=list)
    out_files: list = field(default_factory=list)

    # 输入/输出总大小（bits）
    in_file_size_sum_bits: float = 0.0
    out_file_size_sum_bits: float = 0.0

    # —— 父→子 I/O 映射（由本模块在构图后填充）——
    parent_in_bits: dict = field(default_factory=dict)  # {parent_id: bits}
    from_parents_bits: float = 0.0                      # sum(parent_in_bits.values())
    ext_in_bits: float = 0.0                            # entry input only; zero for non-entry tasks

    # 运行期（可选）
    state: str = "unReady"
    assigned_server_id: int | None = None
    assigned_vm_pc: float | None = None
    ready_time: float | None = None
    arrival_time: float | None = None
    start_processing_time: float | None = None
    end_processing_time: float | None = None
    finish_time: float | None = None

    # 排序用的两个“秩”
    upward_rank: float | None = None
    downward_rank: float | None = None


def _read_dax_to_graph(xml_path):
    """将 DAX 解析为 nx.DiGraph（节点为 0..N-1 的整数）。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'a_dax': 'http://pegasus.isi.edu/schema/DAX'}

    id_map = {}
    G = nx.DiGraph()

    # 注册所有 job
    for idx, job in enumerate(root.findall('a_dax:job', ns)):
        job_id = job.get('id')
        id_map[job_id] = idx
        G.add_node(idx)

    # 建立父子边
    for child in root.findall('a_dax:child', ns):
        child_ref = child.get('ref')
        if child_ref not in id_map:
            continue
        c = id_map[child_ref]
        for parent in child.findall('a_dax:parent', ns):
            p_ref = parent.get('ref')
            if p_ref in id_map:
                G.add_edge(id_map[p_ref], c)

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("解析到的图不是 DAG。")
    return G, id_map


def _randomize_payloads(
    G,
    seed=None,
    length_mi_range=LENGTH_MI_RANGE,    # 任务计算量（MI）
    io_file_count_range=FILE_COUNT_RANGE,      # 输入/输出文件个数范围
    file_size_mb_range=FILE_SIZE_MB_RANGE,  # 每个文件大小范围（MB）
):
    """Generate unique outputs once; each direct child receives every parent output."""
    rng = random.Random(seed)
    tasks = [Task(tid, parents=list(G.predecessors(tid)),
                  children=list(G.successors(tid))) for tid in sorted(G.nodes)]

    def make_files(prefix, tid, producer):
        return [DataFile(f"{prefix}_{tid}_{i}",
                         rng.uniform(*file_size_mb_range) * 1024 * 1024 * 8.0, producer)
                for i in range(rng.randint(*io_file_count_range))]

    for t in tasks:
        t.workload_mi = max(abs(rng.uniform(*length_mi_range)), 1e-6)
        t.out_files = make_files("output", t.task_id, t.task_id)
        t.out_file_size_sum_bits = sum(f.size_bits for f in t.out_files)
        if not t.parents:
            t.in_files = make_files("input", t.task_id, None)

    for t in tasks:
        if t.parents:
            # File names are unique within the workflow; preserve shared immutable identities.
            t.in_files = list({f.name: f for p in t.parents for f in tasks[p].out_files}.values())
            t.parent_in_bits = {p: tasks[p].out_file_size_sum_bits for p in t.parents}
            t.from_parents_bits = sum(t.parent_in_bits.values())
            t.ext_in_bits = 0.0
            t.in_file_size_sum_bits = t.from_parents_bits
        else:
            t.in_file_size_sum_bits = sum(f.size_bits for f in t.in_files)
            t.ext_in_bits = t.in_file_size_sum_bits
    return tasks


def load_workflow_from_dax(
    xml_path,
    seed=None,
    randomize_payloads=True,
    # length_mi_range=(2000, 3000),
    length_mi_range=LENGTH_MI_RANGE,
    io_file_count_range=FILE_COUNT_RANGE,
    file_size_mb_range=FILE_SIZE_MB_RANGE,
):
    """高层接口：解析 DAG + 可选随机负载 + 建立父→子 I/O 映射。"""
    G, id_map = _read_dax_to_graph(xml_path)
    if randomize_payloads:
        tasks = _randomize_payloads(
            G, seed=seed,
            length_mi_range=length_mi_range,
            io_file_count_range=io_file_count_range,
            file_size_mb_range=file_size_mb_range,
        )
    else:
        tasks = [Task(tid) for tid in G.nodes]
        for u, v in G.edges:
            tasks[u].children.append(v)
            tasks[v].parents.append(u)
        # 没有随机 I/O 时，parent_in_bits/ext_in_bits 保持为 0
    return G, tasks


def poisson_arrival_times(lmbda, horizon=None, seed=None, n_max=None, start_time=0.0):
    """
    生成泊松过程到达时刻。
    两种用法二选一：
      - 指定 n_max：返回前 n_max 个到达时刻（推荐）
      - 指定 horizon：返回 [start_time, start_time + horizon) 内所有到达时刻（可能很多）
    """
    if (horizon is None) and (n_max is None):
        raise ValueError("poisson_arrival_times: must common either horizon or n_max")

    import numpy as np
    rng = np.random.default_rng(seed)

    if n_max is not None:
        gaps = rng.exponential(1.0 / max(lmbda, 1e-12), size=int(n_max))
        return (start_time + np.cumsum(gaps)).tolist()

    # 旧逻辑：按 horizon 生成（可能非常多，谨慎使用）
    t = start_time
    arr = []
    # 保护：如果期望数量过大，直接抛错避免 OOM
    expected = lmbda * float(horizon)
    if expected > 2_000_000:
        raise MemoryError(f"poisson_arrival_times: expected arrivals ~{expected:.0f} is too large. "
                          "Use n_max instead of horizon.")
    while t < start_time + horizon:
        t += rng.exponential(1.0 / max(lmbda, 1e-12))
        if t < start_time + horizon:
            arr.append(t)
    return arr
