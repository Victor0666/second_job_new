# F07 revision and acceptance record

Base commit before changes: `012619f6beab40f586b8491b1d8007a979f415ab` (main). Only file dependencies, required cache isolation/validation, tests and documentation are changed. Existing deadline caches, experiment outputs, model weights and the four unrelated untracked Word reports are preserved. No commit or push is performed in this task.

## Model and key changes

`common/read_xml_opt_Tsize.py::_randomize_payloads` now generates every task's outputs once. `DataFile` is immutable. Every direct child shares the complete parent output objects, deduplicated by workflow-local file name. A file's ID, size and producer stay identical in every reference. Distinct workflow objects scope identical local names; cache records scope hashes by episode seed and workflow index.

Before: every task independently sampled 1–3 input files, edges received `parent_output / outdegree`, and external input was a positive residual of independent input minus parent data. These rules are removed, with no fallback.

After: each task has 1–3 outputs; only entry tasks independently sample 1–3 external input files. Non-entry inputs are the union of all direct parents' outputs, without a count cap. Their `parent_in_bits[p]` is the parent's full output sum, `from_parents_bits` is its mapping sum, `in_file_size_sum_bits = from_parents_bits`, and `ext_in_bits = 0`. All default-model edges have positive logical data. The parser uses the original DAX nodes and edges and ignores XML payload sizes.

File sizes still use independent `uniform(512, 1024)` draws and `MB * 1024 * 1024 * 8.0`. Workloads retain `uniform(60, 90)` MI and the existing positive-value guard. `io_file_count_range` remains the existing interface for output counts and entry input counts only. Removing non-entry random input draws changes the payload RNG sequence, including subsequent exact workload/file values; their distributions and ranges are retained. The generator version records this change. Old/new measurements are a model-version comparison, not an isolation experiment with identical workload draws.

`randomize_payloads=False` retains topology-only tasks with no files and zero workloads/data. Dynamic arrivals, DAG topology, resources, alpha, rewards, energy, completion events, networks and `COMMUNICATION_MODEL = "same_vm_only_v1"` are unchanged. Same-VM placement can waive the network input time, never the underlying logical edge data. Output transfer remains governed by the existing execution model.

## Actual algorithm call chains

All chains use `common.read_xml_opt_Tsize.load_workflow_from_dax`:

| Algorithm | Environment chain |
|---|---|
| MAHDRL | `algorithms/mahdrl` -> `mahdrl_fcfs` -> `mahdrl_heft` |
| ERTF-H-FIFS | `algorithms/ertf_fifs` -> `mahdrl_fcfs` -> `mahdrl_heft` |
| MAHDRL-NHA | `algorithms/ablation_hostrand` -> `mahdrl_fcfs` -> `mahdrl_heft` |
| MAHDRL-NTA | `algorithms/ablation_mgrrand` -> `mahdrl_fcfs` -> `mahdrl_heft` |
| MAHDRL-NVA | `algorithms/ablation_vmrand` -> `mahdrl_fcfs` -> `mahdrl_heft` |
| MAHDRL-TL | `algorithms/ablation_nohost` -> `ablation_nohost` -> `mahdrl_fcfs` -> `mahdrl_heft` |
| PD3QN | `algorithms/pd3qn` -> `pd3qn` |
| IRWS | `algorithms/irws` -> `irws` |
| MARL | `algorithms/marl` -> `marl` |
| FCFS/FIFS reference | `common/precompute_deadlines` -> `base_fcfs` -> `base_heft` |

Ranks, shared task sub-deadlines, execution and VM prediction already consume aggregate task data and `parent_in_bits`; the generator fix reaches those consumers without changing their algorithms. The cache-enabled arrival methods additionally validate instance hashes. For SS seed=2, all 50 workflow file hashes, topology/workload/data fields, payload seeds, templates, arrivals, alpha, workflow deadlines and task deadlines agree across all nine algorithms under Tight/Medium/Loose. All nine chains also complete a real CyberShake_30 single-workflow/single-VM smoke episode and reject a deliberately corrupted cache instance hash.

## Paper replacement: Section V-A, Problem Instances

The repository contains `docs/paper.pdf`, but no editable manuscript source was found. The original PDF is not edited. The following is directly usable English LaTeX replacing the file-generation description (retain the surrounding arrival/resource settings):

```latex
We use the original DAG topologies of the Pegasus DAX workflows, but do not
use the file sizes specified in the DAX files. Each task independently
generates one to three output files, with each file size sampled uniformly
from $[512,1024]$ MB. Each entry task additionally receives one to three
external input files sampled from the same size distribution. For a non-entry
task, the input file set is the union of the complete output file sets of
all its immediate predecessors. Consequently, its number of input files is
determined by its predecessors and is not restricted to one to three.
Every immediate successor of a task references the same complete set of
that task's output files, preserving each file's identity, size, and producer;
the output data are not divided among successors. The logical data volume
on each DAG edge is computed from the intersection of the predecessor's
output file set and the successor's input file set, as in Eq.~(3).
```

## Paper clarification: Section III-B, Eqs. (3)–(5)

Keep the equations; add the following text around their symbol definitions:

```latex
A file denotes a unique logical object within a workflow instance. Its
identifier, size, and producing task are unchanged when it is referenced by
multiple successors. In our instance generator, for every edge $(p,c)$,
$\mathcal{F}^{\mathrm{out}}_p \subseteq \mathcal{F}^{\mathrm{in}}_c$;
thus the intersection in Eq.~(3) is the complete output file set of $p$.
For a non-entry task, the input file set is the union of the output file
sets of its immediate predecessors, and its external input volume is zero.
Only entry tasks receive external input files. A task's output volume in
Eq.~(4) counts its unique output files once, regardless of its outdegree.
Logical edge volumes remain positive even when predecessor and successor
are placed on the same VM; the existing same-VM rule affects network
transfer time only.
```

Align the notation in this paragraph with the manuscript's existing file-set symbols. In the paper, Eq. (5)'s factor eight converts bytes to bits. Implementation `size_bits` fields already include that conversion; do not multiply them by eight again. The implementation retains its historical binary MB conversion; clarify that convention in the experimental settings if keeping the label MB.

## Cache versions and commands

The sole data-model constant is `DATA_MODEL = "dynamic_full_parent_output_v1"`. New default caches live in:

`data/deadlines/dynamic_full_parent_output_v1/ddl_fcfs_task{S|M|L}_res{S|M|L}_seed0-1000.json`

Historical files remain in `data/deadlines/`. Scenario, precompute, all training/evaluation defaults and `verify_package.py` agree on the new directory. The separate `Scenario.legacy_cache_path` is for historical audits only. Existing-cache writes, including an explicit `--out`, fail before generation; exclusive file creation also prevents overwriting a file created during computation.

Required metadata are checked exactly: `data_model`, `communication_model`, `file_size_mb_range`, `output_file_count_range`, `entry_input_file_count_range`, `file_size_conversion`, `payload_generator_version`, and `length_mi_range`. Missing/different values are rejected with a regeneration command. Each generated seed record stores `wf_instance_hashes` over topology, workloads, file IDs/sizes/producers, parent mappings and data aggregates. Scheduling states, assignment, ranks, deadlines and execution results are excluded. Runtime arrival validation checks provided hashes before scheduling. Hashless hand-built cache fixtures remain supported when their complete model metadata match; generated production caches always contain hashes.

From the project root, generate all nine production caches:

```powershell
foreach ($taskSize in 'S', 'M', 'L') {
    foreach ($resourceSize in 'S', 'M', 'L') {
        .\run.cmd common/precompute_deadlines.py --task $taskSize --resource $resourceSize --seeds 0-1000
        if ($LASTEXITCODE -ne 0) { throw 'F07 deadline cache generation failed' }
    }
}
```

Training/evaluation automatically select the versioned cache for their scenario. Before running the training commands below, preserve existing checkpoint/log directories separately; their names are unchanged by F07. For example:

```powershell
.\run.cmd algorithms/mahdrl/train.py --task S --resource S --deadline Tight
.\run.cmd algorithms/mahdrl/evaluate.py --task S --resource S --deadline Tight --seeds 2-31
.\run.cmd algorithms/marl/train.py --task S --resource S --deadline Tight
```

No production 0–1000 cache regeneration or full training was performed in this task. Tests generate temporary seed=2 caches and remove only their own temporary directories. Partial seed generation automatically uses a separate filename, e.g. `--seeds 2`, and cannot satisfy a production training run requiring seeds 1–600. Existing cache output paths are deliberately not overwritten: choose a new `--out` for another audit run. Preserve old model outputs and use distinct output locations or archive them before retraining; this task does not rename checkpoint/log directories.

## Validation and measurements

Commands actually executed:

```powershell
.\.venv\Scripts\python.exe -m common.audit_F07 --out docs/F07_before.json
# Previous command ran BEFORE the generator change, at the base model version.
.\.venv\Scripts\python.exe -m common.audit_F07 --out docs/F07_after.json
.\.venv\Scripts\python.exe -m unittest common.test_file_dependencies -v
.\.venv\Scripts\python.exe -m unittest discover -s common -p 'test_*.py' -v
```

The final suite has 52 passing tests, including 15 independent F07 tests. `F07_tests.txt` records the full final run. All 15 real DAX templates pass topology, positive edges, complete shared parent output references, file identity/size/producer consistency, aggregate sums, entry-only external inputs and topology-only mode checks. `F07_templates.md` lists every template's node/edge counts, maximum input file count and before/after reference makespan. The 13-parent synthetic case explicitly retains all 26 files. The old PD3QN numerical finish-time pin was replaced with a duration invariant because payload instances change; all F01 precedence and tail completion checks remain.

Initial test attempts exposed (1) configuration fixtures that assumed production caches already existed, corrected to intercept only cache presence and MARL's seed inventory, and (2) the private generator's task-list assumption about graph insertion order, corrected to index generated tasks by sorted DAX node ID. Neither required a scheduling policy change. Final results have no failures.

`F07_before.json` and `F07_after.json` retain 65 per-workflow measurements each: SS seed=2's 50 workflows in a shared dynamic episode, plus each of the 15 templates in its own single-workflow episode with the same SS resources and seed=2. `F07_comparison.csv` joins every old/new pair. Its mean edge volume is the arithmetic mean of logical edge bits; maximum input is the largest task input bit sum; total logical communication is the sum of all DAG edge volumes (before same-VM network exemption), excluding external ingress/output egress. Reference makespan is measured from workflow arrival through actual completion using FCFS task selection plus first-idle VM, including contention in the SS_50 case.

Averages over SS_50's 50 workflows (each metric averaged per workflow):

| Metric | Before | After |
|---|---:|---:|
| Mean edge data (bits) | 9,477,846,717.285 | 12,601,737,231.854 |
| Maximum task input (bits) | 22,571,554,573.645 | 130,635,934,407.051 |
| Summed logical edge data (bits) | 344,944,955,381.747 | 480,305,912,586.914 |
| FCFS reference makespan (s) | 429.105706 | 413.876672 |

The reference makespan need not monotonically increase: new payload draws, same-VM placement exemptions and dynamic scheduling contention all affect it. No rescaling, truncation or outdegree division was introduced to preserve earlier magnitudes. These are reference measurements, not new trained-algorithm experimental results.

## F07 acceptance table

| F07 | Requirement | Modified function/file | Status | Evidence |
|---|---|---|---|---|
| F07-1 | Remove outdegree division | `_randomize_payloads` | Complete | 1400 MB to each of three children |
| F07-2 | Preserve random output distribution | loader constants / `make_files` | Complete | 1–3 outputs, each 512–1024 MB, original conversion |
| F07-3 | Complete shared parent outputs | immutable `DataFile`, generator | Complete | shared identity, equal IDs/sizes/producers |
| F07-4 | Positive logical edge volume | `parent_in_bits` generation | Complete | all edges of all 15 templates positive |
| F07-5 | Union of every direct parent's outputs | generator | Complete | join and 13-parent cases |
| F07-6 | Zero non-entry external input | generator | Complete | aggregate/external-input assertions |
| F07-7 | Random external input at entries | generator | Complete | entry count, size, producer assertions |
| F07-8 | No non-entry input cap | generator | Complete | 13 parents retain 26 files |
| F07-9 | Preserve all DAG dependencies | loader and template tests | Complete | original node/edge sets unchanged |
| F07-10 | Shared model across all algorithms | shared loader, 5 cache environment arrival paths | Complete | 9 real smoke episodes, paired 50-workflow hashes |
| F07-11 | Paper replacement text | `docs/revision_F07.md` | Complete | V-A replacement and III-B clarifications; PDF unchanged |
| F07-12 | Version/cache/tests | reader, workflow, precompute, scenarios, tests, verification routing | Complete | 52 tests pass; old caches rejected, existing writes protected |

No old outdegree division or external-input residual fallback remains in executable payload generation; source searches found no independent implementation in another algorithm. Downstream code does not cap file counts or allocate fixed file arrays.

## Independent issues and readiness

The existing paper states 60k–90k MI and processing tiers in k MI/s, while current code uses 60–90 MI and tiers 1,2,4,6,8 MI/s. These unit/scaling discrepancies are outside F07 and are not changed. Small SIPHT is the configured 29-task `Sipht_29.xml`; the paper's 25-task entry needs its separately agreed paper-only correction. The cited TSEDA/QPHH texts are not available as repository sources and were not independently verified here; the concrete F07 rules and the available MAHDRL PDF govern this implementation.

F07 code acceptance is satisfied for the tested template/model/environment paths. Formal experiments still require all nine new production caches, a fresh training/evaluation cycle, separate handling of legacy model outputs, and agreement on the independent paper unit discrepancies. Full 27-combination training/evaluation, statistical significance and research conclusions were not tested or claimed.
