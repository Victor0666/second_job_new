# MARL_Q4.2_090

Independent MARL-Q4.2 reward variant: delay=0.9, energy=0.1.
Global and Local preserve their original energy scope, idle-VM-only scheduling and SR092 selection.

Train: `python -u launch.py algorithms/marl_q_v4_2_090/train.py`.
Evaluate: `python -u launch.py algorithms/marl_q_v4_2_090/evaluate.py`.

See [implementation and commands](../../docs/MARL_Q_v4_2_weights_implementation.md).
