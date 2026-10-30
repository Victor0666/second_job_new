# IRWS-R-FE_IDLE_010

Derived directly from `irws_r_fe_idle`: delay weight 0.1, energy weight 0.9.
Idle-only periodic scheduling, V3 learning, and the 92% workflow-success checkpoint gate are retained.

Train: `python -u launch.py algorithms/irws_r_fe_idle_010/train.py`.
Evaluate: `python -u launch.py algorithms/irws_r_fe_idle_010/evaluate.py`.

See [implementation and commands](../../docs/IRWS_R_FE_IDLE_weights_implementation.md).
