# MAHDRL-AS-Fast-R010-NoPF

Independent R010 variant: delay weight 0.1 and energy weight 0.9.

Training: `python -u launch.py algorithms/mahdrl_oe_ge_score_fast_r010_nopf/train.py`.
Evaluation: `python -u launch.py algorithms/mahdrl_oe_ge_score_fast_r010_nopf/evaluate.py`.

See [implementation and run commands](../../docs/R010_variants_implementation.md).
Both host and VM PF/APF are disabled. TA/HA/VA learning and GE remain enabled.
