"""One train/evaluation configuration; historical observation modes retain old models."""
from common.f18_state_normalization import PD3QN_UNITS
from algorithms.pd3qn_dag.identity import RUN_NAME


def run_name(scenario):
    return RUN_NAME if scenario.normalization_mode == PD3QN_UNITS else 'pd3qn'


def agent_kwargs(scenario):
    paper = scenario.normalization_mode == PD3QN_UNITS
    # Paper p.14585: learning rate, discount, batch and replay; p.14587: copy step 60.
    # Width follows the requested MAHDRL trunk; the paper does not specify it.
    # Optimizer, PER exponent and epsilon endpoints remain implementation choices.
    return dict(paper_mode=paper,
                lr=0.01 if paper else 0.0003,
                gamma=0.95 if paper else 0.99,
                batch_size=128 if paper else 256,
                buffer_size=10000 if paper else 120000,
                hidden_dims=(1024, 1024, 512, 512, 256),
                eps_start=1., eps_end=0.05, eps_decay_steps=80000,
                target_update_tau=0. if paper else 0.005,
                target_update_freq=60 if paper else 2000,
                grad_clip=None if paper else 10.,
                use_per=True, per_alpha=0.6,
                per_beta_start=0. if paper else 0.4,
                per_beta_end=0. if paper else 1., per_beta_steps=80000)
