"""Paper learning settings are independent of the shared observation encoding."""


def agent_kwargs(scenario):
    return dict(paper_mode=True, lr=.01, gamma=.95, batch_size=128,
                buffer_size=10000, hidden_dims=(1024, 1024, 512, 512, 256),
                eps_start=1., eps_end=.05, eps_decay_steps=80000,
                target_update_tau=0., target_update_freq=60, grad_clip=None,
                use_per=True, per_alpha=.6, per_beta_start=0., per_beta_end=0.,
                per_beta_steps=80000)
