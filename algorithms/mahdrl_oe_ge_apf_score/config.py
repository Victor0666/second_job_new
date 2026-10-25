"""Versioned, variant-local settings; importing this module changes no registry."""
from dataclasses import asdict, dataclass

ALGORITHM = "mahdrl_oe_ge_apf_score"
DISPLAY_NAME = "MAHDRL-AS (OE + GE + APF + shared scoring)"
ARCHITECTURE = "shared_candidate_dueling_v1"
OBSERVATION = "task23_host10_taskhost30_vm7_v1"
MODEL_VERSION = "mahdrl_as_v1"


@dataclass(frozen=True)
class APFConfig:
    scheme: str = "adaptive_pf_energy_v1"
    host_base: float = 1.5
    vm_base: float = 1.25
    spare_weight: float = 0.5
    urgency_weight: float = 0.25
    lower: float = 1.25
    upper: float = 2.0
    guided_probability: float = 0.5
    time_atol: float = 1e-9
    energy_rtol: float = 1e-9
    energy_atol: float = 1e-9
    energy_retention: str = "all_tied_minimum_marginal_energy_among_ontime_raw_legal"


APF = APFConfig()
TRUNK = (1024, 1024, 512, 512, 256)
ENTITY = (128, 128)
TA_HIDDEN = (512, 512, 256, 128)


def variant_metadata():
    return dict(algorithm=ALGORITHM, scheme=MODEL_VERSION, model_version=MODEL_VERSION,
                architecture=ARCHITECTURE, observation_layout=OBSERVATION,
                components=["OE", "GE", "APF", "shared_scoring"],
                oe_base_scheme="mahdrl_oe_1235_v1", apf=asdict(APF),
                host_summary_scope="final_apf_candidates",
                deadline_action_filter=False,
                pooling="exists_mask_mean_max_log1p_real_and_legal_counts")


def learning_kwargs(role, device):
    return dict(lr=3e-4, gamma=.95, batch_size=256,
                buffer_size=60000 if role == "TA" else 120000,
                eps_start=1., eps_end=.05,
                eps_decay_steps=40000 if role == "TA" else 80000,
                target_update_tau=.01 if role == "TA" else .005,
                grad_clip=10., device=device)
