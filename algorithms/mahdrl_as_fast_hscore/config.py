"""Only HA scores a variable-size set; VA always has nine local actions."""
from algorithms.mahdrl_as_fast_vaaware.config import Mechanism, TA_HIDDEN, learning_kwargs
from algorithms.mahdrl_as_fast_vaaware.config import variant_metadata as base_metadata
from algorithms.mahdrl_oe_ge_apf_score.config import TRUNK

ALGORITHM = "mahdrl_as_fast_hscore"
DISPLAY_NAME = "MAHDRL-AS-Fast-VAware-HScore-FixedVA"
MODEL_VERSION = "mahdrl_hscore_fixedva_v1"
OUTPUT_TAG = "mahdrl_hscore"
CHECKPOINT_METADATA_KEY = "mahdrl_hscore"
VM_CAPACITY = 9
VM_CONTEXT_DIM = 30
VM_FEATURE_DIM = 7
VM_INPUT_DIM = VM_CONTEXT_DIM + VM_CAPACITY * VM_FEATURE_DIM


def variant_metadata(mechanism=Mechanism()):
    metadata = base_metadata(mechanism)
    metadata.update(algorithm=ALGORITHM, scheme=MODEL_VERSION, model_version=MODEL_VERSION,
        base_implementation="mahdrl_as_fast_vaaware_noapf_v1",
        architecture="ha_shared_scoring_va_fixed_slot_dueling_v1",
        observation_layout=f"host23x{mechanism.host_features}_vm93_fixed9_v1",
        pooling="HA_exists_mean_max_counts_VA_none",
        fixed_va=dict(input_dim=VM_INPUT_DIM, output_dim=VM_CAPACITY,
            context_dim=VM_CONTEXT_DIM, slot_features=VM_FEATURE_DIM,
            capacity=VM_CAPACITY, padding="zero_right_pad_masked",
            slot_order="existing_host_to_vm_indices_no_reordering",
            dueling_center="legal_actions_only", shared_across_hosts=True,
            host_identity="physical_features_without_host_one_hot"))
    metadata["components"] = [c for c in metadata["components"] if c != "shared_scoring"]
    metadata["components"] += ["HA_shared_scoring", "VA_fixed_slot_D3QN"]
    return metadata


def implementation_metadata(mechanism=Mechanism()):
    return dict(algorithm=ALGORITHM, implementation_version=MODEL_VERSION,
        scientific_variant=variant_metadata(mechanism),
        replay="HA_frozen_set_VA_frozen_93_vector_9_mask",
        environment="raw_legal_noapf_float64_fixed9_adapter",
        preview="batched_fixed93_no_grad_no_rng_cached9",
        checkpoint_selection="select_sr092_minenergy_v1")
