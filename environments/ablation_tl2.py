"""TL_2 changes only the original TL VA energy scope; TA and dynamics are inherited."""
from environments.ablation_nohost import CloudWorkflowEnv_VMAgents as _TL
from environments.ablation_nohost import MANAGER_ACTION_TABLE


class CloudWorkflowEnv_VMAgents(_TL):
    vm_energy_scope = "global"
