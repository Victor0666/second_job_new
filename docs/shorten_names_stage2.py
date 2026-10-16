from pathlib import Path
import sys
base=Path(__file__).with_name('shorten_names.py')
code=base.read_text(encoding='utf-8-sig')
more=[
('rewardDE_mix_r075_ddlMix_HVrn','r075_ddlMix_hv'),
('rewardDE_mix_r075_ddlMix_noHost','nohost_r075_ddlMix'),
('mgrRAND_r075_ddlMix_HVrn','mgrrand_r075_ddlMix_hv'),
('_hv_HOSTRAND','_hostrand_hv'),('_hv_vmRandom','_vmrand_hv'),
('ckpts_mahdrl_','mahdrl_'),('logs_mahdrl_','mahdrl_'),
('ckpts_hrl2_','hrl2_'),('logs_hrl2_','hrl2_'),
('ckpts_irws_ppo_','irws_'),('logs_irws_ppo_','irws_'),
('ckpts_pd3qn_','pd3qn_'),('logs_pd3qn_','pd3qn_'),
('ckpts_marl_','marl_'),('logs_marl_','marl_'),
('_FCFS_ddlMix_','_mix_'),
('deadlineCACHE_FCFS_alpha075_task_and_wf_metrics_ddlMix','r075_mix'),
('run_hrl_routeA_','eval_mahdrl_'),('run_hrl_mgrEnergyDelay_','eval_hrl_de_'),
('run_hrl0p75_routeA_','eval_hrl_r075_'),
('_task_and_wf_metrics','_metrics'),('rewardDE','mahdrl')]
code=code.replace('RULES=[','RULES='+repr(more)+'+[')
code=code.replace('name_shortening_backup.zip','name_shortening_stage2_backup.zip').replace('name_shortening.json','name_shortening_stage2.json')
exec(compile(code,str(base),'exec'),{'__file__':str(base),'__name__':'__main__'})
