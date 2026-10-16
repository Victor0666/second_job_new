from pathlib import Path
import ast, gc, importlib, json, sys, time, traceback
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")

def main():
 import torch
 torch.set_num_threads(1)
 report = {"root": str(ROOT), "python": sys.executable, "cuda": torch.cuda.is_available(), "imports": [], "failures": [], "episodes": []}
 manifest = json.loads((ROOT / "extraction_manifest.json").read_text(encoding="utf-8"))
 for row in manifest["files"]:
  rel = Path(row["path"])
  if rel.suffix != ".py" or rel.parts[0] in ["e_draw_plot", "output"]: continue
  mod = ".".join(rel.with_suffix("").parts)
  if mod.endswith(".__init__"): mod = mod[:-9]
  try:
   ast.parse((ROOT / rel).read_text(encoding="utf-8-sig"))
   loaded = importlib.import_module(mod)
   if not Path(loaded.__file__).resolve().is_relative_to(ROOT):
    raise RuntimeError("Imported outside extracted package: " + str(loaded.__file__))
   for key in ["ROOT_DIR", "PROJECT_DIR", "PROJECT_ROOT"]:
    value = getattr(loaded, key, None)
    if isinstance(value, str) and Path(value).is_absolute() and not Path(value).resolve().is_relative_to(ROOT):
     raise RuntimeError(key + " points outside extracted package: " + value)
   report["imports"].append(mod)
  except Exception as e:
   report["failures"].append({"module": mod, "error": repr(e), "traceback": traceback.format_exc()})
 print("Imported",len(report["imports"]),"modules; failures",len(report["failures"]),flush=True)
 if "--imports-only" not in sys.argv and not report["failures"]:
  cases = [
   ("MAHDRL", "mahdrl", "build_env_kwargs_hrl_mix", "build_three_layer_agents_and_load", "evaluate_hrl_mix_one_seed", "EnvHRLMix"),
   ("NHA", "ablation_hostrand", "build_env_kwargs_hostrand", "build_agents_and_load", "evaluate_hostrand_one_seed", "EnvHOSTRAND"),
   ("NTA", "ablation_mgrrand", "build_env_kwargs_mgrrand", "build_two_agents_and_load", "evaluate_mgrrand_one_seed", "EnvHRLMix"),
   ("NVA", "ablation_vmrand", "build_env_kwargs_vm_random", "build_host_manager_agents_and_load", "evaluate_hrl_vm_random_one_seed", "EnvHRLVMRandom"),
   ("TL", "ablation_nohost", "build_env_kwargs_nohost", "build_two_layer_agents_and_load", "evaluate_nohost_one_seed", "EnvNoHost"),
  ]
  if "--main-only" in sys.argv: cases=cases[:1]
  for label, folder, kwfn, agentfn, evalfn, envname in cases:
   started=time.time()
   try:
    p=next((ROOT/"c_run"/folder).glob("*taskS_resS*Tight.py"))
    m=importlib.import_module(".".join(p.relative_to(ROOT).with_suffix("").parts))
    kw=getattr(m,kwfn)(random_seed=2)
    agents=getattr(m,agentfn)(kw)
    result=getattr(m,evalfn)(getattr(m,envname),kw,*agents)
    report["episodes"].append({"algorithm":label,"seconds":round(time.time()-started,2),"result":result})
    print(label,"episode complete",round(time.time()-started,2),"s",flush=True)
    del agents;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
   except Exception as e:
    report["failures"].append({"algorithm":label,"error":repr(e),"traceback":traceback.format_exc()})
    print(label,"FAILED",repr(e),flush=True)
  for label in ([] if "--main-only" in sys.argv else ["PD3QN", "IRWS", "MARL", "ERTF-H-FIFS"]):
   started=time.time()
   try:
    folders={"PD3QN":"pd3qn","IRWS":"irws","MARL":"marl","ERTF-H-FIFS":"ertf_fifs"}
    p=next((ROOT/"c_run"/folders[label]).glob("*taskS_resS*Tight.py"))
    m=importlib.import_module(".".join(p.relative_to(ROOT).with_suffix("").parts))
    if label=="PD3QN":
     kw,agent=m.build_agent_and_env_kwargs();kw["random_seed"]=2
     result=m.evaluate_one_episode(m.CloudWorkflowEnv_PD3QN_Cost,kw,agent)
     del agent
    elif label=="IRWS":
     kw=m.build_env_kwargs_irws(random_seed=2);selector,ppo=m.build_agents_and_load(kw)
     result=m.evaluate_one_seed(kw,selector,ppo,seed=2)
     del selector,ppo
    elif label=="MARL":
     cache=ROOT/"d_output/deadlines/ddl_fcfs_taskS_resS_seed0-1000.json"
     kw=m.build_env_kwargs(str(cache))
     ckpt=ROOT/"d_output/ckpts/marl_taskS_resS_eval1_ep1000_mix_Tight"
     device="cuda" if torch.cuda.is_available() else "cpu"
     built=m.build_agents_and_probe_dims(kw,2,device,str(ckpt/"best_global.pth"),str(ckpt/"best_local_actor_critics.pth"))
     result=m.evaluate_one_seed(kw,built[0],built[1],2,0)
     del built
    else:
     hm=importlib.import_module("c_run.mahdrl.eval_taskS_resS_Tight")
     kw=hm.build_env_kwargs_hrl_mix(random_seed=2)
     result=m.run_one_episode(m.CloudWorkflowEnv_VMAgents,kw,2)
    report["episodes"].append({"algorithm":label,"seconds":round(time.time()-started,2),"result":result})
    print(label,"episode complete",round(time.time()-started,2),"s",flush=True)
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
   except Exception as e:
    report["failures"].append({"algorithm":label,"error":repr(e),"traceback":traceback.format_exc()})
    print(label,"FAILED",repr(e),flush=True)
 (ROOT/"verification_result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
 print("Verification failures:",len(report["failures"]),flush=True)
 for x in report["failures"]: print(x,flush=True)
 return bool(report["failures"])
if __name__ == "__main__":sys.exit(main())
