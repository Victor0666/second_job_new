from pathlib import Path
import ast,hashlib,io,json,re,sys,tokenize,zipfile
R=Path(__file__).resolve().parents[1]
SELF=Path(__file__).resolve()
# Shared fragments also cover f-string templates, glob patterns and fallback filters.
RULES=[
('hrl_3layer_routeA_mgc_ave','mahdrl'),('hrl_2layer_routeA_mgc_ave','hrl2'),
('pd3qn_cost_deadlineCACHE_FCFS_alpha15','pd3qn'),
('per_banditWarmup_saveTime_ddlAlpha_Mix_nr','warmup_mix_nr'),
('per_banditWarmup','warmup'),('marl_a2c','marl'),
('evl1_FCFS_1000eps_ddl_Mix','eval1_ep1000_mix'),
('mgrDelayEnergy','rewardDE'),('rAlpha075','r075'),('dalphaMix','ddlMix'),
('ddlAlpha_Mix','ddlMix'),('Mix_ddlAlpha','ddlMix'),
('deadline_cache_fcfs_vmfirst','ddl_fcfs'),
('smallTask','taskS'),('medTask','taskM'),('largeTask','taskL'),
('smallRes','resS'),('medRes','resM'),('largeRes','resL'),
('alignPD3QNReward','rewardPD3QN'),('local_shared_actor_multi_critic','local_actor_critics'),
('train_metrics_3layer_routeA_hostrand','train_hostrand'),
('train_metrics_3layer_routeA_vmRandom','train_vmrand'),
('train_metrics_2layer_routeA_noHost','train_nohost'),
('train_metrics_3layer_routeA','train'),
('precompute_deadline_cache_fcfs_vmfirst','precompute_ddl_fcfs'),
]
MODULES={
'env_cloud_workflow_d3qn_state_reworked_multiagent_Tsize_alpha15_DDLCACHE_FCFS_noHost':'env_nohost',
'env_cloud_workflow_d3qn_state_reworked_multiagent_Tsize_alpha15_DDLCACHE_FCFS':'env_fcfs',
'env_cloud_workflow_d3qn_state_reworked_multiagent_Tsize_alpha15_DDLHEFT':'env_heft',
'env_cloud_workflow_marl_Tsize_alpha15_FCFS':'env_marl',
'env_cloud_workflow_pd3qn_Tsize_cost':'env_pd3qn',
'env_irws_routeA_Tsize_alpha15':'env_irws',
'plot_train_eval_hrl_3layer_triplet_mgc':'plot_hrl_triplet',
'plot_train_eval_hrl_new_triplet_Mix_ddlAlpha_merge_One_short':'plot_hrl_mix_one_short',
'plot_train_eval_hrl_new_triplet_Mix_ddlAlpha_merge_Three':'plot_hrl_mix_three',
'plot_train_eval_hrl_new_triplet_Mix_ddlAlpha_merge_One':'plot_hrl_mix_one',
'plot_train_eval_hrl_new_triplet_Mix_ddlAlpha':'plot_hrl_mix',
'algorithm_boxPlot_Merge_byDDL_groupedAlgo':'plot_algorithms_by_ddl',
'Wilcoxon_signed_rank_Mix_ddlAlpha_OK':'wilcoxon_mix',
}
def compact(s):
 for a,b in RULES:s=s.replace(a,b)
 return s

def excluded(p):
 return any(x in {'.venv','__pycache__','.git'} for x in p.relative_to(R).parts) or p==SELF or p.relative_to(R).parts[0] in {'docs','output'}
paths=[p for p in R.rglob('*') if not excluded(p)]
files=[p for p in paths if p.is_file()]
# Training/evaluation scenario belongs in the filename; algorithm belongs in its parent.
for p in files:
 if p.suffix=='.py' and p.stem.startswith(('train_','run_')):
  m=re.search(r'(small|med|large)Task_(small|med|large)Res.*_(Tight|Medium|Loose)$',p.stem)
  if m:
   size={'small':'S','med':'M','large':'L'}
   MODULES[p.stem]=f"{'train' if p.stem.startswith('train_') else 'eval'}_task{size[m[1]]}_res{size[m[2]]}_{m[3]}"
MP=re.compile(r'(?<![A-Za-z0-9_])('+'|'.join(map(re.escape,sorted(MODULES,key=len,reverse=True)))+r')(?![A-Za-z0-9_])')
def rewrite(s):return compact(MP.sub(lambda m:MODULES[m[0]],s))
def dest(p):return R.joinpath(*(rewrite(x) for x in p.relative_to(R).parts))
allmap={p.relative_to(R).as_posix():dest(p).relative_to(R).as_posix() for p in paths if dest(p)!=p}
seen={}
for p in paths:
 key=str(dest(p)).casefold()
 if key in seen:raise RuntimeError(f'Collision: {p} and {seen[key]} -> {dest(p)}')
 seen[key]=p
changes={}
for p in files:
 if p.suffix not in {'.py','.md','.cmd'}:continue
 raw=p.read_bytes();s=raw.decode('utf-8-sig')
 if p.suffix=='.py':
  lines=s.splitlines(keepends=True);offset=[0]
  for line in lines:offset.append(offset[-1]+len(line))
  edits=[]
  for t in tokenize.generate_tokens(io.StringIO(s).readline):
   new=t.string
   if t.type in {tokenize.STRING,tokenize.COMMENT}:new=rewrite(new)
   elif t.type==tokenize.NAME and t.string in MODULES:new=MODULES[t.string]
   if new!=t.string:edits.append((offset[t.start[0]-1]+t.start[1],offset[t.end[0]-1]+t.end[1],new))
  new=s
  for a,b,t in reversed(edits):new=new[:a]+t+new[b:]
  ast.parse(new)
 else:new=rewrite(s)
 if new!=s:changes[p]=new.encode('utf-8-sig' if raw.startswith(b'\xef\xbb\xbf') else 'utf-8')
print(json.dumps({'renamed_files':sum(p.is_file() and dest(p)!=p for p in paths),'renamed_directories':sum(p.is_dir() and p.name!=dest(p).name for p in paths),'modified_sources':len(changes),'examples':dict(list(allmap.items())[:8])},ensure_ascii=False,indent=2))
if '--apply' not in sys.argv:sys.exit()
B=R/'docs/name_shortening_backup.zip'
assert not B.exists(),'Migration already applied'
pre={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
historical=['extraction_manifest.json','package_checksums.json','verification_result.json']
with zipfile.ZipFile(B,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files:
  if p in changes or p.name in historical:z.write(p,p.relative_to(R).as_posix())
 z.writestr('mapping.json',json.dumps(allmap,ensure_ascii=False,indent=2))
for p,data in changes.items():p.write_bytes(data)
# Rename children first. Every destination is constrained to the package root.
for p in sorted(paths,key=lambda p:len(p.parts),reverse=True):
 name=rewrite(p.name)
 if name!=p.name:
  target=p.with_name(name)
  assert p.resolve().is_relative_to(R) and target.resolve().is_relative_to(R) and not target.exists()
  p.rename(target)
manifest=json.loads((R/'extraction_manifest.json').read_text(encoding='utf-8'))
for row in manifest['files']:
 old=row['path'];new='/'.join(rewrite(x) for x in Path(old).parts)
 if old!=new:row.setdefault('source_path',old);row['path']=new
(R/'extraction_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
report={'mapping':allmap,'source_changes':[dest(p).relative_to(R).as_posix() for p in changes],'pre_file_sha256':pre,'renamed_files':sum(Path(k).suffix!='' for k in allmap),'rules':RULES,'module_stems':MODULES,'backup':str(B)}
(R/'docs/name_shortening.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
# Assert every existing artifact survives, with bytes unchanged except the listed text edits.
for p in files:
 new=dest(p);assert new.is_file(),new
 if p not in changes and p.name!='extraction_manifest.json':assert hashlib.sha256(new.read_bytes()).hexdigest()==pre[p.relative_to(R).as_posix()],new
print('APPLIED; all pre-existing artifacts preserved')
