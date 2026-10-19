"""Summarize fixed-model, paired-instance measurements; no policy execution."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats
from common.scenarios import parse_seeds


def interval(values, confidence=.95):
    values=np.asarray(values,dtype=float);n=len(values)
    mean=float(values.mean());sd=float(values.std(ddof=1)) if n>1 else 0.
    half=float(stats.t.ppf((1+confidence)/2,n-1)*sd/np.sqrt(n)) if n>1 else None
    return dict(n=n,mean=mean,std=sd,mean_ci=[mean-half,mean+half] if half is not None else None,
                ci_method='Student t over instance seeds, conditional on one trained model')


def load(directory,seeds):
    rows=[]
    for seed in seeds:
        p=Path(directory)/'episodes'/f'seed{seed}.json'
        row=json.loads(p.read_text(encoding='utf-8'))
        if row['seed']!=seed:raise ValueError('Measurement seed mismatch: '+str(p))
        rows.append(row)
    for key in ('algorithm','model_seed','task','resource','deadline','checkpoints'):
        if any(r.get(key)!=rows[0].get(key) for r in rows):
            raise ValueError('Mixed model, scenario or checkpoint: '+key)
    return rows


def summarize(directory,seeds,qos=.99,reference=None,confidence=.95):
    if not 0<=qos<=1 or not 0<confidence<1:raise ValueError('Invalid QoS/confidence')
    rows=load(directory,seeds)
    metrics={key:interval([r['metrics'][key] for r in rows],confidence) for key in (
        'total_energy','makespan','wf_success_rate','wf_avg_tardiness','wf_avg_response_time',
        'task_success_rate','task_avg_tardiness')}
    # The Bernoulli event is explicitly per-episode WSR >= qos, not per-workflow success.
    n=len(rows);k=sum(r['metrics']['wf_success_rate']>=qos for r in rows);alpha=1-confidence
    bounds=[float(stats.beta.ppf(alpha/2,k,n-k+1)) if k else 0.,
            float(stats.beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.]
    result=dict(algorithm=rows[0]['algorithm'],model_seed=rows[0]['model_seed'],seeds=list(seeds),
        metrics=metrics,qos_event=dict(definition='per-instance workflow success rate >= threshold',threshold=qos,
            successes=k,trials=n,empirical_probability=k/n,confidence=confidence,exact_binomial_ci=bounds,
            assumptions='independent instance seeds; workflows within an instance are not counted as independent trials'),
        training_variance_estimated=False,latency=dict(scope=rows[0]['timing']['scope'],
            mean_of_instance_decision_mean_ms=interval([r['timing']['decision_mean_ms'] for r in rows],confidence)))
    decisions=[]
    for seed in seeds:
        path=Path(directory)/'decisions'/f'seed{seed}.json'
        if path.exists():decisions.extend(json.loads(path.read_text(encoding='utf-8'))['allocation_seconds'])
    if decisions:
        result['latency']['pooled_allocation_descriptive_ms']=dict(count=len(decisions),mean=1000*float(np.mean(decisions)),
            p95=1000*float(np.percentile(decisions,95)),p99=1000*float(np.percentile(decisions,99)))
    if reference:
        base=load(reference,seeds)
        for a,b in zip(rows,base):
            if (a['task'],a['resource'],a['deadline'],a['seed'])!=(b['task'],b['resource'],b['deadline'],b['seed']):
                raise ValueError('Cannot pair different scenarios or seeds')
            pa=a.get('provenance',{});pb=b.get('provenance',{})
            for key in ('data_split','deadline_cache','instance_generator'):
                va=pa.get(key);vb=pb.get(key)
                if key=='deadline_cache':va=va.get('sha256') if va else None;vb=vb.get('sha256') if vb else None
                if va!=vb:raise ValueError('Cannot pair different physical instances: '+key)
        pairs={}
        for key in ('total_energy','wf_success_rate','wf_avg_tardiness'):
            a=np.asarray([r['metrics'][key] for r in rows]);b=np.asarray([r['metrics'][key] for r in base]);diff=a-b
            pairs[key]=dict(difference=interval(diff,confidence),
                wilcoxon_p_uncorrected=float(stats.wilcoxon(diff).pvalue) if np.any(diff) else 1.)
            if key=='total_energy':
                pairs[key]['energy_reduction_percent_from_means']=float(100*(b.mean()-a.mean())/b.mean()) if b.mean()!=0 else None
        result['paired_reference']=dict(algorithm=base[0]['algorithm'],model_seed=base[0]['model_seed'],comparisons=pairs,
            multiplicity='p-values uncorrected; define and adjust the comparison family before publication')
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--directory',type=Path,required=True,help='Measurement directory containing episodes/seed*.json')
    p.add_argument('--reference',type=Path);p.add_argument('--seeds',default='700-729')
    p.add_argument('--qos-threshold',type=float,default=.99);p.add_argument('--confidence',type=float,default=.95)
    p.add_argument('--out',type=Path)
    a=p.parse_args();result=summarize(a.directory,parse_seeds(a.seeds),a.qos_threshold,a.reference,a.confidence)
    output=a.out or a.directory/'statistical_summary.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(output)


if __name__=='__main__':main()
