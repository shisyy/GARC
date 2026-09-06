#!/usr/bin/env python3
import argparse,hashlib,json,math,os,pathlib,tempfile,time,torch
from splart.node22_head import VARIANTS,FROZEN_CONFIG,aggregate_confirmatory,exact_swap_error,fit_joint_conformal,train_once
FORBID=('b_test','full22')
def atomic(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n');os.replace(t,p)
def hstate(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode()+v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def target_pair(e):
 t=e['endpoint_truth']['local_endpoint_targets']
 if set(t)!={'extension0_joint_units','extension0_observation_units','extension1_joint_units','extension1_observation_units','local_lower_scalar','local_upper_scalar'}:raise ValueError('truth schema mismatch')
 if any(type(t[k]) not in (int,float) for k in t):raise ValueError('truth scalar type mismatch')
 return [float(t['extension0_observation_units']),float(t['extension1_observation_units'])]
def metric_multiplier(e):
 n=e['endpoint_truth']['target_nmae_normalization']; expected={'local_scalar_error_multiplier','object_score_formula','per_endpoint_formula','physical_range'}
 if set(n)!=expected or type(n['local_scalar_error_multiplier']) not in (int,float) or not n['local_scalar_error_multiplier']>0 or type(n['physical_range']) not in (int,float) or not n['physical_range']>0 or n['per_endpoint_formula']!='abs(predicted_local_scalar-target_local_scalar)*local_scalar_error_multiplier' or n['object_score_formula']!='max(lower_endpoint_nmae,upper_endpoint_nmae)':raise ValueError('normalization schema mismatch')
 return float(n['local_scalar_error_multiplier'])
def normalized_object_metrics(pred,target,lo,hi,multiplier):
 err=(pred-target).abs()*multiplier[:,None]; width=(hi-lo)*multiplier[:,None]
 return err.amax(1),width.mean(1)
def validate_baselines(b,ids,schema):
 if b.get('schema')!='splart-node22-target-free-baseline-predictions/v2' or set(b.get('methods',{}))!=set(schema['required_methods']):raise ValueError('baseline schema/methods')
 forbidden=set(schema['forbidden_fields'])
 def deny(v):
  if isinstance(v,dict):
   if set(map(str.lower,v))&forbidden:raise ValueError('baseline private/aggregate field')
   for x in v.values():deny(x)
  elif isinstance(v,list):
   for x in v:deny(x)
 deny(b);out={}
 for name,p in b['methods'].items():
  if set(p)!={'rows','provenance'} or set(p['provenance'])!={'source_commit','source_tree','config_sha256','prediction_tree_sha256'}:raise ValueError('baseline provenance')
  rr=p['rows'];rid=[x.get('object_id') for x in rr]
  if len(rr)!=len(set(rid))==36 or set(rid)!=ids:raise ValueError('baseline IDs')
  for x in rr:
   if set(x)!={'object_id','distances'} or len(x['distances'])!=2 or any(type(z) not in (int,float) or not math.isfinite(z) or z<0 for z in x['distances']):raise ValueError('baseline distance')
  out[name]={x['object_id']:x['distances'] for x in rr}
 return out
def main():
 q=argparse.ArgumentParser();q.add_argument('--index',action='append',required=True);q.add_argument('--truth',required=True);q.add_argument('--authorization',required=True);q.add_argument('--baseline-export',required=True);q.add_argument('--output',required=True);q.add_argument('--device',default='cuda');a=q.parse_args()
 if any(x in ' '.join(vars(a).values() if False else a.index+[a.truth,a.authorization,a.output]).lower() for x in FORBID):raise ValueError('protected path')
 out=pathlib.Path(a.output)
 if out.exists():raise FileExistsError('single execution output already exists')
 out.mkdir(mode=0o700,parents=True);state=out/'RUN_GUARD.json';atomic(state,{'stage':'PREFLIGHT','target_read':False,'optimizer_initialized':False,'retry_allowed':True})
 try:
  auth=json.load(open(a.authorization));runner_sha=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest();assert auth['schema']=='splart-node22-v2-authorization/v1' and auth['status']=='AUTHORIZED_FOR_EVALUATOR' and auth['runner_sha256']==runner_sha
  rows=[]
  for p in a.index: rows+=json.load(open(p))['objects']
  assert len(rows)==len({r['object_id'] for r in rows})==36
  ids={r['object_id'] for r in rows}; baseline=validate_baselines(json.load(open(a.baseline_export)),ids,json.load(open(pathlib.Path(__file__).with_name('node22_baseline_export_schema.json'))))
  payload={r['object_id']:torch.load(r['artifact'],map_location=a.device) for r in rows}
  truth=json.load(open(a.truth));atomic(state,{'stage':'TARGET_READ','target_file_deserialized':True,'target_values_consumed':False,'optimizer_initialized':False,'retry_allowed':False}); episodes=truth['episodes'];assert len(episodes)==len({e['object_id'] for e in episodes})==36 and {e['object_id'] for e in episodes}==set(payload)
  groups={k:sorted([e for e in episodes if e['split']==k],key=lambda x:x['object_id']) for k in ('train','calibration','confirmatory')};assert list(map(lambda k:len(groups[k]),groups))==[18,9,9]
  results={}
  for variant in VARIANTS:
   tr=groups['train'];f=torch.stack([payload[e['object_id']]['features'] for e in tr]);s=torch.stack([payload[e['object_id']]['scalars'] for e in tr]);y=torch.tensor([target_pair(e) for e in tr],device=a.device)
   atomic(state,{'stage':'OPTIMIZER_INITIALIZED','target_read':True,'optimizer_initialized':True,'retry_allowed':False,'method':variant});m,n,receipt=train_once(f,s,y,[e['object_id'] for e in tr],variant)
   def pred(group):
    ff=torch.stack([payload[e['object_id']]['features'] for e in group]);ss=torch.stack([payload[e['object_id']]['scalars'] for e in group]);nf,nx=n(ff,ss);return m(nf,nx)
   cp,cs=pred(groups['calibration']);cy=torch.tensor([target_pair(e) for e in groups['calibration']],device=a.device);cm=torch.tensor([metric_multiplier(e) for e in groups['calibration']],device=a.device);conf=fit_joint_conformal(cp,cs,cy,[e['object_id'] for e in groups['calibration']]);scores=((cp-cy).abs()*cm[:,None]).amax(1);r=float(scores.sort().values[8])
   ep,es=pred(groups['confirmatory']);ey=torch.tensor([target_pair(e) for e in groups['confirmatory']],device=a.device);lo,hi=conf.interval(ep,es);em=torch.tensor([metric_multiplier(e) for e in groups['confirmatory']],device=a.device);cl=(ep-r/em[:,None]).clamp_min(0);ch=ep+r/em[:,None]
   mul=torch.tensor([metric_multiplier(e) for e in groups['confirmatory']],device=a.device);base,width=normalized_object_metrics(ep,ey,lo,hi,mul);rowsout=[{'object_id':str(i),'endpoint_nmae':base[i].item(),'joint_covered':float(((ey[i]>=lo[i])&(ey[i]<=hi[i])).all()),'mean_joint_width':width[i].item()} for i in range(9)]
   ag=aggregate_confirmatory(rowsout);ag.update({'constant_joint_coverage':float(((ey>=cl)&(ey<=ch)).all(1).float().mean()),'constant_mean_joint_width':float(((ch-cl)*em[:,None]).mean()),'swap_error':None if variant=='unshared_head' else exact_swap_error(m,*n(f,s)),'model_sha256':hstate(m),'runtime_config':receipt['config']});results[variant]=ag
  confirm=groups['confirmatory'];ey=torch.tensor([target_pair(e) for e in confirm],device=a.device);em=torch.tensor([metric_multiplier(e) for e in confirm],device=a.device)
  for name,mp in baseline.items():
   pp=torch.tensor([mp[e['object_id']] for e in confirm],device=a.device);score=((pp-ey).abs()*em[:,None]).amax(1);results[name]={'schema':'splart-node2.2-baseline-aggregate/v1','metrics':{'endpoint_nmae':float(score.mean())}}
  tr=groups['train'];ty=torch.tensor([target_pair(e) for e in tr],device=a.device);priors={'global_prior_train18':ty.mean(0),'range_prior_train18':ty.mean().repeat(2)}
  for name,pp in priors.items():results[name]={'schema':'splart-node2.2-baseline-aggregate/v1','metrics':{'endpoint_nmae':float((((pp[None]-ey).abs()*em[:,None]).amax(1)).mean())}}
  full=results['shared']['metrics']['endpoint_nmae'];competitors=[k for k in results if k!='shared'];wins=sum(results[k]['metrics']['endpoint_nmae']<full for k in competitors);final={'schema':'splart-node2.2-sealed-aggregate/v2','status':'PASS','methods':results,'wins_definition':'number of non-shared methods with lower confirmatory object-macro max-side normalized NMAE than shared','baseline_wins_over_full':wins,'competitor_count':len(competitors),'objects':{'train':18,'calibration':9,'confirmatory':9},'membership_emitted':False,'targets_emitted':False,'protected_splits_read':[]};atomic(out/'RESULT.json',final);atomic(state,{'stage':'COMPLETE','target_read':True,'optimizer_initialized':True,'retry_allowed':False});print(json.dumps({'status':'PASS','result':str(out/'RESULT.json')}))
 except Exception as e:
  atomic(state,{'stage':'FAILED_TERMINAL','target_read':True,'retry_allowed':False,'error_type':type(e).__name__});raise
if __name__=='__main__':main()
