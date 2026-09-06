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
def fsha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def sync(device):
 if str(device).startswith('cuda') and torch.cuda.is_available():torch.cuda.synchronize()
def bootstrap_ci(scores,indices):
 means=scores.detach().cpu()[indices].mean(1);q=torch.quantile(means,torch.tensor([.025,.975]));return [float(q[0]),float(q[1])]
def validate_authorization(auth,a,rows):
 ids=sorted(r['object_id'] for r in rows);here=pathlib.Path(__file__).resolve(); expected={'schema':'splart-node22-v2-authorization/v1','status':'AUTHORIZED_FOR_EVALUATOR','runner_sha256':fsha(here),'head_sha256':fsha(here.parent/'src/splart/node22_head.py'),'baseline_schema_sha256':fsha(here.parent/'node22_baseline_export_schema.json'),'supersession_sha256':fsha(here.parent/'evaluator_v2_supersession.json'),'truth':{'path':str(pathlib.Path(a.truth).resolve()),'sha256':fsha(a.truth)},'profile_indexes':[{'path':str(pathlib.Path(x).resolve()),'sha256':fsha(x)} for x in a.index],'profile_set_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),'baseline_export':{'path':str(pathlib.Path(a.baseline_export).resolve()),'sha256':fsha(a.baseline_export)},'output_absolute_path':str(pathlib.Path(a.output).resolve()),'one_execution_only':True,'v3_allowed':False}
 if set(auth)!=set(expected) or auth!=expected:raise ValueError('authorization binding mismatch')
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
def error_summary(pred,target,multiplier):
 e=(pred-target).abs()*multiplier[:,None];m=e.amax(1)
 return {'lower_nmae':float(e[:,0].mean()),'upper_nmae':float(e[:,1].mean()),'endpoint_nmae':float(m.mean()),'median_endpoint_nmae':float(m.median())},m
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
 if any(x in ' '.join(a.index+[a.truth,a.authorization,a.baseline_export,a.output]).lower() for x in FORBID):raise ValueError('protected path')
 out=pathlib.Path(a.output)
 if out.exists():raise FileExistsError('single execution output already exists')
 out.mkdir(mode=0o700,parents=True);state=out/'RUN_GUARD.json';atomic(state,{'stage':'PREFLIGHT','target_file_deserialized':False,'target_values_consumed_for_training':False,'optimizer_initialized':False,'retry_allowed':True});total_start=time.perf_counter()
 try:
  auth=json.load(open(a.authorization))
  rows=[]
  for p in a.index: rows+=json.load(open(p))['objects']
  assert len(rows)==len({r['object_id'] for r in rows})==36
  validate_authorization(auth,a,rows)
  ids={r['object_id'] for r in rows}; baseline=validate_baselines(json.load(open(a.baseline_export)),ids,json.load(open(pathlib.Path(__file__).with_name('node22_baseline_export_schema.json'))))
  payload={r['object_id']:torch.load(r['artifact'],map_location=a.device) for r in rows}
  truth=json.load(open(a.truth));atomic(state,{'stage':'TARGET_READ','target_file_deserialized':True,'target_values_consumed_for_training':False,'optimizer_initialized':False,'retry_allowed':False}); episodes=truth['episodes'];assert len(episodes)==len({e['object_id'] for e in episodes})==36 and {e['object_id'] for e in episodes}==set(payload)
  groups={k:sorted([e for e in episodes if e['split']==k],key=lambda x:x['object_id']) for k in ('train','calibration','confirmatory')};assert list(map(lambda k:len(groups[k]),groups))==[18,9,9]
  results={};internal_scores={}
  for variant in VARIANTS:
   sync(a.device);variant_start=time.perf_counter()
   tr=groups['train'];f=torch.stack([payload[e['object_id']]['features'] for e in tr]);s=torch.stack([payload[e['object_id']]['scalars'] for e in tr]);atomic(state,{'stage':'TARGET_VALUES_CONSUMING','target_file_deserialized':True,'target_values_consumed_for_training':True,'optimizer_initialized':False,'retry_allowed':False,'method':variant});y=torch.tensor([target_pair(e) for e in tr],device=a.device)
   mark=lambda:atomic(state,{'stage':'OPTIMIZER_INITIALIZED','target_file_deserialized':True,'target_values_consumed_for_training':True,'optimizer_initialized':True,'retry_allowed':False,'method':variant});m,n,receipt=train_once(f,s,y,[e['object_id'] for e in tr],variant,on_optimizer_initialized=mark)
   def pred(group):
    ff=torch.stack([payload[e['object_id']]['features'] for e in group]);ss=torch.stack([payload[e['object_id']]['scalars'] for e in group]);nf,nx=n(ff,ss);return m(nf,nx)
   cp,cs=pred(groups['calibration']);cy=torch.tensor([target_pair(e) for e in groups['calibration']],device=a.device);cm=torch.tensor([metric_multiplier(e) for e in groups['calibration']],device=a.device);conf=fit_joint_conformal(cp,cs,cy,[e['object_id'] for e in groups['calibration']]);scores=((cp-cy).abs()*cm[:,None]).amax(1);r=float(scores.sort().values[8])
   ep,es=pred(groups['confirmatory']);ey=torch.tensor([target_pair(e) for e in groups['confirmatory']],device=a.device);lo,hi=conf.interval(ep,es);em=torch.tensor([metric_multiplier(e) for e in groups['confirmatory']],device=a.device);cl=(ep-r/em[:,None]).clamp_min(0);ch=ep+r/em[:,None]
   mul=torch.tensor([metric_multiplier(e) for e in groups['confirmatory']],device=a.device);base,width=normalized_object_metrics(ep,ey,lo,hi,mul);rowsout=[{'object_id':str(i),'endpoint_nmae':base[i].item(),'joint_covered':float(((ey[i]>=lo[i])&(ey[i]<=hi[i])).all()),'mean_joint_width':width[i].item()} for i in range(9)]
   metrics,internal_scores[variant]=error_summary(ep,ey,em);sync(a.device);ag={'schema':'splart-node2.2-head-aggregate/v2','metrics':metrics,'joint_coverage':float(((ey>=lo)&(ey<=hi)).all(1).float().mean()),'normalized_mean_width':float(((hi-lo)*em[:,None]).mean()),'constant_joint_coverage':float(((ey>=cl)&(ey<=ch)).all(1).float().mean()),'constant_normalized_mean_width':float(((ch-cl)*em[:,None]).mean()),'swap_error':None if variant=='unshared_head' else exact_swap_error(m,*n(f,s)),'model_sha256':hstate(m),'runtime_config':receipt['config'],'wall_time_seconds':time.perf_counter()-variant_start};results[variant]=ag
  confirm=groups['confirmatory'];ey=torch.tensor([target_pair(e) for e in confirm],device=a.device);em=torch.tensor([metric_multiplier(e) for e in confirm],device=a.device)
  for name,mp in baseline.items():
   sync(a.device);bt=time.perf_counter();pp=torch.tensor([mp[e['object_id']] for e in confirm],device=a.device);metrics,internal_scores[name]=error_summary(pp,ey,em);sync(a.device);results[name]={'schema':'splart-node2.2-baseline-aggregate/v2','metrics':metrics,'wall_time_seconds':time.perf_counter()-bt,'runtime_scope':'sealed confirmatory aggregation only'}
  tr=groups['train'];ty=torch.tensor([target_pair(e) for e in tr],device=a.device);priors={'global_prior_train18':ty.mean(0),'range_prior_train18':ty.mean().repeat(2)}
  for name,pp in priors.items():sync(a.device);pt=time.perf_counter();metrics,internal_scores[name]=error_summary(pp[None].expand_as(ey),ey,em);sync(a.device);results[name]={'schema':'splart-node2.2-baseline-aggregate/v2','metrics':metrics,'wall_time_seconds':time.perf_counter()-pt,'runtime_scope':'sealed train18 prior fit plus confirmatory aggregation'}
  gen=torch.Generator().manual_seed(220290);boot=torch.randint(0,9,(10000,9),generator=gen);boot_sha=hashlib.sha256(boot.numpy().tobytes()).hexdigest()
  for name,scores in internal_scores.items():results[name]['metrics']['endpoint_nmae_bootstrap95_ci']=bootstrap_ci(scores,boot)
  results['shared']['wins_vs_full_d2']=int((internal_scores['shared']<internal_scores['full_d2']).sum());sync(a.device);final={'schema':'splart-node2.2-sealed-aggregate/v3','status':'PASS','methods':results,'wins_definition':'shared lower object max-side normalized NMAE than full_d2; count only','bootstrap':{'unit':'confirmatory object','replicates':10000,'analysis_rng_seed':220290,'training_seed':False,'index_sha256':boot_sha,'selection_use':False},'total_wall_time_seconds':time.perf_counter()-total_start,'objects':{'train':18,'calibration':9,'confirmatory':9},'per_object_values_emitted':False,'membership_emitted':False,'targets_emitted':False,'protected_splits_read':[]};atomic(out/'RESULT.json',final);atomic(state,{'stage':'COMPLETE','target_file_deserialized':True,'target_values_consumed_for_training':True,'optimizer_initialized':True,'retry_allowed':False});print(json.dumps({'status':'PASS','result':str(out/'RESULT.json')}))
 except Exception as e:
  prior=json.loads(state.read_text());prior.update({'failed_from_stage':prior.get('stage'),'stage':'FAILED_TERMINAL','retry_allowed':False,'error_type':type(e).__name__});atomic(state,prior);raise
if __name__=='__main__':main()
