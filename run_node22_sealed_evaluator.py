#!/usr/bin/env python3
import argparse,hashlib,json,os,pathlib,tempfile,time,torch
from splart.node22_head import VARIANTS,FROZEN_CONFIG,aggregate_confirmatory,exact_swap_error,fit_joint_conformal,train_once
FORBID=('b_test','full22')
def atomic(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n');os.replace(t,p)
def hstate(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode()+v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def main():
 q=argparse.ArgumentParser();q.add_argument('--index',action='append',required=True);q.add_argument('--truth',required=True);q.add_argument('--authorization',required=True);q.add_argument('--output',required=True);q.add_argument('--device',default='cuda');a=q.parse_args()
 if any(x in ' '.join(vars(a).values() if False else a.index+[a.truth,a.authorization,a.output]).lower() for x in FORBID):raise ValueError('protected path')
 out=pathlib.Path(a.output)
 if out.exists():raise FileExistsError('single execution output already exists')
 out.mkdir(mode=0o700,parents=True);state=out/'RUN_GUARD.json';atomic(state,{'stage':'PREFLIGHT','target_read':False,'optimizer_initialized':False,'retry_allowed':True})
 try:
  auth=json.load(open(a.authorization));assert auth['status']=='AUTHORIZED_FOR_EVALUATOR'
  rows=[]
  for p in a.index: rows+=json.load(open(p))['objects']
  assert len(rows)==len({r['object_id'] for r in rows})==36
  payload={r['object_id']:torch.load(r['artifact'],map_location=a.device) for r in rows}
  truth=json.load(open(a.truth));atomic(state,{'stage':'TARGET_READ','target_read':True,'optimizer_initialized':False,'retry_allowed':False}); episodes=truth['episodes'];assert len(episodes)==36
  groups={k:sorted([e for e in episodes if e['split']==k],key=lambda x:x['object_id']) for k in ('train','calibration','confirmatory')};assert list(map(lambda k:len(groups[k]),groups))==[18,9,9]
  results={}
  for variant in VARIANTS:
   tr=groups['train'];f=torch.stack([payload[e['object_id']]['features'] for e in tr]);s=torch.stack([payload[e['object_id']]['scalars'] for e in tr]);y=torch.tensor([[e['endpoint_truth']['physical_endpoint_distances_normalized_by_range']['lower'],e['endpoint_truth']['physical_endpoint_distances_normalized_by_range']['upper']] for e in tr],device=a.device)
   atomic(state,{'stage':'OPTIMIZER_INITIALIZED','target_read':True,'optimizer_initialized':True,'retry_allowed':False,'method':variant});m,n,receipt=train_once(f,s,y,[e['object_id'] for e in tr],variant)
   def pred(group):
    ff=torch.stack([payload[e['object_id']]['features'] for e in group]);ss=torch.stack([payload[e['object_id']]['scalars'] for e in group]);nf,nx=n(ff,ss);return m(nf,nx)
   cp,cs=pred(groups['calibration']);cy=torch.tensor([[e['endpoint_truth']['physical_endpoint_distances_normalized_by_range']['lower'],e['endpoint_truth']['physical_endpoint_distances_normalized_by_range']['upper']] for e in groups['calibration']],device=a.device);conf=fit_joint_conformal(cp,cs,cy,[e['object_id'] for e in groups['calibration']]);const=fit_joint_conformal(cp,cs,cy,[e['object_id'] for e in groups['calibration']],kind='constant')
   ep,es=pred(groups['confirmatory']);ey=torch.tensor([[e['endpoint_truth']['physical_endpoint_distances_normalized_by_range']['lower'],e['endpoint_truth']['physical_endpoint_distances_normalized_by_range']['upper']] for e in groups['confirmatory']],device=a.device);lo,hi=conf.interval(ep,es);cl,ch=const.interval(ep,es)
   base=(ep-ey).abs().mean(1);rowsout=[{'object_id':str(i),'endpoint_nmae':base[i].item(),'joint_covered':float(((ey[i]>=lo[i])&(ey[i]<=hi[i])).all()),'mean_joint_width':(hi[i]-lo[i]).mean().item()} for i in range(9)]
   ag=aggregate_confirmatory(rowsout);ag.update({'constant_joint_coverage':float(((ey>=cl)&(ey<=ch)).all(1).float().mean()),'constant_mean_joint_width':float((ch-cl).mean()),'swap_error':None if variant=='unshared_head' else exact_swap_error(m,*n(f,s)),'model_sha256':hstate(m),'runtime_config':receipt['config']});results[variant]=ag
  full=results['shared']['metrics']['endpoint_nmae'];wins=sum(results[v]['metrics']['endpoint_nmae']<full for v in VARIANTS if v!='shared');final={'schema':'splart-node2.2-sealed-aggregate/v1','status':'PASS','methods':results,'baseline_wins_over_full':wins,'objects':{'train':18,'calibration':9,'confirmatory':9},'membership_emitted':False,'targets_emitted':False,'protected_splits_read':[]};atomic(out/'RESULT.json',final);atomic(state,{'stage':'COMPLETE','target_read':True,'optimizer_initialized':True,'retry_allowed':False});print(json.dumps({'status':'PASS','result':str(out/'RESULT.json')}))
 except Exception as e:
  atomic(state,{'stage':'FAILED_TERMINAL','target_read':True,'retry_allowed':False,'error_type':type(e).__name__});raise
if __name__=='__main__':main()
