import json,hashlib,pathlib,torch,sys
old,new,inv,salt,out=map(pathlib.Path,sys.argv[1:]); H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); fail=[]
def ck(x,n):
  if not x: fail.append(n)
a=json.loads(old.read_text());b=json.loads(new.read_text()); rows=a['objects']+b['objects']; ids=[x['object_id'] for x in rows]
ck(H(new)=='c539e423db9d957f873dbb6b381728c6eb8840a6d5cbc1479cd388b3b6e61991','new-index');ck(len(ids)==len(set(ids))==36,'36-unique');ck(a['d2_commit']==b['d2_commit']=='dd78dcc5355fd0f41fb00541d6a18dc36d87ef91','d2')
for i,r in enumerate(rows):
 p=pathlib.Path(r['artifact']);ck(p.is_file() and H(p)==r['artifact_sha256'],'artifact-'+str(i));ck(r['shape']==[2,3,257,9],'shape-'+str(i));ck(r['base_state_sha256_before']==r['base_state_sha256_after'],'state-'+str(i));ck(r['split']=='unassigned','public-split-'+str(i));ck(pathlib.Path(r['checkpoint']['path']).name=='step-000024999.ckpt','step-'+str(i));d=torch.load(p,map_location='cpu');ck(list(d['features'].shape)==[2,3,257,9] and list(d['scalars'].shape)==[2,3,257] and torch.isfinite(d['features']).all() and torch.isfinite(d['scalars']).all(),'tensor-'+str(i))
forbidden={'target','targets','presentation','canonical_side','joint_limits','fractions','b_test','full22'}
def walk(v):
 if isinstance(v,dict):
  ck(not(set(map(str.lower,v))&forbidden),'denylist');[walk(x) for x in v.values()]
 elif isinstance(v,list):[walk(x) for x in v]
walk(a);walk(b)
inventory=json.loads(inv.read_text())['records']; eligible=[]
for r in inventory:
 j=r.get('primary_joint') or {}; ok=r.get('error') is None and r.get('visual_geometry_count',0)>0 and j.get('type') in ('revolute','prismatic') and j.get('zero_relation') in ('lower','upper') and j.get('upper',0)>j.get('lower',0)
 if ok: eligible.append(r)
rank=lambda r:hashlib.sha256(('node7.1-object-selection-v1|'+r['source']+'|'+r['object_id']).encode()).hexdigest(); first=sorted(eligible,key=rank)[:36]; ck(set(ids)=={r['object_id'] for r in first},'first36-prefix')
s=salt.read_bytes();ck(hashlib.sha256(s).hexdigest()=='7d94832908e12ec238b4abba232b140214e47cda6f1d2074c95a27f919c600c9','salt'); sr=sorted(first,key=lambda r:hashlib.sha256(s+b'|node7.2-split-rank-v1|'+r['source'].encode()+b'|'+r['object_id'].encode()).digest()); sets=[{r['object_id'] for r in sr[:18]},{r['object_id'] for r in sr[18:27]},{r['object_id'] for r in sr[27:]}];ck(list(map(len,sets))==[18,9,9] and len(set.union(*sets))==36 and not any(sets[i]&sets[j] for i in range(3) for j in range(i)),'split')
receipt={'schema':'splart-gate-a-public/v1','status':'AUTHORIZED_FOR_EVALUATOR' if not fail else 'FAIL','failures':fail,'profile_count':36,'profile_set_sha256':hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),'new24_index_sha256':H(new),'old12_index_sha256':H(old),'first36_prefix_verified':not fail,'split_counts':[18,9,9],'split_coverage_and_disjoint':not fail,'salt_sha256':hashlib.sha256(s).hexdigest(),'membership_emitted':False,'targets_emitted':False,'protected_splits_read':[],'source_commit':'f1f2bb9581e3465e6e320adf3868b015bcf15eab','source_tree':'c915fb1f2902540f5a1eaa68fc9e2dce213a478b','evaluator_started':False};out.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n');print(json.dumps(receipt))
