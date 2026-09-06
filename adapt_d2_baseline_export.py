import argparse,json,hashlib,math,pathlib
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 q=argparse.ArgumentParser();q.add_argument('--index',type=pathlib.Path,required=True);q.add_argument('--output',type=pathlib.Path,required=True);a=q.parse_args();d=json.loads(a.index.read_text());root=a.index.parent; names={'full':'full_d2','single-radius':'single_radius','no-contact':'no_contact','no-penetration':'no_penetration','no-terminal-support':'no_terminal_support'};methods={}
 rows=[]
 for r in d['rows']:
  p=root/r['artifact'];assert H(p)==r['artifact_sha256'];rows.append(json.loads(p.read_text()))
 ids={x['object_id'] for x in rows};assert len(ids)==36
 prov={'source_commit':d['d2_source']['commit'],'source_tree':d['d2_source']['tree'],'config_sha256':d['input_sha256'],'prediction_tree_sha256':H(a.index)}
 for src,dst in names.items():methods[dst]={'provenance':prov,'rows':[{'object_id':x['object_id'],'distances':[-float(x['modes'][src]['lower_scalar']),float(x['modes'][src]['upper_scalar'])-1]} for x in rows]}
 for name,val in [('scratch',0.),('symmetric_linear',.5)]:methods[name]={'provenance':{'source_commit':'public-fixed-definition','source_tree':'public-fixed-definition','config_sha256':hashlib.sha256((name+str(val)).encode()).hexdigest(),'prediction_tree_sha256':H(a.index)},'rows':[{'object_id':x,'distances':[val,val]} for x in sorted(ids)]}
 for p in methods.values():
  assert all(all(math.isfinite(v) and v>=0 for v in x['distances']) for x in p['rows'])
 a.output.write_text(json.dumps({'schema':'splart-node22-target-free-baseline-predictions/v2','methods':methods},sort_keys=True,indent=2)+'\n')
if __name__=='__main__':main()
