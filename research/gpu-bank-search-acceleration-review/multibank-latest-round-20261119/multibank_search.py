"""Coupled 3/4-bank GPU search; local exact replay on shortlisted true changes."""
import argparse,copy,gzip,importlib.util,itertools,json,sys,time
from collections import Counter,defaultdict
from decimal import Decimal,localcontext
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bank_graph import component_graph,price_tables,incidence,read,INVALID,UNIT

def encode(g):return json.dumps(g,sort_keys=True,separators=(',',':')).encode()
def motifs(free,neighbors):
 adj={u:{v for _,_,v,_ in neighbors[u] if v in free} for u in free}
 triples=set()
 for u in free:
  for v,w in itertools.combinations(sorted(adj[u]),2):triples.add(tuple(sorted((u,v,w))))
 yield from sorted(triples)
 quads=set()
 for c in sorted(triples):
  for v in set().union(*(adj[u] for u in c))-set(c):quads.add(tuple(sorted((*c,v))))
 yield from sorted(quads)

def prepare(cluster,old,neighbors,table,option_limit):
 local={u:i for i,u in enumerate(cluster)};opts=[];exterior=[];internal=Counter()
 baseline=0;edgecount=0
 for u in cluster:
  terms=[]
  for kind,chart,v,key in neighbors[u]:
   assert kind==0 # all public endpoints fixed
   if v in local:
    if u<v:internal[local[u],local[v]]+=1;baseline+=int(table[old[u],chart]);edgecount+=1
   else:terms.append(table[:,chart]);baseline+=int(table[old[u],chart]);edgecount+=1
  if terms:
   terms=np.asarray(terms);valid=np.all(terms!=INVALID,axis=0);scores=terms.sum(0)
  else:valid=np.ones(table.shape[0],dtype=bool);scores=np.zeros(table.shape[0],dtype=np.int64)
  ids=np.flatnonzero(valid)
  ids=ids[np.argsort(-scores[ids],kind='stable')]
  if option_limit and len(ids)>option_limit:
   retained=list(ids[:option_limit]);special=[int(old[u])]+[int(old[v]) for v in cluster]
   for v in special:
    if valid[v] and v not in retained:retained.append(v)
   ids=np.array(retained)
  opts.append(ids);exterior.append(scores[ids])
 upper=sum(int(v.max()) for v in exterior)-baseline
 return opts,exterior,internal,baseline,edgecount,upper

def assignments(opts,start,end):
 indices=np.arange(start,end,dtype=np.int64);cols=[]
 for o in reversed(opts):cols.append(o[indices%len(o)]);indices//=len(o)
 return np.stack(cols[::-1],axis=1)

def scores_cpu(cs,cluster,old,neighbors,table):
 lookup={u:i for i,u in enumerate(cluster)};score=np.zeros(len(cs),dtype=np.int64);valid=np.ones(len(cs),dtype=bool)
 for u in cluster:
  for _,chart,v,_ in neighbors[u]:
   if v in lookup and u>v:continue
   q=table[cs[:,lookup[u]],cs[:,lookup[v]] if v in lookup else chart]
   valid&=q!=INVALID;score+=q-int(table[old[u],chart])
 valid&=(cs!=old[list(cluster)]).sum(1)>=3
 return np.where(valid,score,INVALID)

def scores_gpu(cs,cluster,old,neighbors,table):
 lookup={u:i for i,u in enumerate(cluster)};ts=torch.as_tensor(cs,device='cuda');score=torch.zeros(len(cs),dtype=torch.int64,device='cuda');valid=torch.ones(len(cs),dtype=torch.bool,device='cuda')
 for u in cluster:
  for _,chart,v,_ in neighbors[u]:
   if v in lookup and u>v:continue
   q=table[ts[:,lookup[u]],ts[:,lookup[v]] if v in lookup else chart]
   valid&=q!=INVALID;score+=q-table[int(old[u]),chart]
 valid&=(ts!=torch.as_tensor(old[list(cluster)],device='cuda')).sum(1)>=3
 return torch.where(valid,score,INVALID).cpu().numpy()

def synthesize(g,groups,cluster,charts):
 move={k:[a,b] for k,a,b in g['moves']};forms={k:{i:Fraction(q) for i,q in r} for k,r in g['forms']}
 for u,chart in zip(cluster,charts):
  for side,k in groups[u]:
   assert k in move,'Public boundary attempted'
   move[k][0 if side=='form' else 1]=int(chart)
 collapsed=[]
 for k in g['schedule']:
  if k not in move or move[k][0]!=move[k][1]:continue
  replacement=forms[k]
  for j,row in forms.items():
   if k in row:
    q=row.pop(k)
    for i,v in replacement.items():
     row[i]=row.get(i,0)+q*v
     if not row[i]:del row[i]
  del forms[k];del move[k];collapsed.append(k)
 out=dict(g,moves=[[k,*v] for k,v in sorted(move.items())],forms=[[k,[[i,str(q)] for i,q in sorted(r.items())]] for k,r in sorted(forms.items())],schedule=[k for k in g['schedule'] if k in forms])
 for field in ('initial','final','frames','first_output'):assert out[field]==g[field]
 return out,collapsed

def exact_credit(g,child,kappa):
 def hist(x):return Counter(abs(len(x['frames'][a])-len(x['frames'][b])) for _,a,b in x['moves'])
 a,b=hist(g),hist(child);delta={r:a[r]-b[r] for r in a.keys()|b.keys() if a[r]!=b[r]}
 with localcontext() as ctx:
  ctx.prec=80;net=sum((Decimal(n)*(Decimal(r)**(1-Decimal(kappa))) for r,n in delta.items()),Decimal(0))
 return delta,net

def main():
 p=argparse.ArgumentParser();p.add_argument('--graph',required=True,type=Path);p.add_argument('--layout',required=True,type=Path);p.add_argument('--reader',required=True,type=Path);p.add_argument('--kappa',required=True);p.add_argument('--out',required=True,type=Path);p.add_argument('--motifs',type=int,default=1500);p.add_argument('--options',type=int,default=24);p.add_argument('--batch',type=int,default=131072);p.add_argument('--shortlist',type=int,default=12);args=p.parse_args()
 args.out.mkdir(exist_ok=False,parents=True);start=time.perf_counter();g,layout=read(args.graph),read(args.layout)
 groups,fixed,old,edges=component_graph(g,layout);neighbors=incidence(old,edges);tables,_=price_tables(g['frames'],args.kappa);table=tables[0];free=set(range(len(groups)))-fixed
 prepared=[];census=Counter()
 for cluster in motifs(free,neighbors):
  info=prepare(cluster,old,neighbors,table,args.options);census[len(cluster)]+=1
  if info[-1]>0:prepared.append((info[-1],cluster,info))
 prepared.sort(key=lambda t:(-t[0],t[1]));selected=prepared[:args.motifs]
 print('Prepared',dict(census),'positive upper bounds',len(prepared),'selected',len(selected),flush=True)
 dt=torch.as_tensor(table,device='cuda');proposals=[];count=0;matched=False;bench={};gpu_seconds=0;cpu_seconds=0;host_prepare_seconds=time.perf_counter()-start;completed=0
 for _,cluster,info in selected:
  opts=info[0];total=int(np.prod([len(x) for x in opts]));edgecount=info[4]
  for begin in range(0,total,args.batch):
   cs=assignments(opts,begin,min(begin+args.batch,total));t=time.perf_counter();scores=scores_gpu(cs,cluster,old,neighbors,dt);gpu_seconds+=time.perf_counter()-t;count+=len(cs)
   t=time.perf_counter();cpu=scores_cpu(cs,cluster,old,neighbors,table);cpu_seconds+=time.perf_counter()-t;assert np.array_equal(cpu,scores)
   if not matched:
    bench={'cpu_s':time.perf_counter()-t,'rows':len(cs)};assert np.array_equal(cpu,scores);matched=True
   ix=np.flatnonzero(scores>edgecount+2)
   if len(ix):
    ix=ix[np.argsort(-scores[ix])[:args.shortlist]]
    for j in ix:proposals.append((int(scores[j]),cluster,cs[j].tolist()))
  completed+=1
  if completed%50==0:print('Progress',completed,'/',len(selected),'assignments',count,'positive',len(proposals),flush=True)
  if len(proposals)>args.shortlist*4:proposals=sorted(proposals,reverse=True)[:args.shortlist*2]
 proposals=sorted(proposals,reverse=True)[:args.shortlist]
 report={'baseline_graph':sha256(encode(g)).hexdigest(),'kappa':args.kappa,'clusters':dict(census),'positive_upper_bound_clusters':len(prepared),'selected_motifs':len(selected),'options_limit':args.options,'proposals_tested':count,'cpu_gpu_sample':bench,'gpu_seconds':gpu_seconds,'cpu_same_assignments_seconds':cpu_seconds,'common_host_preparation_seconds':host_prepare_seconds,'all_cpu_gpu_scores_match':True,'positive_shortlist':len(proposals),'candidates':[],'scope':'All public boundaries fixed, at least three components must change. Not exhaustive outside selected motifs/options.'}
 spec=importlib.util.spec_from_file_location('official_reader',args.reader);reader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reader;spec.loader.exec_module(reader)
 for j,(score,cluster,charts) in enumerate(proposals):
  r={'cluster':cluster,'charts':charts,'screen':score};report['candidates'].append(r)
  try:
   child,collapsed=synthesize(g,groups,cluster,charts);delta,credit=exact_credit(g,child,args.kappa);r.update(paid_histogram_credit=delta,paid_credit=str(credit),collapsed=collapsed)
   assert credit>Decimal('1e-50'),'No robust exact-price credit'
   graph=reader.Graph.parse(child);expected,_=reader.public_operator(graph,layout,[reader.span(f) for f in graph.frames]);forward,_=reader.forward(graph);adjoint,_=reader.adjoint(graph)
   assert forward==expected and adjoint==reader.transpose(expected),'Full rational endpoint mismatch'
   path=args.out/f'candidate-{j}.json.gz';path.write_bytes(gzip.compress(encode(child),mtime=0));r.update(status='EXACT_OPERATOR_PASS_FULL_APPLICATION_PENDING',path=str(path),graph_sha256=sha256(encode(child)).hexdigest())
  except Exception as e:r.update(status='REJECTED',error=str(e))
  (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 report['elapsed_s']=time.perf_counter()-start;(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
