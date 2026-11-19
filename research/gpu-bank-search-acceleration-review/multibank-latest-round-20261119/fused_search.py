"""Fuse different motifs into large CUDA batches; retain score uncertainty.
Exact integer histograms decide every nonnegative/uncertain price pattern.
"""
import argparse,itertools,json,sys,time
from pathlib import Path
from collections import Counter
from decimal import Decimal,localcontext
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import multibank_search as s
from bank_graph import component_graph,price_tables,incidence,read,INVALID,UNIT
PAIRS=list(itertools.combinations(range(4),2))

def batch_score(cs,base,weights,olds,table,backend):
 if backend=='cuda':
  cs=torch.as_tensor(cs,device='cuda');value=torch.as_tensor(base,device='cuda');w=torch.as_tensor(weights,device='cuda');old=torch.as_tensor(olds,device='cuda')
  legal=(cs!=old).sum(1)>=3
  for j,(u,v) in enumerate(PAIRS):
   term=table[cs[:,u],cs[:,v]];legal&=(w[:,j]==0)|(term!=INVALID);value+=term*w[:,j]
  return torch.where(legal,value,INVALID).cpu().numpy()
 value=base.copy();legal=(cs!=olds).sum(1)>=3
 for j,(u,v) in enumerate(PAIRS):
  term=table[cs[:,u],cs[:,v]];legal&=(weights[:,j]==0)|(term!=INVALID);value+=term*weights[:,j]
 return np.where(legal,value,INVALID)

def exact_patterns(g,cluster,old,neighbors,cs,seen,kappa):
 lookup={u:i for i,u in enumerate(cluster)};dims=np.array(list(map(len,g['frames'])))
 deltas=np.zeros((len(cs),10),dtype=np.int16);ix=np.arange(len(cs))
 for u in cluster:
  for _,chart,v,_ in neighbors[u]:
   if v in lookup and u>v:continue
   before=abs(dims[old[u]]-dims[chart]);after=abs(dims[cs[:,lookup[u]]]-dims[cs[:,lookup[v]]] if v in lookup else dims[cs[:,lookup[u]]]-dims[chart])
   deltas[:,before]+=1;deltas[ix,after]-=1
 deltas[:,0]=0
 unique,first=np.unique(deltas,axis=0,return_index=True);positive=[]
 with localcontext() as ctx:
  ctx.prec=90;h=[Decimal(0)]+[Decimal(r)**(1-Decimal(kappa)) for r in range(1,10)]
  for delta,i in zip(unique,first):
   key=tuple(map(int,delta))
   if key not in seen:
    seen[key]=sum((Decimal(n)*h[r] for r,n in enumerate(key)),Decimal(0))
   if seen[key]>Decimal('1e-70'):positive.append((str(seen[key]),cluster,cs[i,:len(cluster)].tolist(),key))
 return positive,len(unique)

def main():
 p=argparse.ArgumentParser();p.add_argument('--graph',required=True,type=Path);p.add_argument('--layout',required=True,type=Path);p.add_argument('--kappa',required=True);p.add_argument('--out',required=True,type=Path);p.add_argument('--motifs',type=int,default=10000);p.add_argument('--options',type=int,default=24);p.add_argument('--batch',type=int,default=262144);p.add_argument('--backend',choices=['compare','cpu','cuda'],default='compare');a=p.parse_args()
 a.out.mkdir(exist_ok=False,parents=True);start=time.perf_counter();g,layout=read(a.graph),read(a.layout);groups,fixed,old,edges=component_graph(g,layout);nb=incidence(old,edges);tables,_=price_tables(g['frames'],a.kappa);table=tables[0];free=set(range(len(groups)))-fixed
 selected=[];census=Counter()
 for cluster in s.motifs(free,nb):
  info=s.prepare(cluster,old,nb,table,a.options);census[len(cluster)]+=1
  if info[-1]>=-info[4]:selected.append((info[-1],cluster,info))
 selected.sort(key=lambda x:(-x[0],x[1]));all_count=len(selected);selected=selected[:a.motifs]
 prep=time.perf_counter()-start;dt=torch.as_tensor(table,device='cuda') if a.backend!='cpu' else None;parts=[];pending=0;count=0;gpu=cpu=pack=exact=0.;uncertain=0;patterns={};positives=[]
 def flush():
  nonlocal parts,pending,count,gpu,cpu,pack,exact,uncertain,positives
  if not parts:return
  t=time.perf_counter();cs=np.concatenate([x[0] for x in parts]);base=np.concatenate([x[1] for x in parts]);weights=np.concatenate([x[2] for x in parts]);olds=np.concatenate([x[3] for x in parts]);pack+=time.perf_counter()-t
  if a.backend!='cpu':
   t=time.perf_counter();gs=batch_score(cs,base.copy(),weights,olds,dt,'cuda');gpu+=time.perf_counter()-t
  if a.backend!='cuda':
   t=time.perf_counter();ref=batch_score(cs,base,weights,olds,table,'cpu');cpu+=time.perf_counter()-t
   if a.backend=='cpu':gs=ref
   else:assert np.array_equal(gs,ref),'CPU/GPU full score mismatch'
  offset=0;t=time.perf_counter()
  for item in parts:
   n=len(item[0]);cluster,edgecount=item[4:];scores=gs[offset:offset+n];near=np.flatnonzero(scores>=-edgecount-2);uncertain+=len(near)
   if len(near):
    pos,_=exact_patterns(g,cluster,old,nb,item[0][near],patterns,a.kappa);positives.extend(pos)
   offset+=n
  exact+=time.perf_counter()-t;count+=len(cs);parts=[];pending=0
 for index,(_,cluster,info) in enumerate(selected):
  opts,ext,internal,baseline,edgecount,_=info;total=int(np.prod([len(o) for o in opts]))
  for begin in range(0,total,a.batch):
   t=time.perf_counter();end=min(total,begin+a.batch);z=np.arange(begin,end,dtype=np.int64);columns=[];indices=[]
   for o in reversed(opts):indices.append(z%len(o));columns.append(o[z%len(o)]);z//=len(o)
   cs=np.zeros((end-begin,4),dtype=np.int64);cs[:,:len(cluster)]=np.stack(columns[::-1],axis=1)
   base=sum(ext[j][indices[::-1][j]] for j in range(len(cluster)))-baseline
   weights=np.tile(np.array([internal.get(pair,0) for pair in PAIRS],dtype=np.int64),(len(cs),1));olds=np.zeros_like(cs);olds[:,:len(cluster)]=old[list(cluster)]
   parts.append((cs,base,weights,olds,cluster,edgecount));pending+=len(cs);pack+=time.perf_counter()-t
   if pending>=a.batch:flush()
  if (index+1)%1000==0:print('motifs',index+1,'scored',count,'positive patterns',len(positives),flush=True)
 flush()
 positives.sort(key=lambda x:-Decimal(x[0]));report={'baseline_graph':s.sha256(s.encode(g)).hexdigest(),'kappa':a.kappa,'total_connected_motifs':dict(census),'nonexcluded_motifs':all_count,'selected_motifs':len(selected),'options_limit':a.options,'assignments_scored':count,'uncertain_or_positive_assignments_checked_by_exact_histogram':uncertain,'unique_histograms':len(patterns),'backend':a.backend,'all_cpu_gpu_scores_match':True if a.backend=='compare' else 'validated in separate same-work compare run','common_preparation_s':prep,'common_packing_s':pack,'gpu_transfer_score_return_s':gpu,'same_work_cpu_score_s':cpu,'exact_shortlisting_s':exact,'observed_total_s':time.perf_counter()-start,'cpu_time_is_audit_not_production_gpu_time':True,'positive_examples':positives[:30],'scope':'Subset of motifs and chart options, not exhaustive. All score-near-zero cases eligible for exact integer histogram pricing. Full application not run.'}
 (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
