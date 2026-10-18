"""Recompute controlled v2 ranking metrics from hits and frozen evidence IDs."""
from pathlib import Path
import json,collections,math
R=Path(__file__).resolve().parent
q={v['sample_id']:v for v in map(json.loads,(R/'evaluation/controlled_scenario_v2/questions-test.jsonl').read_text(encoding='utf-8').splitlines())}
groups=collections.defaultdict(list);pairs=set()
for path in [R/'results/controlled-path-fusion-v2/test-predictions.jsonl',R/'reports/gated-triple-paper-2026-10-16/supplement-predictions.jsonl']:
 for line in path.read_text(encoding='utf-8').splitlines():
  v=json.loads(line);pair=(v['method'],v['sample_id']);assert pair not in pairs; pairs.add(pair)
  gold=set(q[v['sample_id']]['relevant_chunk_ids']);hits=v['hits'];assert len(hits)==len(set(hits));scores={}
  for k in [1,3,5,10]:
   n=len(gold.intersection(hits[:k]));dcg=sum(1/math.log2(i+2) for i,h in enumerate(hits[:k]) if h in gold);ideal=sum(1/math.log2(i+2) for i in range(min(len(gold),k)))
   scores[str(k)]={'complete':int(gold.issubset(hits[:k])),'recall':n/len(gold),'precision':n/k,'ndcg':dcg/ideal if ideal else 0}
   for key,x in scores[str(k)].items():assert abs(v['metrics']['at'][str(k)][key]-x)<1e-9,(pair,k,key)
  scores['mrr']=next((1/(i+1) for i,h in enumerate(hits[:10]) if h in gold),0)
  scores['complete_mrr']=next((1/k for k in range(1,11) if gold.issubset(hits[:k])),0)
  for key in ['mrr','complete_mrr']:assert abs(scores[key]-v['metrics'][key])<1e-9,(pair,key)
  for kind in ['ALL',v['kind']]:groups[v['method'],kind].append(scores)
print('method,type,n,complete_at_5,recall_at_5,ndcg_at_5,mrr')
for (method,kind),vs in sorted(groups.items()):
 print(','.join([method,kind,str(len(vs)),str(sum(v['5']['complete'] for v in vs))]+[f'{sum(v["5"][k] for v in vs)/len(vs):.8f}' for k in ['recall','ndcg']]+[f'{sum(v["mrr"] for v in vs)/len(vs):.8f}']))
assert len(pairs)==4104,len(pairs)
print('Verified 4104 paired ranking records,19 configurations; zero API calls.')
