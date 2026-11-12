"""Read-only paired audit of archived retrieval rankings; no model calls."""
from pathlib import Path
import collections,hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]/'.runtime/github-research-publish/research/knowledge-enhancement-2026-10-18'
def load(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
def ratio(a,b):return a/b if b else None
def compare(rows,baseline,target,k):
 pairs=collections.defaultdict(dict)
 for r in rows:pairs[r['sample_id']][r['method']]=r
 groups=collections.defaultdict(list)
 for q,rs in pairs.items():
  if baseline not in rs or target not in rs:continue
  a,b=rs[baseline],rs[target]
  old=bool(a['metrics']['at'][str(k)]['complete']);new=bool(b['metrics']['at'][str(k)]['complete'])
  for kind in ['ALL',a['kind']]:groups[kind].append((q,old,new))
 out={}
 for kind,vs in groups.items():
  improved=[q for q,a,b in vs if not a and b];damaged=[q for q,a,b in vs if a and not b]
  old=sum(a for _,a,_ in vs);new=sum(b for _,_,b in vs)
  out[kind]={'n':len(vs),'baseline_complete':old,'target_complete':new,'baseline_incomplete':len(vs)-old,'improved':len(improved),'damaged':len(damaged),'tied':len(vs)-len(improved)-len(damaged),'completion_rate':ratio(len(improved),len(vs)-old),'damage_rate':ratio(len(damaged),old),'net_gain':len(improved)-len(damaged),'improved_ids':improved,'damaged_ids':damaged}
 return out
if __name__=='__main__':
 paths=[ROOT/'results/controlled-path-fusion-v2/test-predictions.jsonl',ROOT/'reports/gated-triple-paper-2026-10-16/supplement-predictions.jsonl']
 rows=sum((load(p) for p in paths),[])
 methods=sorted({r['method'] for r in rows});print(methods)
 result={'date':'2026-11-12','scope':'historical controlled data, post-selection descriptive audit, not new independent evidence','inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'methods':methods,'comparisons':{}}
 for method in methods:
  if method=='BD_scope':continue
  for k in [3,5,10]:result['comparisons'][f'BD_scope -> {method} @ {k}']=compare(rows,'BD_scope',method,k)
 (HERE/'paired-evidence-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print('Wrote paired audit:',len(rows),'rankings')

