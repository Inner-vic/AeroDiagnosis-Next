"""Standard-library-only published hash and paired completeness recomputation."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
for r in json.loads((P/'MANIFEST.json').read_text(encoding='utf8'))['files']:
 f=P/r['path'];assert len(f.read_bytes())==r['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'],r['path']
Q={q['sample_id']:q for q in json.loads((P/'pilot/questions-public.json').read_text(encoding='utf8'))}
rows=[json.loads(x) for x in (P/'pilot/rankings-public.jsonl').read_text(encoding='utf8').splitlines()]
assert len(rows)==288 and len({(r['scope'],r['method'],r['sample_id']) for r in rows})==288
summary=json.loads((P/'pilot/summary.json').read_text(encoding='utf8'))
for scope in ['document','all']:
 for method in sorted({r['method'] for r in rows}):
  pairs={r['sample_id']:r for r in rows if r['scope']==scope and r['method']==method}
  base={r['sample_id']:r for r in rows if r['scope']==scope and r['method']=='BD'}
  for kind in ['ALL','fact','paraphrase','chain','chain_paraphrase']:
   qs=[q for q in Q.values() if q['answerable'] and (kind=='ALL' or q['kind']==kind)]
   complete=lambda q,r:any(set(s)<=set(r['hits'][:5]) for s in q['acceptable_support_sets'])
   a={q['sample_id'] for q in qs if complete(q,base[q['sample_id']])};b={q['sample_id'] for q in qs if complete(q,pairs[q['sample_id']])}
   v=summary[scope][method+':'+kind]
   assert (v['n'],v['baseline_complete'],v['complete'],v['gain'],v['loss'],v['ties'])==(len(qs),len(a),len(b),len(b-a),len(a-b),len(qs)-len(a^b))
   assert v['ECR']==(len(b-a)/(len(qs)-len(a)) if len(qs)>len(a) else None)
   assert v['EDR']==(len(a-b)/len(a) if a else None)
print('Verified published file hashes, 288 rankings, and all paired completeness/ECR/EDR summaries; zero API calls.')
