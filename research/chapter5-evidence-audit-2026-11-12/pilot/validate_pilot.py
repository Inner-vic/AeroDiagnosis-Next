"""Independent recomputation from the frozen support sets and returned IDs."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_text(encoding='utf8'))
f=read('freeze.json');assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in f['files'].items())
qs={q['sample_id']:q for q in read('questions.json')};chunks={c['chunk_id']:c for c in read('corpus.json')['chunks']}
rows=[json.loads(x) for x in (P/'predictions.jsonl').read_text(encoding='utf8').splitlines()]
assert len(rows)==288 and len({(r['scope'],r['method'],r['sample_id']) for r in rows})==288
for r in rows:
 assert len(r['hits'])==len(set(r['hits'])) and set(r['hits'])<=chunks.keys()
 if r['scope']=='document':assert all(chunks[h]['document_id']==r['document_id'] for h in r['hits'])
 for k in [3,5,10]:
  gold=qs[r['sample_id']]['acceptable_support_sets'];m=r['metrics'][str(k)]
  if not gold:assert m is None;continue
  assert m['complete']==int(any(set(s)<=set(r['hits'][:k]) for s in gold))
 for path in r['paths']:assert set(path['source_chunks'])<=set(r['hits']) and path['complete_in_context']
for edge in read('graph.json')['edges']:
 proof=edge['properties'];c=chunks[proof['chunk_id']]
 assert c['content'][proof['canonical_start']:proof['canonical_end']]==proof['evidence_text']
 assert hashlib.sha256(c['content'].encode()).hexdigest()==proof['canonical_sha256']
summary=read('summary.json')
for scope in ['document','all']:
 for method in read('protocol.json')['methods']:
  pairs={r['sample_id']:r for r in rows if r['scope']==scope and r['method']==method}
  base={r['sample_id']:r for r in rows if r['scope']==scope and r['method']=='BD'}
  for kind in ['ALL','fact','paraphrase','chain','chain_paraphrase']:
   ids=[k for k,q in qs.items() if q['answerable'] and (kind=='ALL' or q['kind']==kind)]
   a={k for k in ids if base[k]['metrics']['5']['complete']};b={k for k in ids if pairs[k]['metrics']['5']['complete']};v=summary[scope][method+':'+kind]
   assert (v['n'],v['baseline_complete'],v['complete'],v['gain'],v['loss'],v['ties'])==(len(ids),len(a),len(b),len(b-a),len(a-b),len(ids)-len(a^b))
   assert v['ECR']==(len(b-a)/(len(ids)-len(a)) if len(ids)>len(a) else None)
   assert v['EDR']==(len(a-b)/len(a) if a else None)
print('Verified 288 rankings, all support sets, paired denominators, paths, six edge proofs and frozen hashes; no API calls.')
