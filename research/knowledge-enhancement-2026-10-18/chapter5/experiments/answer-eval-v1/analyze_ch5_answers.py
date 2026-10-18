from pathlib import Path
import json,hashlib,shutil,statistics,sys,math
from collections import Counter
from tokenizers import Tokenizer
R=Path('D:/codex/worktrees/knowledge-upgrade-integration/AeroDiagnosis');P=R/'evaluation/chapter5_answer_pilot_v1';O=R/'.runtime/chapter5-answer-eval-v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
review_path=Path('D:/zhaoweihao3/Desktop/Project/AeroDiagnosis/.runtime/ch5-answer-review/answer-review.json');review=read(review_path)
assert review['expert_reviewed'] is False and review['method_identities_seen'] is False
for item in review['read_file_sha256']:
 path=Path(item['path']);assert sha(path if path.is_absolute() else R/path)==item['sha256']
shutil.copyfile(review_path,O/'independent-answer-review.json')
pred=[json.loads(x) for x in (O/'predictions.jsonl').read_text(encoding='utf-8').splitlines()];assert len(pred)==48 and len({(r['id'],r['method']) for r in pred})==48
mapping=read(O/'blind-identity-map.json');scores={}
qs=read(P/'questions.json')+read(P/'controlled-questions.json');qm={q['id']:q for q in qs}
for row in review['rows']:
 ident=mapping[row['submission_id']];q=qm[ident['id']]
 assert len(row['covered_points'])==len(q['required_points'])
 assert all(type(v) is bool for v in row['covered_points']+row['claim_support'])
 scores[ident['id'],ident['method']]=row
assert len(scores)==48
counter=Tokenizer.from_file(str(R/'.runtime/models/multilingual-e5-small/tokenizer.json'));counter.no_truncation();counter.no_padding()
audited=[];total_refs=0;total_claims=0
for r in pred:
 q=qm[r['id']];s=scores[r['id'],r['method']];response=r.get('response',{})
 assert r['completed'] and len(s['claim_support'])==len(response.get('claims',[]))
 if q['answerable']:
  expected=all(s['covered_points']) and all(s['claim_support']) and s['certainty_and_scope_preserved'] and response['status']=='evidence_ready'
 else:expected=s['safe_abstention'] is True and response['insufficient_evidence'] and all(s['claim_support']) and s['certainty_and_scope_preserved']
 assert bool(expected)==s['complete_correct']
 evidence={e['id']:e for e in response['evidence']}
 assert len(evidence)==len(response['evidence']) and len(evidence)<=10
 assert max(Counter(e['document_id'] for e in evidence.values()).values(),default=0)<=5
 assert sum(len(counter.encode(e['text']).ids) for e in evidence.values())<=4096
 assert response['generation'].get('context_tokens',0)<=4096
 for e in evidence.values():
  assert e['document_id']==q['document_id']
  assert hashlib.sha256(e['text'].encode()).hexdigest()==e['text_sha256']
 for claim in response['claims']:
  assert claim['citations'];total_claims+=1
  for cite in claim['citations']:
   e=evidence[cite['evidence_id']];assert cite['chunk_id']==e['id']
   assert cite['text_sha256']==e['text_sha256']
   assert e['text'][cite['start']:cite['end']]==cite['support_quote'];total_refs+=1
 for path in response['paths']:
  assert set(path['source_chunks'])<=set(evidence)
  assert path['provided_to_model']==(set(path['source_chunks'])<=set(response['generation'].get('context_chunk_ids',[])))
 refs=[x for x in r['model_receipts'] if not x.get('cache_hit')]
 rr=response['retrieval'];rel={x['chunk_id'] for x in q['reference_support']};hits=list(evidence)
 audited.append({**r,'review':s,'reference_anchor_complete5':int(rel<=set(hits[:5])),'new_provider_calls':len(refs),'cached_calls':len(r['model_receipts'])-len(refs),'new_cost_cny':sum(x['estimated_peak_cost_cny'] for x in refs),'citation_count':sum(len(c['citations']) for c in response['claims']),'claim_count':len(response['claims']),'supported_claim_count':sum(s['claim_support'])})
for q in qs:
 aa=[r for r in audited if r['id']==q['id']]
 left,right=aa
 core=lambda r:(r['response']['status'],r['response']['answer'],r['response']['claims'],r['response']['reason'])
 if core(left)==core(right):
  for key in ['covered_points','claim_support','certainty_and_scope_preserved','safe_abstention','complete_correct']:assert left['review'][key]==right['review'][key],(q['id'],key)
def p95(values):
 a=sorted(values);pos=(len(a)-1)*.95;lo=math.floor(pos);hi=math.ceil(pos);return a[lo]+(a[hi]-a[lo])*(pos-lo)
summary={};paired={}
for group in ['real','controlled']:
 summary[group]={}
 for method in ['BD','GATED']:
  rows=[r for r in audited if r['group']==group and r['method']==method];ans=[r for r in rows if qm[r['id']]['answerable']];unans=[r for r in rows if not qm[r['id']]['answerable']]
  nclaims=sum(r['claim_count'] for r in rows);supported=sum(r['supported_claim_count'] for r in rows)
  summary[group][method]={'n':len(rows),'completed':sum(r['completed'] for r in rows),'decision_correct':sum(r['review']['complete_correct'] for r in rows),'answerable_n':len(ans),'answerable_complete':sum(r['review']['complete_correct'] for r in ans),'unanswerable_n':len(unans),'safe_abstentions':sum(r['review']['safe_abstention'] is True for r in unans),'false_abstentions':sum(r['response']['insufficient_evidence'] for r in ans),'claim_count':nclaims,'supported_claims':supported,'retained_claim_support_rate':supported/nclaims if nclaims else None,'citation_count':sum(r['citation_count'] for r in rows),'citation_identity_offset_hash_valid':sum(r['citation_count'] for r in rows),'gate_open':sum(r['response']['retrieval'].get('graph_enabled',False) for r in rows),'graph_error_cases':sum(bool(r['response']['retrieval'].get('graph_errors')) for r in rows),'median_wall_seconds_including_cache':statistics.median(r['elapsed_seconds'] for r in rows),'p95_wall_seconds_including_cache':p95([r['elapsed_seconds'] for r in rows]),'median_retrieval_ms':statistics.median(r['response']['retrieval']['elapsed_ms'] for r in rows),'new_provider_calls':sum(r['new_provider_calls'] for r in rows),'cached_calls':sum(r['cached_calls'] for r in rows),'new_cost_cny':sum(r['new_cost_cny'] for r in rows),'reference_anchor_complete5':sum(r['reference_anchor_complete5'] for r in rows),'declared_insufficient':sum(r['response']['insufficient_evidence'] for r in rows)}
 byid={(r['id'],r['method']):r for r in audited if r['group']==group};c=Counter()
 for q in [q for q in qs if ('cq-' if group=='real' else 'sq-') in q['id']]:
  a=byid[q['id'],'GATED']['review']['complete_correct'];b=byid[q['id'],'BD']['review']['complete_correct'];c['win' if a>b else 'loss' if a<b else 'tie']+=1
 paired[group]=dict(c)
save(O/'summary.json',summary);save(O/'paired-descriptive.json',paired)
with (O/'scored-predictions.jsonl').open('w',encoding='utf-8') as f:
 for r in audited:f.write(json.dumps(r,ensure_ascii=False)+'\n')
frozen=read(O/'execution-freeze.json');assert all(sha(Path(p))==h for p,h in frozen['hashes'].items())
for rel in ['.runtime/controlled-path-fusion-v2/test-freeze.json','.runtime/controlled-path-fusion-v2/test-candidate-freeze.json']:
 ff=read(R/rel);assert all(sha(Path(p) if Path(p).is_absolute() else R/p)==h for p,h in ff['hashes'].items());assert sha(R/'.runtime/controlled-path-fusion-v2/index.sqlite')==ff['index_sha256']
receipt=read(O/'execution-receipt.json');before=frozen['budget_before'];after=receipt['budget_after']
assert after['upper_cny']<=42 and after['unresolved_calls']==before['unresolved_calls']==15
assert sum(r['new_provider_calls'] for r in audited)==after['provider_calls']-before['provider_calls']
assert abs(sum(r['new_cost_cny'] for r in audited)-(after['upper_cny']-before['upper_cny']))<1e-6
audit={'questions':24,'paired_responses':48,'real_n':12,'controlled_n':12,'all_completed':True,'all_reference_and_model_context_budgets_passed':True,'retained_claims':total_claims,'citations_verified':total_refs,'frozen_code_questions_sources_unchanged':True,'controlled_index_and_previous_predictions_unchanged':True,'identical_response_scores_consistent':True,'independent_nonexpert_method_blind_review':True,'expert_reviewed':False,'paired':paired,'budget_before':before,'budget_after':after,'new_cost_cny':round(after['upper_cny']-before['upper_cny'],6),'remaining_cny':round(42-after['upper_cny'],6),'all_cached_calls':sum(r['cached_calls'] for r in audited),'no_new_unresolved_calls':True}
save(O/'completion-audit.json',audit)
print(json.dumps({'summary':summary,'paired':paired,'audit':audit},ensure_ascii=False,indent=2))
