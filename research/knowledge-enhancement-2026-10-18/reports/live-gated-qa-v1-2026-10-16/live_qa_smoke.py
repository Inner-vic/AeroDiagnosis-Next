"""Engineering smoke checks on actual live sources; not a blinded quality benchmark."""
import json,urllib.request,urllib.error,sqlite3,time,hashlib
from pathlib import Path
R=Path('D:/codex/worktrees/knowledge-upgrade-integration/AeroDiagnosis')
O=R/'.runtime/qa-integration-v1';O.mkdir(exist_ok=True)
def budget():
 with sqlite3.connect(R/'.runtime/thesis-research-provider-v1/calls.sqlite3') as db:
  rows=db.execute('SELECT state,reservation_micro,cost_micro FROM calls').fetchall()
 return {'calls':len(rows),'upper_cny':sum(a if s in ('pending','unknown_charge') else b for s,a,b in rows)/1e6,'unresolved':sum(s in ('pending','unknown_charge') for s,a,b in rows)}
def request(path,body=None):
 r=urllib.request.Request('http://127.0.0.1:8082'+path,data=json.dumps(body,ensure_ascii=False).encode() if body is not None else None,headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(r,timeout=110))
cases=[{'name':'source_fact','body':{'question':'压气机叶片损坏可能由什么引起？','knowledge_context':{'document_id':'prebuilt-compressor-cleaning'}},'expected':'supported_fact_with_verifiable_citations'},
 {'name':'uncertainty_boundary','body':{'question':'仅凭排气温度升高，是否一定能确定压气机叶片损坏？','knowledge_context':{'document_id':'prebuilt-compressor-cleaning'}},'expected':'no_certain_root_cause_from_temperature_only'},
 {'name':'unknown_source','body':{'question':'QB-99999999的机理路径是什么？'},'expected':'no_evidence_no_model_calls'},
 {'name':'document_scope_empty','body':{'question':'压气机效率降低的原因？','knowledge_context':{'document_id':'qa-smoke-nonexistent'}},'expected':'no_evidence_no_model_calls'}]
(O/'live-smoke-protocol.json').write_text(json.dumps({'kind':'engineering acceptance, not independently labeled answer-quality experiment','cases':cases,'before_dispatch_budget':budget(),'no_result_conditioned_question_rewrites':True},ensure_ascii=False,indent=2),encoding='utf-8')
results=[]
for case in cases:
 if case['name'] in ('source_fact','uncertainty_boundary'):
  preflight=request('/api/qa/answer',{**case['body'],'evidence_only':True})
  assert preflight['evidence']
  assert all(e['document_id']=='prebuilt-compressor-cleaning' and e['source_url']=='https://www.faa.gov/sites/faa.gov/files/12_amtp_ch10_0.pdf' for e in preflight['evidence'])
 start=time.perf_counter();r=request('/api/qa/answer',case['body']);checks=[]
 for claim in r['claims']:
  assert claim['citations']
  for citation in claim['citations']:
   source=request('/api/qa/evidence/'+citation['chunk_id'])
   assert source['text_sha256']==citation['text_sha256']
   assert source['text'][citation['start']:citation['end']]==citation['support_quote']
   checks.append('source_span_hash_and_offset')
 if case['name'] in ('unknown_source','document_scope_empty'):
  assert not r['evidence'] and r['insufficient_evidence'] and r['generation']['model_calls']==0
 if case['name']=='source_fact':assert r['claims'] and not r['insufficient_evidence']
 if case['name']=='uncertainty_boundary':
  assert not r['retrieval']['graph_enabled']
  # Preserve actual output for human semantic review; do not equate citation validity with entailment.
 results.append({'name':case['name'],'seconds':round(time.perf_counter()-start,3),'citation_checks':len(checks),'response':r})
 (O/'live-smoke-results.json').write_text(json.dumps({'results':results,'budget_after':budget()},ensure_ascii=False,indent=2),encoding='utf-8')
 print(case['name'],r['status'],'claims',len(r['claims']),'citations',len(checks),flush=True)
print('Budget',json.dumps(budget()),flush=True)
