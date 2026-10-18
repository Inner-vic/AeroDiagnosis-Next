from pathlib import Path
import json,hashlib,random
R=Path('D:/codex/worktrees/knowledge-upgrade-integration/AeroDiagnosis');P=R/'evaluation/chapter5_answer_pilot_v1';O=R/'.runtime/chapter5-answer-eval-v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
rows=[json.loads(x) for x in (O/'predictions.jsonl').read_text(encoding='utf-8').splitlines()];assert len(rows)==48
questions=read(P/'questions.json')+read(P/'controlled-questions.json');qm={q['id']:q for q in questions}
pack=[];mapping={};registry={}
for r in rows:
 token=hashlib.sha256(('blind20261017'+r['id']+r['method']).encode()).hexdigest()[:14];mapping[token]={'id':r['id'],'method':r['method'],'group':r['group']}
 response=r.get('response',{})
 for e in response.get('evidence',[]):registry[e['id']]={'id':e['id'],'document_id':e['document_id'],'version_id':e['version_id'],'text':e['text'],'text_sha256':e['text_sha256']}
 pack.append({'submission_id':token,'question_id':r['id'],'group':r['group'],'question':qm[r['id']]['question'],'answerable':qm[r['id']]['answerable'],'required_points':qm[r['id']]['required_points'],'reference_support':qm[r['id']]['reference_support'],'completed':r['completed'],'status':response.get('status'),'insufficient_evidence':response.get('insufficient_evidence'),'answer':response.get('answer',''),'reason':response.get('reason',''),'claims':response.get('claims',[])})
for sourcefile in ['sources.json','controlled-sources.json']:
 for doc in read(P/sourcefile).values():
  for c in doc['chunks']:
   if any(c['chunk_id']==p['chunk_id'] for q in questions for p in q['reference_support']):registry.setdefault(c['chunk_id'],{'id':c['chunk_id'],**c})
random.Random(20261017).shuffle(pack)
save(O/'blind-submissions.json',pack);save(O/'blind-source-registry.json',registry);save(O/'blind-identity-map.json',mapping)
save(O/'blind-rubric.json',{'reviewer':'Independent AI nonexpert; method identities hidden','primary':'complete_correct: for answerable, every required point is correctly covered AND no unsupported factual claim or certainty/scope error AND status evidence_ready; for unanswerable explicit insufficiency with no fabricated requested threshold and safe source-bound partial facts','per_submission_fields':['submission_id','covered_points (one bool per required point)','claim_support (one bool per output claim; every factual proposition supported by supplied cited text and context, no certainty upgrade)','certainty_and_scope_preserved (bool)','safe_abstention (bool for unanswerable, null otherwise)','complete_correct (bool)','issues (specific concise list)'],'no_claim_cases':'No claims is not a correct answer to an answerable query. For unanswerable explicit safe abstention can be correct with no claims; do not invent a citation precision denominator.','sources':'Use blind-source-registry.json and q reference anchors; read additional source-pack paragraphs only when context needed. Both groups source-reviewed; controlled rules explicitly virtual, source-world question not empirical engine diagnosis.','no_real_quality_claim':'24 questions across two different groups are not pooled as real-world accuracy; source-only refs are nonexpert labels.'})
print('Blinded48 submissions; identity map separate; evidence registry',len(registry))
