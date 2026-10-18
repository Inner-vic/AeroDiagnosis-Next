import sys,json,hashlib,time
from pathlib import Path
R=Path('D:/codex/worktrees/knowledge-upgrade-integration/AeroDiagnosis')
sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import controlled_path_fusion_v2 as b
from aerodiagnosis.application.knowledge_qa_live import GatedKnowledgeQA
from aerodiagnosis.domain.knowledge_query import KnowledgeQueryContext
O=R/'.runtime/qa-integration-v1';O.mkdir(exist_ok=True)
env=b.environment('test');chunks=env[0];provider=b.LocalE5EmbeddingProvider(Path('.runtime/models/multilingual-e5-small'))
raws={q['query']:b.read(b.OUT/'candidates/test'/(q['sample_id'].replace(':','_')+'.json')) for q in b.questions('test')}
class Cached:
 backend_name='frozen_real_candidate_replay'
 def __init__(self,route):self.route=route
 def search(self,query,*,top_k,active_version_ids):
  return [b.VectorMatch(chunks[x['chunk_id']],x['score']) for x in raws[query]['routes'][self.route] if chunks[x['chunk_id']].version_id in active_version_ids][:top_k]
graph=b.FrozenGraph(b.read(b.DATA/'graph.json'))
qa=GatedKnowledgeQA(sparse=Cached('Bs'),dense=Cached('Ds'),graph=graph,catalog=lambda c:chunks,token_count=lambda text:len(provider.counter.encode(text).ids))
expected={r['sample_id']:r for r in b.jsonl(b.OUT/'test-predictions.jsonl') if r['method']=='BDG_gated_scope'}
rows=[];started=time.perf_counter()
for i,q in enumerate(b.questions('test')):
 r=qa.retrieve(q['query'],KnowledgeQueryContext());actual=[e['id'] for e in r['evidence']]
 assert actual==expected[q['sample_id']]['hits'],q['sample_id']
 for p in r['paths']:assert set(p['source_chunks'])<=set(actual)
 rows.append({'sample_id':q['sample_id'],'kind':q['kind'],'hits_equal_frozen':True,'graph_enabled':r['retrieval']['graph_enabled'],'complete_paths':len(r['paths'])})
 if (i+1)%36==0:print('Parity',i+1,flush=True)
b.save(O/'ranking-parity.json',{'n':len(rows),'all_exact_top10_equal':True,'graph_open':sum(x['graph_enabled'] for x in rows),'with_complete_display_path':sum(bool(x['complete_paths']) for x in rows),'seconds':time.perf_counter()-started,'candidate_source':'Frozen original BM25 and E5 caches; live GraphPathCandidates and live gate/fusion code; no LLM','rows':rows})
print('PASS parity216',flush=True)
