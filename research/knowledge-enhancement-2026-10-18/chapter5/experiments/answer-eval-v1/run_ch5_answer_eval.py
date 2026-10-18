"""Frozen paired answer experiment; no reference answers passed to retrieval or LLM."""
from __future__ import annotations
import sys,json,time,hashlib,sqlite3,math,shutil
from pathlib import Path
from dataclasses import replace
from urllib.parse import urlsplit
R=Path('D:/codex/worktrees/knowledge-upgrade-integration/AeroDiagnosis')
sys.path[:0]=[str(R/'scripts'),str(R/'src')]
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.adapters.api.qa import create_qa,QACatalog
from aerodiagnosis.application.knowledge_qa_live import GatedKnowledgeQA,display_text
from aerodiagnosis.domain.knowledge_query import KnowledgeQueryContext
from aerodiagnosis.ports import VectorChunk,VectorMatch,LanguageModelError
from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
from aerodiagnosis.adapters.persistence.sqlite_chinese_vector import SQLiteChineseVectorStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.adapters.llm.research_client import ResearchClient,ResearchCallError
from aerodiagnosis.adapters.llm.openai_compatible import _TASK_INSTRUCTIONS
from aerodiagnosis.retrieval.atomic_path_fusion import identifier_scope
from aerodiagnosis.retrieval.fusion_policy import fuse_ranked,select_fused
from controlled_scenario_benchmark_v1 import FrozenGraph

P=R/'evaluation/chapter5_answer_pilot_v1';O=R/'.runtime/chapter5-answer-eval-v1';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def budget():
 with sqlite3.connect(R/'.runtime/thesis-research-provider-v1/calls.sqlite3') as db:rows=db.execute('SELECT state,reservation_micro,cost_micro FROM calls').fetchall()
 return {'provider_calls':len(rows),'upper_cny':sum(a if s in ('pending','unknown_charge') else b for s,a,b in rows)/1e6,'unresolved_calls':sum(s in ('pending','unknown_charge') for s,_,_ in rows)}

class TextBaseline(GatedKnowledgeQA):
 def retrieve(self,query,context):
  start=time.perf_counter();chunks=self.catalog(context);active=frozenset(c.version_id for c in chunks.values());scope=identifier_scope(query,chunks);active=active if scope is None else active&scope
  chunks={k:c for k,c in chunks.items() if c.version_id in active};routes={}
  if not active:return {'evidence':[],'paths':[],'retrieval':{'policy':'source_B1D025_75each','graph_enabled':False,'reason':'no_active_sources'}}
  for name,store in [('B',self.sparse),('D',self.dense)]:
   found=store.search(query,top_k=75,active_version_ids=active);assert len(found)<=75
   routes[name]=[VectorMatch(chunks[m.chunk.chunk_id],m.score) for m in found if m.chunk.chunk_id in chunks and (m.chunk.document_id,m.chunk.version_id,m.chunk.content)==(chunks[m.chunk.chunk_id].document_id,chunks[m.chunk.chunk_id].version_id,chunks[m.chunk.chunk_id].content)]
  pool=fuse_ranked(routes,{'B':1.,'D':.25});selected,trace=select_fused(pool,token_count=lambda c:self.tokens(display_text(c)),mode='legacy_dedup',top_k=10,pool_k=50,max_per_document=5,token_budget=4096)
  evidence=[]
  for candidate,score in selected:
   c=candidate.chunk;text=display_text(c)
   evidence.append({'id':c.chunk_id,'chunk_id':c.chunk_id,'document_id':c.document_id,'version_id':c.version_id,'title':c.metadata.get('qa_title',c.document_id),'text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'text_kind':'chinese_translation' if c.metadata.get('display_content_zh') else 'canonical_source','locator':c.metadata.get('locator',{}),'source_url':c.metadata.get('qa_source_url'),'score':score,'route_ranks':dict(candidate.route_ranks),'applicability_status':c.metadata.get('qa_applicability','not_checked')})
  return {'evidence':evidence,'paths':[],'retrieval':{**trace,'policy':'source_B1D025_75each','graph_enabled':False,'reason':'text_control','route_counts':{k:len(v) for k,v in routes.items()},'elapsed_ms':round((time.perf_counter()-start)*1000,2)}}

class Model:
 identity='deepseek-flash:paired-answer-eval-max1024'
 def __init__(self,client,questions):self.client=client;self.questions=questions;self.receipts=[];self.error=None
 def complete_json(self,task,payload):
  assert task in {'answer_knowledge','verify_knowledge'} and payload['question'] in self.questions
  try:
   c=self.client.complete(task=task,instruction=_TASK_INSTRUCTIONS[task],payload=payload,stage='generation_evaluation',max_tokens=1024)
   self.receipts.append(c.receipt);return c.value
  except ResearchCallError as exc:
   self.error=type(exc).__name__;raise LanguageModelError(type(exc).__name__) from None

def setup():
 settings=RuntimeSettings.from_env(project_root=R/'.runtime/chinese-hybrid-v1/profile')
 assert settings.embedding_backend=='local_e5' and settings.graph_backend=='neo4j' and settings.external_store_mode=='external-primary'
 application=bootstrap(settings);live=create_qa(application);public=read(P/'sources.json');all_public_chunks={c['chunk_id']:c for d in public.values() for c in d['chunks']}
 original_catalog=QACatalog(application)
 def public_catalog(context):
  assert context.document_id in public
  chunks=original_catalog(context)
  for k,c in chunks.items():
   expected=all_public_chunks[k]
   assert c.document_id==context.document_id and c.version_id==expected['version_id']
   assert urlsplit(str(c.metadata['qa_source_url'])).hostname=='www.faa.gov'
   assert hashlib.sha256(display_text(c).encode()).hexdigest()==expected['text_sha256']
  return chunks
 live.catalog=public_catalog
 text=TextBaseline(sparse=live.sparse,dense=live.dense,graph=live.graph,catalog=public_catalog,token_count=live.tokens)
 data=read(R/'evaluation/controlled_scenario_v2/corpus.json');controlled=read(P/'controlled-sources.json')
 all_controlled={c['chunk_id']:replace(VectorChunk(**c),metadata={**c['metadata'],'qa_title':'程序化构造的虚拟试验台规则资料','qa_source_url':None,'qa_applicability':'not_checked'}) for c in data['chunks']}
 def controlled_catalog(context):
  assert context.document_id in controlled
  return {k:c for k,c in all_controlled.items() if c.document_id==context.document_id}
 frozen=FrozenGraph(read(R/'evaluation/controlled_scenario_v2/graph.json'))
 sparse=SQLiteLexicalStore(R/'.runtime/controlled-path-fusion-v2/index.sqlite')
 counter=LocalE5EmbeddingProvider(R/'.runtime/models/multilingual-e5-small')
 dense=SQLiteChineseVectorStore(R/'.runtime/controlled-path-fusion-v2/index.sqlite',embedding_provider=counter)
 token_count=lambda t:len(counter.counter.encode(t).ids)
 clive=GatedKnowledgeQA(sparse=sparse,dense=dense,graph=frozen,catalog=controlled_catalog,token_count=token_count)
 ctext=TextBaseline(sparse=sparse,dense=dense,graph=frozen,catalog=controlled_catalog,token_count=token_count)
 key=settings.default_llm_api_key.get_secret_value() if settings.default_llm_api_key else None
 assert key
 client=ResearchClient(api_key=key,runtime=settings.knowledge_budget_runtime,model='deepseek-flash',cap_cny=42,timeout=60)
 return {'real':{'BD':text,'GATED':live},'controlled':{'BD':ctext,'GATED':clive}},client

def freeze():
 assert not (O/'predictions.jsonl').exists()
 review=Path('D:/zhaoweihao3/Desktop/Project/AeroDiagnosis/.runtime/ch5-answer-review')
 for filename in ['source-review.json','controlled-source-review.json']:
  r=read(review/filename);assert r['approved'] and not r['expert_reviewed'] and not r['answers_seen'];shutil.copyfile(review/filename,O/filename)
 files=list(P.glob('*.json'))+[Path(__file__),R/'src/aerodiagnosis/application/knowledge_qa_live.py',R/'src/aerodiagnosis/retrieval/fusion_policy.py',R/'src/aerodiagnosis/retrieval/atomic_path_fusion.py',R/'src/aerodiagnosis/retrieval/graph_paths.py',R/'src/aerodiagnosis/adapters/llm/openai_compatible.py',O/'source-review.json',O/'controlled-source-review.json']
 save(O/'execution-freeze.json',{'before_model_answers':True,'hashes':{str(p):sha(p) for p in files},'budget_before':budget(),'groups_separate':True,'same_payload_cache_reuse':True,'cache_timing_not_speed_evidence':True,'model_max_tokens':1024,'ordered_pairs':'for i0..11: real_i,controlled_i; order BD/GATED if i even, reverse if odd'})
 print('Frozen24 paired questions and executable code before answers',flush=True)

def run():
 frozen=read(O/'execution-freeze.json');assert all(sha(Path(p))==h for p,h in frozen['hashes'].items())
 services,client=setup();real=read(P/'questions.json');controlled=read(P/'controlled-questions.json')
 existing=[json.loads(x) for x in (O/'predictions.jsonl').read_text(encoding='utf-8').splitlines()] if (O/'predictions.jsonl').exists() else []
 done={(r['id'],r['method']) for r in existing};allowed={q['question'] for q in real+controlled};model=Model(client,allowed)
 budget_stopped=False
 for i in range(12):
  for group,q in [('real',real[i]),('controlled',controlled[i])]:
   for method in (['BD','GATED'] if i%2==0 else ['GATED','BD']):
    if (q['id'],method) in done:continue
    if budget_stopped:
     row={'id':q['id'],'group':group,'method':method,'completed':False,'error':'budget_cap_stopped','model_receipts':[]}
    else:
     model.receipts=[];model.error=None;started=time.perf_counter();before=budget()
     service=services[group][method]
     result=service.answer(q['question'],KnowledgeQueryContext(document_id=q['document_id']),model)
     assert all(e['document_id']==q['document_id'] for e in result['evidence'])
     if group=='real':assert all(urlsplit(e['source_url']).hostname=='www.faa.gov' for e in result['evidence'])
     row={'id':q['id'],'group':group,'method':method,'completed':model.error is None,'error':model.error,'elapsed_seconds':time.perf_counter()-started,'budget_before':before,'budget_after':budget(),'model_receipts':model.receipts,'response':result}
     budget_stopped=model.error=='BudgetExceeded'
    with (O/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(q['id'],method,'ok' if row['completed'] else row['error'],flush=True)
 save(O/'execution-receipt.json',{'budget_after':budget(),'predictions_sha256':sha(O/'predictions.jsonl'),'no_runtime_or_question_changes':all(sha(Path(p))==h for p,h in frozen['hashes'].items())})
 print(json.dumps(budget()),flush=True)

if __name__=='__main__':
 assert sys.argv[1] in ['freeze','run']
 (freeze if sys.argv[1]=='freeze' else run)()
