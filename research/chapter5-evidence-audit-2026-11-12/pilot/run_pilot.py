"""Frozen zero-provider retrieval pilot. Gold is used only after search completes."""
from pathlib import Path
import collections,dataclasses,hashlib,json,math,sys,time
P=Path(__file__).resolve().parent
R=Path('D:/codex/worktrees/knowledge-upgrade-integration/AeroDiagnosis')
sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
from aerodiagnosis.adapters.persistence.sqlite_chinese_vector import SQLiteChineseVectorStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.application.knowledge_qa_live import GatedKnowledgeQA,display_text
from aerodiagnosis.domain.knowledge_query import KnowledgeQueryContext
from aerodiagnosis.ports import VectorChunk
from aerodiagnosis.retrieval.graph_paths import GraphPathCandidates
from aerodiagnosis.retrieval.fusion_policy import fuse_ranked,select_fused
from evaluate_capability_development import FrozenGraph
def read(n):return json.loads((P/n).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(P/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
CODE=[R/'src/aerodiagnosis/application/knowledge_qa_live.py',R/'src/aerodiagnosis/retrieval/atomic_path_fusion.py',R/'src/aerodiagnosis/retrieval/graph_paths.py',R/'src/aerodiagnosis/retrieval/fusion_policy.py']
METHODS=['B','D','G','BD','BG','DG','BDG','GATED','BD_matched50']
def freeze():
 assert not (P/'predictions.jsonl').exists()
 ids=[s['id'] for s in read('sources.json')];matches={k:[] for k in ids};files=0
 for f in (R/'evaluation').rglob('*'):
  if f.suffix.lower() not in {'.json','.jsonl','.md'}:continue
  files+=1;t=f.read_text(encoding='utf8',errors='replace').lower()
  for k in ids:
   if k in t:matches[k].append(str(f.relative_to(R)))
 assert not any(matches.values()),matches
 save('source-overlap-audit.json',{'checked_files':files,'historical_root':str(R/'evaluation'),'report_id_matches':matches,'method':'case-insensitive report identifier scan; not semantic near-duplicate detection','expert_review':False})
 protocol={'date':'2026-11-12','status':'source-disjoint excerpt pilot, not final confirmatory test','source_clusters':2,'questions':16,'answerable_questions':14,'strict_cross_block_questions':4,'primary_K':5,'secondary_K':[3,10],'primary_scope':'explicit document selection supplied equally to every method','secondary_scope':'all 16 excerpts','baseline':'BD','same_text_quota_control':'BD_matched50','methods':METHODS,'weights':{'B':1.,'D':.25,'G':.25},'RRF_constant':60,'budget':{'single':150,'dual_each':75,'triple_each':50,'gate_closed_text_each':50,'pool':50,'evidence':10,'max_per_document':5,'tokens':4096},'primary_metrics':['ECR@5','EDR@5','paired gains/losses/ties','complete@5'],'zero_denominator':'null, never zero or one','insufficient_questions':'reported as diagnostic cases, excluded from evidence-completeness metrics; no generation or refusal claim','annotation':'single AI authored and checked; not independent/expert gold','graph':'source-curated six assertions; measures retrieval mechanism conditional on graph availability, not automatic extraction accuracy or Neo4j throughput','translation':'AI translation, source excerpts retained; no provider calls','limits':['Only two source clusters','One piston-engine report is a domain-extension control, not turbine-generalization evidence','Indexed excerpts rather than full official report capture; summaries elsewhere may answer the question','Strict cross-block status is relative to the captured excerpt set only','Some exact-endpoint and some natural paraphrase queries retained; no post-result rewriting'],'source_acquisition_failures':['official direct download HTTP403','ept webfetch department quota exhausted'],'paid_calls':0,'post_result_tuning':False}
 save('protocol.json',protocol)
 files=[P/n for n in ['corpus.json','graph.json','questions.json','sources.json','protocol.json','source-overlap-audit.json','build_pilot.py','run_pilot.py']]+CODE
 save('freeze.json',{'before_any_retrieval':True,'files':{str(p):sha(p) for p in files}})
 print('Frozen 16 questions, 2 new source clusters, code and protocol')
def metrics(q,hits,k):
 sets=[set(x) for x in q['acceptable_support_sets']]
 if not sets:return None
 hs=hits[:k];gold=max(sets,key=lambda s:(len(s&set(hs))/len(s),-len(s)))
 dcg=sum(1/math.log2(i+2) for i,h in enumerate(hs) if h in gold);ideal=sum(1/math.log2(i+2) for i in range(min(k,len(gold))))
 return {'complete':int(any(s<=set(hs) for s in sets)),'recall':len(gold&set(hs))/len(gold),'ndcg':dcg/ideal if ideal else 0}
def run():
 frozen=read('freeze.json');assert all(sha(Path(k))==v for k,v in frozen['files'].items())
 assert not (P/'predictions.jsonl').exists(),'Do not overwrite first-run results'
 provider=LocalE5EmbeddingProvider(R/'.runtime/models/multilingual-e5-small')
 db=P/'index.sqlite';dense=SQLiteChineseVectorStore(db,embedding_provider=provider);sparse=SQLiteLexicalStore(db)
 dense.upsert([VectorChunk(**c) for c in read('corpus.json')['chunks']]);sparse.upsert([VectorChunk(**c) for c in read('corpus.json')['chunks']])
 # Persisted dense metadata is added explicitly for identity checking.
 chunks={c['chunk_id']:VectorChunk(**c) for c in read('corpus.json')['chunks']}
 chunks={k:dataclasses.replace(c,metadata={**c.metadata,'vector_embedding_input':{'policy':'chinese_display@1','provider':provider.name,'sha256':hashlib.sha256(display_text(c).encode()).hexdigest()}}) for k,c in chunks.items()}
 graph=FrozenGraph(read('graph.json'));tokens=lambda text:len(provider.counter.encode(text).ids)
 def retrieve(query,doc,method,scope):
  catalog={k:c for k,c in chunks.items() if scope=='all' or c.document_id==doc};active=frozenset(c.version_id for c in catalog.values());load=lambda keys:{k:catalog[k] for k in keys if k in catalog}
  if method=='GATED':
   service=GatedKnowledgeQA(sparse=sparse,dense=dense,graph=graph,catalog=lambda _:catalog,token_count=tokens)
   out=service.retrieve(query,KnowledgeQueryContext(document_id=doc if scope=='document' else None))
   return [e['id'] for e in out['evidence']],out['retrieval'],out['paths']
  paths=GraphPathCandidates(graph,load);stores={'B':sparse,'D':dense,'G':paths};channels=list(method) if method!='BD_matched50' else ['B','D'];k=50 if method=='BD_matched50' else 150//len(channels)
  routes={n:stores[n].search(query,top_k=k,active_version_ids=active) for n in channels}
  pool=fuse_ranked(routes,{n:{'B':1.,'D':.25,'G':.25}[n] for n in channels})
  selected,trace=select_fused(pool,token_count=lambda c:tokens(display_text(c)),mode='legacy_dedup',top_k=10,pool_k=50,max_per_document=5,token_budget=4096)
  return [c.chunk.chunk_id for c,_ in selected],{**trace,'route_counts':{n:len(v) for n,v in routes.items()}},[]
 rows=[]
 for scope in ['document','all']:
  for q in read('questions.json'):
   for method in METHODS:
    start=time.perf_counter();hits,trace,paths=retrieve(q['question'],q['document_id'],method,scope)
    row={'sample_id':q['sample_id'],'kind':q['kind'],'document_id':q['document_id'],'scope':scope,'method':method,'hits':hits,'trace':trace,'paths':paths,'elapsed_seconds':time.perf_counter()-start,'metrics':{str(k):metrics(q,hits,k) for k in [3,5,10]}}
    rows.append(row)
    with (P/'predictions.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
   print(scope,q['sample_id'],flush=True)
 assert all(sha(Path(k))==v for k,v in frozen['files'].items())
 groups=collections.defaultdict(dict)
 for row in rows:groups[row['scope'],row['sample_id']][row['method']]=row
 summary={}
 for scope in ['document','all']:
  summary[scope]={}
  for method in METHODS:
   for kind in ['ALL','fact','paraphrase','chain','chain_paraphrase']:
    pairs=[rs for (s,_),rs in groups.items() if s==scope and (kind=='ALL' or rs['BD']['kind']==kind) and rs['BD']['metrics']['5'] is not None]
    a=sum(rs['BD']['metrics']['5']['complete'] for rs in pairs);b=sum(rs[method]['metrics']['5']['complete'] for rs in pairs)
    gain=sum(not rs['BD']['metrics']['5']['complete'] and rs[method]['metrics']['5']['complete'] for rs in pairs);loss=sum(rs['BD']['metrics']['5']['complete'] and not rs[method]['metrics']['5']['complete'] for rs in pairs)
    summary[scope][method+':'+kind]={'n':len(pairs),'baseline_complete':a,'complete':b,'gain':gain,'loss':loss,'ties':len(pairs)-gain-loss,'ECR':gain/(len(pairs)-a) if len(pairs)>a else None,'EDR':loss/a if a else None,'gate_open':sum(bool(rs[method]['trace'].get('graph_enabled')) for rs in pairs),'graph_errors':sum(bool(rs[method]['trace'].get('graph_errors')) for rs in pairs)}
 save('summary.json',summary);save('execution-receipt.json',{'rows':len(rows),'paid_calls':0,'code_and_data_unchanged':True,'predictions_sha256':sha(P/'predictions.jsonl'),'first_run_retained':True})
 print(json.dumps(summary['document']['GATED:ALL'],ensure_ascii=False))
if __name__=='__main__':
 assert len(sys.argv)==2 and sys.argv[1] in {'freeze','run'}
 (freeze if sys.argv[1]=='freeze' else run)()
