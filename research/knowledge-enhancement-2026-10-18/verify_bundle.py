"""Offline integrity verification of the public research bundle; no model calls."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent
m=json.loads((R/'MANIFEST.json').read_text(encoding='utf-8'))
for x in m['files']:
 p=R/x['path'];assert p.is_file(),x['path'];assert p.stat().st_size==x['bytes'],x['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],x['path']
nodes=[json.loads(x) for x in (R/'knowledge-base/graph_nodes.jsonl').read_text(encoding='utf-8').splitlines()]
edges=[json.loads(x) for x in (R/'knowledge-base/graph_edges.jsonl').read_text(encoding='utf-8').splitlines()]
ids={x['node_id'] for x in nodes}
assert all(e['source_id'] in ids and e['target_id'] in ids for e in edges)
print(json.dumps({'verified_files':len(m['files']),'graph_nodes':len(nodes),'graph_edges':len(edges),'api_calls':0}))
