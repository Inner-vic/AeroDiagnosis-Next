"""Rebuild affected documents from preserved Docling structures and repair source-grounded roles."""

import hashlib
import json
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path

from aerodiagnosis.adapters.docling_parser import DoclingParser
from aerodiagnosis.adapters.llm.research_client import ResearchClient
from aerodiagnosis.adapters.persistence.neo4j_graph import Neo4jGraphStore
from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.application.graph_quality import normalize_graph
from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.ingestion import DocumentManifest, ParserRegistry

root = Path.cwd()
out = root / ".runtime/experiments/zh-graph-repair-20261014"
out.mkdir(parents=True, exist_ok=True)
settings = RuntimeSettings.from_env(root / ".runtime/fusion/profile")
manifest = DocumentManifest(settings.database_path)
graph = Neo4jGraphStore(
    uri=settings.neo4j_uri,
    user=settings.neo4j_user,
    password=settings.neo4j_password.get_secret_value(),
    database=settings.neo4j_database,
)
nodes, edges = graph.snapshot(active_version_ids=manifest.active_version_ids(), limit=2000)
if not (out / "before-graph.json").exists():
    (out / "before-graph.json").write_text(
        json.dumps(
            {"nodes": [asdict(n) for n in nodes], "edges": [asdict(e) for e in edges]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf8",
    )
    with (
        sqlite3.connect(settings.database_path) as source,
        sqlite3.connect(out / "before.sqlite") as target,
    ):
        source.backup(target)
# Recover immutable source surfaces from exact-input translation receipts.
surfaces = {}
for path in (settings.knowledge_budget_runtime / "cache").glob("*.json"):
    saved = json.loads(path.read_text(encoding="utf8"))
    if saved.get("receipt", {}).get("task") == "graph-display-zh":
        payload = json.loads(saved["request"]["messages"][1]["content"])
        surfaces.update(payload.get("items", {}))
quotes = {
    key: e.properties.get("evidence_text", "") for e in edges for key in [e.source_id, e.target_id]
}
updated = []
for n in nodes:
    origin = n.properties.get("display_translation", {}).get("source_name_sha256")
    updated.append(
        replace(
            n,
            properties={
                **n.properties,
                "source_surface": surfaces.get(origin, n.properties.get("source_surface", n.name)),
                "source_assertion": quotes.get(n.node_id, ""),
            },
        )
    )
normalized, normalized_edges = normalize_graph(updated, edges)
graph.upsert_nodes(normalized)
graph.upsert_edges(normalized_edges)
mirror = SQLiteGraphStore(settings.database_path)
mirror.upsert_nodes(normalized)
mirror.upsert_edges(normalized_edges)
(
    out
    / ("role-repair.json" if not (out / "role-repair.json").exists() else "role-repair-replay.json")
).write_text(
    json.dumps(
        {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "direction_repairs": [
                e.edge_id
                for e in normalized_edges
                if e.source_id != next(old.source_id for old in edges if old.edge_id == e.edge_id)
            ],
            "type_repairs": [
                n.node_id
                for n in normalized
                if n.kind != next(old.kind for old in nodes if old.node_id == n.node_id)
            ],
        },
        indent=2,
    ),
    encoding="utf8",
)
print("Grounded role repairs saved", flush=True)
catalog = json.loads(
    (root / ".runtime/fusion/source-catalog/catalog.json").read_text(encoding="utf8")
)
affected = {
    "prebuilt-compressor-damage",
    "prebuilt-combustor-inspection",
    "prebuilt-starting-performance",
    "prebuilt-vibration-guidance",
}
picture_receipt = out / "picture-affected.json"
if picture_receipt.exists():
    affected.update(json.loads(picture_receipt.read_text(encoding="utf8")))
structures = {}
for doc in manifest.list_documents():
    if doc.document_id in affected:
        folder = settings.runtime_dir / "sources" / doc.active_version_id
        raw = (folder / "original.bin").read_bytes()
        structures[hashlib.sha256(raw).hexdigest()] = json.loads(
            (folder / "docling.json").read_text(encoding="utf8")
        )
parser = DoclingParser(".pdf", tokenizer_path=settings.knowledge_tokenizer_path)


class CachedStructureParser:
    def parse(self, content):
        digest = hashlib.sha256(content).hexdigest()
        return parser.parse_structure(content, structures[digest])


app = bootstrap(settings)
app.ingest_document._ingestion._parsers = ParserRegistry(parsers={".pdf": CachedStructureParser()})
records = []
receipt = out / "rebuild-receipt.json"
if receipt.exists():
    records = json.loads(receipt.read_text(encoding="utf8"))["documents"]
completed = {r["document_id"] for r in records}
for row in catalog["documents"]:
    if row["document_id"] not in affected or row["document_id"] in completed:
        continue
    from aerodiagnosis.domain import DocumentMetadata

    content = (root / ".runtime/fusion/source-catalog" / row["path"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != row["sha256"]:
        raise ValueError("Original source hash changed")
    result = app.ingest_document.execute(
        display_name=row["display_name"],
        content=content,
        document_id=row["document_id"],
        metadata=DocumentMetadata.model_validate(row["metadata"]),
    )
    records.append(
        {**asdict(result), "status": result.status.value, "original_sha256": row["sha256"]}
    )
    receipt.write_text(
        json.dumps(
            {"documents": records, "old_versions_preserved": True}, ensure_ascii=False, indent=2
        ),
        encoding="utf8",
    )
    print(row["document_id"], result.chunk_count, result.status.value, flush=True)
client = ResearchClient(
    api_key=settings.default_llm_api_key.get_secret_value(),
    runtime=settings.knowledge_budget_runtime,
    model=settings.default_llm_model,
    cap_cny=5,
)
print(
    json.dumps({k: v for k, v in client.summary().items() if k != "receipts"}, ensure_ascii=False),
    flush=True,
)
