# ruff: noqa: RUF001 -- Chinese user-facing punctuation.
"""Request-local gated retrieval and span-bound answers, without benchmark labels."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any, cast

from aerodiagnosis.application.grounded_answer import (
    EvidenceSpan,
    assemble_answer,
    spans_for_evidence,
)
from aerodiagnosis.application.knowledge_applicability import (
    assess_assertion_applicability,
    permits,
)
from aerodiagnosis.domain.knowledge_query import KnowledgeQueryContext
from aerodiagnosis.ports import (
    GraphEdge,
    GraphNode,
    GraphStore,
    LanguageModel,
    LanguageModelError,
    VectorChunk,
    VectorMatch,
)
from aerodiagnosis.ports.vector import CandidateSearch
from aerodiagnosis.retrieval.atomic_path_fusion import (
    PathBundle,
    identifier_scope,
    verified_bundles,
)
from aerodiagnosis.retrieval.fusion_policy import fuse_ranked, select_fused
from aerodiagnosis.retrieval.graph_paths import GraphPathCandidates

POLICY = "live-source-scoped-gated-bdg@1"
RELATIONS = {"CAUSES": "导致", "CONTRIBUTES_TO": "促成", "HAS_MECHANISM": "作用机理"}


def display_text(chunk: VectorChunk) -> str:
    translated = chunk.metadata.get("display_content_zh")
    if isinstance(translated, str) and translated.strip():
        if (
            chunk.metadata.get("display_translation_source_sha256")
            != hashlib.sha256(chunk.content.encode()).hexdigest()
        ):
            raise ValueError("中文译文与资料版本不一致")
        return translated
    return chunk.content


class GatedKnowledgeQA:
    def __init__(
        self,
        *,
        sparse: CandidateSearch,
        dense: CandidateSearch,
        graph: GraphStore,
        catalog: Callable[[KnowledgeQueryContext], dict[str, VectorChunk]],
        token_count: Callable[[str], int],
    ) -> None:
        self.sparse, self.dense, self.graph = sparse, dense, graph
        self.catalog, self.tokens = catalog, token_count

    def retrieve(self, query: str, context: KnowledgeQueryContext) -> dict[str, Any]:
        started = time.perf_counter()
        chunks = self.catalog(context)
        scope = identifier_scope(query, chunks)
        active = frozenset(c.version_id for c in chunks.values())
        active = active if scope is None else active & scope
        chunks = {k: c for k, c in chunks.items() if c.version_id in active}
        trace: dict[str, Any] = {
            "policy": POLICY,
            "graph_enabled": False,
            "reason": "no_active_sources" if not active else "no_verified_path",
            "scope_versions": sorted(active),
            "weights": {"B": 1.0, "D": 0.25, "G": 0.25},
            "requested_per_route": 50,
            "graph_errors": [],
            "path_rejections": {},
        }
        if not active:
            return {"evidence": [], "paths": [], "retrieval": trace}

        def load(ids: Sequence[str]) -> dict[str, VectorChunk]:
            return {k: chunks[k] for k in ids if k in chunks}

        routes: dict[str, list[VectorMatch]] = {}
        for name, store in [("B", self.sparse), ("D", self.dense)]:
            found = store.search(query, top_k=50, active_version_ids=active)
            if len(found) > 50:
                raise ValueError("检索候选超过通道配额")
            routes[name] = [
                VectorMatch(chunks[m.chunk.chunk_id], m.score)
                for m in found
                if m.chunk.chunk_id in chunks
                and m.chunk.content == chunks[m.chunk.chunk_id].content
                and m.chunk.version_id == chunks[m.chunk.chunk_id].version_id
                and m.chunk.document_id == chunks[m.chunk.chunk_id].document_id
            ]
        bundles: list[PathBundle] = []
        graph_hits: list[VectorMatch] = []
        nodes, edges = {}, {}
        try:
            # One immutable request snapshot is shared by discovery and verification.
            for version in sorted(active):
                nn, ee = self.graph.snapshot(active_version_ids=frozenset({version}), limit=2000)
                if len(nn) >= 2000 or len(ee) >= 4000:
                    raise ValueError("图谱快照超出校验上限")
                nodes.update({n.node_id: n for n in nn if n.version_id == version})
                edges.update(
                    {
                        e.edge_id: e
                        for e in ee
                        if e.version_id == version
                        and permits(
                            assess_assertion_applicability(
                                e.properties,
                                next(
                                    (
                                        c.metadata.get("qa_document_metadata", {})
                                        for c in chunks.values()
                                        if c.version_id == version
                                    ),
                                    {},
                                ),
                                context,
                            ),
                            context,
                        )
                    }
                )

            class Snapshot:
                backend_name = "request_snapshot"

                def snapshot(
                    self, *, active_version_ids: frozenset[str] | None = None, limit: int = 2000
                ) -> tuple[list[GraphNode], list[GraphEdge]]:
                    return (
                        [
                            n
                            for n in nodes.values()
                            if n.version_id in (active_version_ids or frozenset())
                        ],
                        [
                            e
                            for e in edges.values()
                            if e.version_id in (active_version_ids or frozenset())
                        ],
                    )

            path_store = GraphPathCandidates(cast(GraphStore, Snapshot()), load)
            graph_hits = path_store.search(query, top_k=50, active_version_ids=active)
            bundles, rejected = verified_bundles(
                query, graph_hits, nodes=nodes, edges=edges, chunks=chunks
            )
            trace["path_rejections"] = rejected
        except Exception as exc:
            # Graph unavailability never promotes an unverified path or suppresses text search.
            trace["graph_errors"] = [type(exc).__name__]
            trace["reason"] = "graph_unavailable"
        if bundles:
            routes["G"] = graph_hits
            trace.update(graph_enabled=True, reason="verified_causal_path")
        elif trace["path_rejections"].get("noncausal_or_boundary"):
            trace["reason"] = "noncausal_or_boundary"
        weights = {k: trace["weights"][k] for k in routes}
        pool = fuse_ranked(routes, weights)
        selected, selection = select_fused(
            pool,
            token_count=lambda c: self.tokens(display_text(c)),
            mode="legacy_dedup",
            top_k=10,
            pool_k=50,
            max_per_document=5,
            token_budget=4096,
        )
        evidence = []
        for candidate, score in selected:
            c = candidate.chunk
            evidence.append(
                {
                    "id": c.chunk_id,
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "version_id": c.version_id,
                    "title": c.metadata.get("qa_title", c.document_id),
                    "text": display_text(c),
                    "text_sha256": hashlib.sha256(display_text(c).encode()).hexdigest(),
                    "text_kind": "chinese_translation"
                    if c.metadata.get("display_content_zh")
                    else "canonical_source",
                    "locator": c.metadata.get("locator", {}),
                    "source_url": c.metadata.get("qa_source_url"),
                    "score": score,
                    "route_ranks": dict(candidate.route_ranks),
                    "applicability_status": c.metadata.get("qa_applicability", "not_checked"),
                }
            )
        selected_ids = {e["id"] for e in evidence}
        paths = []
        for bundle in bundles:
            if not set(bundle.chunks) <= selected_ids:
                continue
            ee = [edges[key] for key in bundle.edges]
            nn = [nodes[ee[0].source_id], *[nodes[e.target_id] for e in ee]]
            paths.append(
                {
                    "edge_ids": list(bundle.edges),
                    "source_chunks": list(bundle.chunks),
                    "entities": [str(n.properties.get("name_zh") or n.name) for n in nn],
                    "relations": [RELATIONS.get(e.relation, e.relation) for e in ee],
                    "hops": bundle.hops,
                    "complete_in_context": True,
                    "source_proofs": [
                        {
                            "chunk_id": str(p["chunk_id"]),
                            "canonical_start": p["canonical_start"],
                            "canonical_end": p["canonical_end"],
                            "canonical_sha256": p["canonical_sha256"],
                            "evidence_text": p["evidence_text"],
                        }
                        for e in ee
                        for p in [e.properties, *e.properties.get("supporting_evidence", [])]
                    ],
                }
            )
        trace.update(
            selection,
            route_counts={k: len(v) for k, v in routes.items()},
            verified_path_count=len(bundles),
            complete_selected_paths=len(paths),
            missing_path_context=bool(bundles and not paths),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return {"evidence": evidence, "paths": paths, "retrieval": trace}

    def answer(
        self, query: str, context: KnowledgeQueryContext, model: LanguageModel | None
    ) -> dict[str, Any]:
        data = self.retrieve(query, context)
        result: dict[str, Any] = {
            "kind": "grounded_qa",
            "run_id": str(uuid.uuid4()),
            "question": query,
            **data,
            "status": "insufficient_evidence",
            "claims": [],
            "answer": "现有证据不足以完整回答该问题。",
            "insufficient_evidence": True,
            "reason": "未找到当前筛选条件下的证据。",
            "generation": {"model_calls": 0},
        }
        if not data["evidence"]:
            return result
        if model is None:
            result["reason"] = "尚未配置回答模型，以下仅展示检索证据，未生成结论。"
            return result
        # Compact full-evidence spans, bounded INCLUDING selectors and source labels.
        packed: list[str] = []
        catalog: list[dict[str, Any]] = []
        aliases: dict[str, str] = {}
        spans: list[EvidenceSpan] = []
        for item in data["evidence"]:
            ss = spans_for_evidence([{"id": item["id"], "text": item["text"]}])
            selectors = [{"id": f"s{len(spans) + i + 1}", "text": s.text} for i, s in enumerate(ss)]
            entry = {
                "id": item["id"],
                "title": item["title"],
                "spans": selectors,
                "applicability_status": item["applicability_status"],
            }
            if (
                self.tokens(
                    json.dumps(
                        {"question": query, "evidence": [*catalog, entry]}, ensure_ascii=False
                    )
                )
                > 4096
            ):
                continue
            packed.append(item["id"])
            catalog.append(entry)
            spans.extend(ss)
            aliases.update({a["id"]: s.span_id for a, s in zip(selectors, ss, strict=True)})
        result["generation"]["context_chunk_ids"] = packed
        for path in result["paths"]:
            path["provided_to_model"] = set(path["source_chunks"]) <= set(packed)
        result["generation"]["context_tokens"] = self.tokens(
            json.dumps({"question": query, "evidence": catalog}, ensure_ascii=False)
        )
        if not packed:
            result["reason"] = "证据超出回答上下文预算，无法据此形成完整回答。"
            return result
        try:
            result["generation"]["model_calls"] += 1
            value = model.complete_json(
                "answer_knowledge", {"question": query, "evidence": catalog}
            )
            raw = value.get("claims", [])
            if not isinstance(raw, list):
                raise ValueError("claims must be list")
            converted = []
            for claim in raw:
                if not isinstance(claim, dict):
                    raise ValueError("invalid claim")
                refs = claim.get("support_span_ids")
                if not isinstance(refs, list) or any(not isinstance(k, str) for k in refs):
                    raise ValueError("invalid citations")
                converted.append({**claim, "support_span_ids": [aliases.get(k, k) for k in refs]})
            assembled = assemble_answer({**value, "claims": converted}, tuple(spans))
            if not assembled["gate_valid"]:
                result["reason"] = "回答的引用或不确定性检查未通过，未展示未经核实的结论。"
                result["generation"]["issues"] = assembled["gate_issues"]
                return result
            if assembled["claims"]:
                result["generation"]["model_calls"] += 1
                verdict = model.complete_json(
                    "verify_knowledge",
                    {"question": query, "claims": assembled["claims"], "evidence": catalog},
                )
                indexes = verdict.get("supported_claim_indexes")
                coverage = verdict.get("question_coverage")
                valid_indexes = (
                    isinstance(indexes, list)
                    and all(type(i) is int for i in indexes)
                    and sorted(indexes) == list(range(len(assembled["claims"])))
                )
                if (
                    verdict.get("sufficient") is not True
                    or not valid_indexes
                    or not isinstance(coverage, list)
                    or not coverage
                    or not all(
                        isinstance(x, dict)
                        and x.get("covered") is True
                        and isinstance(x.get("requirement"), str)
                        and x["requirement"].strip()
                        for x in coverage
                    )
                ):
                    result["reason"] = "现有证据未通过回答支持性或问题覆盖检查，无法给出完整结论。"
                    result["generation"]["issues"] = ["semantic_or_coverage_verification_failed"]
                    return result
            result.update(
                answer=assembled["answer"],
                claims=assembled["claims"],
                insufficient_evidence=assembled["insufficient_evidence"],
                status="insufficient_evidence"
                if assembled["insufficient_evidence"]
                else "evidence_ready",
                reason=str(value.get("abstention_reason", "")),
            )
            # Citation identity and offsets are copied from selected source spans, never the LLM.
            for claim in result["claims"]:
                for citation in claim["citations"]:
                    citation["chunk_id"] = citation["evidence_id"]
            result["generation"]["verification"] = "span_uncertainty_and_model_support_checked"
        except (LanguageModelError, ValueError, TypeError, KeyError):
            result["reason"] = "模型服务或输出校验未完成，以下仅展示检索证据，不作为已验证回答。"
        return result
