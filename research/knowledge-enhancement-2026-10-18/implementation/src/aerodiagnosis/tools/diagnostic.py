"""Manual, graph, case and parameter evidence tools with one typed surface."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Mapping
from typing import Any

from aerodiagnosis.application.fault_graph import (
    FaultGraphPath,
    FaultGraphReasoner,
    graph_edge_is_active,
)
from aerodiagnosis.application.knowledge_applicability import (
    POLICY_ID,
    assess_applicability,
    assess_assertion_applicability,
    permits,
)
from aerodiagnosis.domain import (
    DiagnosisCommand,
    EvidenceItem,
    FusedEvidenceHit,
    HybridRetrievalResult,
    RetrievalContribution,
    RetrievalRouteSummary,
    RetrievalStrategy,
    SourceKind,
    make_evidence_id,
)
from aerodiagnosis.ports import (
    CaseStore,
    GraphNode,
    GraphStore,
    VectorChunk,
    VectorMatch,
    VectorStore,
)
from aerodiagnosis.retrieval.chinese import normalize_query, relevance
from aerodiagnosis.retrieval.reranker import LexicalReranker, RankedText, Reranker

MANUAL_TOOL = "search_manual_chunks"
GRAPH_TOOL = "traverse_fault_graph"
CASE_TOOL = "find_similar_cases"
PARAMETER_TOOL = "analyze_gas_path_parameters"
HYBRID_TOOL = "hybrid_retrieve_evidence"
HYBRID_ALGORITHM = "weighted_rrf@2"

ExternalToolHandler = Callable[[DiagnosisCommand, str, frozenset[str]], tuple[EvidenceItem, ...]]

_ROUTE_WEIGHTS = {"manual": 1.0, "graph": 0.88, "case": 0.94}
_RRF_K = 60


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _keywords(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).lower()
    words = re.findall(r"[a-z0-9_]{2,}|[\u3400-\u9fff]{2,}", normalized)
    return tuple(dict.fromkeys((normalized.strip(), *words)))


class DiagnosticToolset:
    """Routes typed commands to deterministic evidence providers."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        graph_store: GraphStore,
        case_store: CaseStore,
        external_tools: Mapping[str, ExternalToolHandler] | None = None,
        reranker: Reranker | None = None,
        version_metadata: Callable[[str], Mapping[str, Any] | None] | None = None,
        version_document_id: Callable[[str], str | None] | None = None,
        source_guided_graph: bool = False,
        source_reranker: Reranker | None = None,
        text_hybrid_only: bool = False,
    ) -> None:
        self._vector = vector_store
        self._graph = graph_store
        self._cases = case_store
        self._external_tools = dict(external_tools or {})
        self._reranker = reranker or LexicalReranker()
        self._version_metadata = version_metadata
        self._version_document_id = version_document_id
        self._source_guided_graph = source_guided_graph
        self._source_reranker = source_reranker
        self._text_hybrid_only = text_hybrid_only

    @property
    def available_tools(self) -> tuple[str, ...]:
        return (
            HYBRID_TOOL,
            MANUAL_TOOL,
            *((GRAPH_TOOL, CASE_TOOL) if not self._text_hybrid_only else ()),
            PARAMETER_TOOL,
            *sorted(self._external_tools),
        )

    @property
    def backend_identity(self) -> tuple[str, ...]:
        return (
            "text_hybrid_only@1" if self._text_hybrid_only else "all_evidence_routes@1",
            self._vector.backend_name,
            self._vector.embedding_identity,
            self._graph.backend_name,
            self._cases.backend_name,
            ("grounded_source_rank@4" if self._source_reranker else "grounded_source_rank@3")
            if self._source_guided_graph
            else HYBRID_ALGORITHM,
            "range_check@1",
            POLICY_ID,
            self._reranker.identity,
        )

    def execute(
        self,
        tool_name: str,
        *,
        command: DiagnosisCommand,
        query: str,
        active_version_ids: frozenset[str],
    ) -> tuple[EvidenceItem, ...]:
        if tool_name in self._external_tools:
            return tuple(
                scoped
                for item in self._external_tools[tool_name](command, query, active_version_ids)
                if (
                    scoped := self._scope_item(
                        item, command, active_version_ids=active_version_ids, external=True
                    )
                )
                is not None
            )
        if tool_name == MANUAL_TOOL:
            return self._manual(command, query, active_version_ids)
        if tool_name == GRAPH_TOOL:
            return self._graph_evidence(command, query, active_version_ids)
        if tool_name == CASE_TOOL:
            return self._case_evidence(command, query)
        if tool_name == PARAMETER_TOOL:
            return self._parameter_evidence(command)
        if tool_name == HYBRID_TOOL:
            return tuple(
                hit.evidence
                for hit in self.hybrid_search(
                    command=command,
                    query=query,
                    active_version_ids=active_version_ids,
                ).hits
            )
        raise KeyError(f"unknown diagnostic tool: {tool_name}")

    def hybrid_search(
        self,
        *,
        command: DiagnosisCommand,
        query: str,
        active_version_ids: frozenset[str],
        strategy: RetrievalStrategy = RetrievalStrategy.WEIGHTED_RRF,
        rrf_k: int = _RRF_K,
        route_weights: Mapping[str, float] | None = None,
        enforce_route_coverage: bool = True,
    ) -> HybridRetrievalResult:
        """Rank evidence with one production policy or a controlled experiment policy."""

        cleaned = normalize_query(query.strip()) if self._source_guided_graph else query.strip()
        if not cleaned:
            raise ValueError("hybrid retrieval query must not be empty")
        if not 1 <= rrf_k <= 1000:
            raise ValueError("rrf_k must be between 1 and 1000")
        weights = dict(_ROUTE_WEIGHTS if route_weights is None else route_weights)
        if set(weights) != set(_ROUTE_WEIGHTS) or any(
            not 0 < value <= 1 for value in weights.values()
        ):
            raise ValueError("route_weights must define manual, graph and case in (0, 1]")
        candidate_limit = min(20, max(command.top_k, command.top_k * 3))
        expanded = command.model_copy(update={"top_k": candidate_limit})
        text_only = self._text_hybrid_only and strategy not in {
            RetrievalStrategy.GRAPH_ONLY,
            RetrievalStrategy.CASE_ONLY,
        }
        all_route_items = {
            "manual": self._manual(expanded, cleaned, active_version_ids),
            "graph": ()
            if text_only
            else self._graph_evidence(expanded, cleaned, active_version_ids),
            "case": () if text_only else self._case_evidence(expanded, cleaned),
        }
        selected_route = {
            RetrievalStrategy.MANUAL_ONLY: "manual",
            RetrievalStrategy.GRAPH_ONLY: "graph",
            RetrievalStrategy.CASE_ONLY: "case",
        }.get(strategy)
        if text_only:
            selected_route = "manual"
        route_items = (
            {selected_route: all_route_items[selected_route]}
            if selected_route is not None
            else all_route_items
        )
        algorithm = (
            ("grounded_source_rank@4" if self._source_reranker else "grounded_source_rank@3")
            if self._source_guided_graph and selected_route is None
            else f"{strategy.value}@{2 if strategy is RetrievalStrategy.WEIGHTED_RRF else 1}"
        )
        if text_only:
            algorithm = "text_hybrid_preserve_manual_rank@1"
        candidates: dict[str, tuple[EvidenceItem, list[RetrievalContribution]]] = {}
        for route, items in route_items.items():
            for route_rank, evidence in enumerate(items, start=1):
                normalized = max(0.0, min(1.0, evidence.score))
                contribution = RetrievalContribution(
                    route=route,
                    route_rank=route_rank,
                    raw_score=evidence.score,
                    normalized_score=normalized,
                    reciprocal_rank_score=(rrf_k + 1) / (rrf_k + route_rank),
                    weight=1.0 if strategy is RetrievalStrategy.RRF else weights[route],
                )
                existing = candidates.get(evidence.evidence_id)
                if existing is None:
                    candidates[evidence.evidence_id] = (evidence, [contribution])
                else:
                    if "graph_guidance" in evidence.locator:
                        prior = existing[0].model_copy(
                            update={
                                "locator": {
                                    **existing[0].locator,
                                    "graph_guidance": evidence.locator["graph_guidance"],
                                }
                            }
                        )
                        candidates[evidence.evidence_id] = (prior, existing[1])
                    existing[1].append(contribution)

        scored: list[tuple[EvidenceItem, float, tuple[RetrievalContribution, ...]]] = []
        for evidence, contribution_list in candidates.values():
            contributions = tuple(contribution_list)
            if selected_route is not None:
                fused_score = contributions[0].normalized_score
            elif strategy is RetrievalStrategy.NORMALIZED_SCORE:
                fused_score = max(item.weight * item.normalized_score for item in contributions)
            else:
                fused_score = min(
                    1.0,
                    sum(item.weight * item.reciprocal_rank_score for item in contributions),
                )
            if self._source_guided_graph and selected_route is None:
                text = str(evidence.locator.get("display_text_zh", evidence.excerpt))
                manual_rank = next(
                    (part.route_rank for part in contributions if part.route == "manual"), 100
                )
                lexical = relevance(cleaned, text)
                rank_signal = (rrf_k + 1) / (rrf_k + manual_rank)
                graph_signal = next(
                    (part.normalized_score for part in contributions if part.route == "graph"), 0.0
                )
                fused_score = min(1.0, 0.65 * lexical + 0.25 * rank_signal + 0.08 * graph_signal)
            scored.append((evidence, fused_score, contributions))
        scored.sort(
            key=lambda item: (
                (item[2][0].route_rank, item[0].evidence_id)
                if text_only
                else (-item[1], item[0].evidence_id)
            )
        )
        if self._source_guided_graph and self._source_reranker is not None and not text_only:
            by_id = {item[0].evidence_id: item for item in scored}
            ordered = self._source_reranker.rerank(
                cleaned,
                [
                    RankedText(
                        e.evidence_id, str(e.locator.get("display_text_zh", e.excerpt)), score
                    )
                    for e, score, _parts in scored
                ],
            )
            scored = [
                (by_id[item.key][0], 1 / (rank + 1), by_id[item.key][2])
                for rank, item in enumerate(ordered)
            ]
        if self._source_guided_graph:
            unique: dict[
                tuple[str, str], tuple[EvidenceItem, float, tuple[RetrievalContribution, ...]]
            ] = {}
            for evidence, score, parts in scored:
                key = (
                    " ".join(evidence.excerpt.split()).casefold(),
                    json.dumps(
                        {
                            "version_id": evidence.version_id,
                            "document_id": evidence.document_id,
                            **{
                                k: evidence.document_metadata.get(k)
                                for k in ["applicability", "source_revision", "source_type"]
                            },
                        },
                        sort_keys=True,
                    ),
                )
                if key in unique:
                    prior, prior_score, prior_parts = unique[key]
                    aliases = list(prior.locator.get("duplicate_sources", []))
                    aliases.append(
                        {
                            "evidence_id": evidence.evidence_id,
                            "document_id": evidence.document_id,
                            "version_id": evidence.version_id,
                        }
                    )
                    unique[key] = (
                        prior.model_copy(
                            update={"locator": {**prior.locator, "duplicate_sources": aliases}}
                        ),
                        prior_score,
                        prior_parts,
                    )
                    continue
                unique[key] = (evidence, score, parts)
            scored = list(unique.values())

        selected: list[tuple[EvidenceItem, float, tuple[RetrievalContribution, ...], str]] = []
        selected_ids: set[str] = set()
        coverage_enabled = (
            enforce_route_coverage
            and not self._source_guided_graph
            and selected_route is None
            and command.top_k >= len(route_items)
        )
        if coverage_enabled:
            coverage = []
            for route in route_items:
                best = next(
                    (item for item in scored if any(part.route == route for part in item[2])),
                    None,
                )
                if best is not None:
                    coverage.append(best)
            coverage.sort(key=lambda item: (-item[1], item[0].evidence_id))
            for evidence, score, contributions in coverage:
                if evidence.evidence_id in selected_ids or len(selected) >= command.top_k:
                    continue
                selected.append((evidence, score, contributions, "route_coverage"))
                selected_ids.add(evidence.evidence_id)
        for evidence, score, contributions in scored:
            if evidence.evidence_id in selected_ids or len(selected) >= command.top_k:
                continue
            selected.append((evidence, score, contributions, "fused_score"))
            selected_ids.add(evidence.evidence_id)

        hits = []
        for final_rank, (evidence, score, contributions, reason) in enumerate(selected, start=1):
            retrieval = {
                "algorithm": algorithm,
                "rrf_k": rrf_k,
                "selection_reason": reason,
                "contributions": [item.model_dump(mode="json") for item in contributions],
            }
            ranked_evidence = evidence.model_copy(
                update={"score": score, "locator": {**evidence.locator, "retrieval": retrieval}}
            )
            hits.append(
                FusedEvidenceHit(
                    rank=final_rank,
                    evidence=ranked_evidence,
                    fused_score=score,
                    selection_reason=reason,
                    contributions=contributions,
                )
            )
        included_by_route = {
            route: sum(any(part.route == route for part in hit.contributions) for hit in hits)
            for route in all_route_items
        }
        routes = tuple(
            RetrievalRouteSummary(
                route=route,
                candidate_count=len(items),
                included_count=included_by_route[route],
                top_raw_score=max((item.score for item in items), default=None),
            )
            for route, items in all_route_items.items()
        )
        return HybridRetrievalResult(
            query=cleaned,
            algorithm=algorithm,
            top_k=command.top_k,
            rrf_k=rrf_k,
            route_weights=weights,
            route_coverage_enabled=coverage_enabled,
            candidate_count=len(candidates),
            hits=tuple(hits),
            routes=routes,
            knowledge_context=command.knowledge_context,
            scope_filter_policy=POLICY_ID,
            scope_notes=("Case filtering scans at most 200 ranked candidates.",)
            if command.knowledge_context.has_constraints
            else (),
        )

    def _eligible_versions(
        self, command: DiagnosisCommand, version_ids: frozenset[str]
    ) -> frozenset[str]:
        context = command.knowledge_context
        if context.document_id is not None:
            if self._version_document_id is None:
                return frozenset()
            version_ids = frozenset(
                v for v in version_ids if self._version_document_id(v) == context.document_id
            )
        if not context.has_constraints or self._version_metadata is None:
            return version_ids
        return frozenset(
            version_id
            for version_id in version_ids
            if permits(assess_applicability(self._version_metadata(version_id), context), context)
        )

    def _scope_item(
        self,
        item: EvidenceItem,
        command: DiagnosisCommand,
        *,
        active_version_ids: frozenset[str] | None = None,
        external: bool = False,
    ) -> EvidenceItem | None:
        if item.excerpt_truncated:
            return None
        if (
            command.knowledge_context.document_id is not None
            and item.source_kind != SourceKind.PARAMETER_ANALYSIS
            and item.document_id != command.knowledge_context.document_id
        ):
            return None
        document_backed = item.source_kind in {
            SourceKind.DOCUMENT_CHUNK,
            SourceKind.KNOWLEDGE_GRAPH_PATH,
        }
        if (
            document_backed
            and command.knowledge_context.document_id is not None
            and (
                self._version_document_id is None
                or self._version_document_id(item.version_id)
                != command.knowledge_context.document_id
            )
        ):
            return None
        if document_backed and (
            active_version_ids is not None and item.version_id not in active_version_ids
        ):
            return None
        metadata = item.document_metadata
        if document_backed and self._version_metadata is not None:
            metadata = dict(self._version_metadata(item.version_id) or {})
        elif external:
            # A plugin's self-reported scope has not passed the local document review.
            metadata = {}
        status = assess_applicability(metadata, command.knowledge_context)
        if item.source_kind == SourceKind.KNOWLEDGE_GRAPH_PATH:
            properties = item.locator.get("coordinates", {}).get("properties", {})
            if isinstance(properties, Mapping):
                status = assess_assertion_applicability(
                    properties, metadata, command.knowledge_context
                )
        if not permits(status, command.knowledge_context):
            return None
        return item.model_copy(
            update={"document_metadata": metadata, "applicability_status": status}
        )

    def _manual(
        self,
        command: DiagnosisCommand,
        query: str,
        active_version_ids: frozenset[str],
    ) -> tuple[EvidenceItem, ...]:
        eligible_versions = self._eligible_versions(command, active_version_ids)
        candidate_limit = (
            200
            if command.knowledge_context.has_constraints and self._version_metadata is None
            else command.top_k
        )
        matches = self._vector.search(
            query,
            top_k=candidate_limit,
            active_version_ids=eligible_versions,
        )
        matches = [match for match in matches if match.chunk.version_id in eligible_versions]
        if not getattr(self._vector, "handles_reranking", False):
            match_by_id = {match.chunk.chunk_id: match for match in matches}
            reranked = self._reranker.rerank(
                query,
                [
                    RankedText(
                        key=match.chunk.chunk_id,
                        text=match.chunk.content,
                        score=match.score,
                    )
                    for match in matches
                ],
            )
            matches = [
                VectorMatch(chunk=match_by_id[item.key].chunk, score=item.score)
                for item in reranked
            ]
        evidence = []
        for match in matches:
            item = self._manual_item(match, command.min_relevance)
            if item is not None:
                scoped = self._scope_item(item, command)
                if scoped is not None:
                    evidence.append(scoped)
        return tuple(evidence[: command.top_k])

    @staticmethod
    def _manual_item(match: VectorMatch, min_relevance: float) -> EvidenceItem | None:
        if match.score < min_relevance or len(match.chunk.content) > 8000:
            return None
        metadata: Mapping[str, Any] = match.chunk.metadata
        locator_value = metadata.get(
            "locator",
            {"kind": "chunk", "coordinates": {"chunk_id": match.chunk.chunk_id}},
        )
        if not isinstance(locator_value, Mapping):
            return None
        locator = dict(locator_value)
        # Selection audit is display metadata, not part of immutable evidence identity.
        retrieval = metadata.get("retrieval")
        if isinstance(retrieval, Mapping):
            locator["manual_retrieval"] = dict(retrieval)
        evidence_id = metadata.get("evidence_id")
        if not isinstance(evidence_id, str):
            evidence_id = make_evidence_id(
                SourceKind.DOCUMENT_CHUNK,
                match.chunk.chunk_id,
                dict(locator_value),
            )
        if metadata.get("source_boundary_warning"):
            locator["source_boundary_warning"] = metadata["source_boundary_warning"]
        if metadata.get("source_integrity_warning"):
            locator["source_integrity_warning"] = metadata["source_integrity_warning"]
        content_hash = metadata.get("content_hash")
        if not isinstance(content_hash, str):
            content_hash = _hash(match.chunk.content)
        if isinstance(metadata.get("display_content_zh"), str):
            locator["display_text_zh"] = metadata["display_content_zh"]
        try:
            return EvidenceItem(
                evidence_id=evidence_id,
                source_kind=SourceKind.DOCUMENT_CHUNK,
                source_ref=match.chunk.chunk_id,
                document_id=match.chunk.document_id,
                version_id=match.chunk.version_id,
                content_hash=content_hash,
                excerpt=match.chunk.content.strip(),
                locator=locator,
                score=match.score,
                document_metadata=dict(metadata.get("document_metadata", {}))
                if isinstance(metadata.get("document_metadata"), Mapping)
                else {},
            )
        except ValueError:
            return None

    def _guided_graph_evidence(
        self, command: DiagnosisCommand, query: str, active_version_ids: frozenset[str]
    ) -> tuple[EvidenceItem, ...]:
        """Use grounded graph matches to fetch full source children, never synthetic prose."""
        from aerodiagnosis.retrieval.concepts import concept_record, guided_edges

        nodes, edges = [], []
        for version in sorted(active_version_ids):
            version_nodes, version_edges = self._graph.snapshot(
                active_version_ids=frozenset({version}), limit=2000
            )
            if len(version_nodes) >= 2000:
                raise ValueError("Source graph snapshot may be truncated")
            nodes.extend(version_nodes)
            edges.extend(version_edges)
        membership_getter = getattr(self._graph, "concept_memberships", None)
        memberships = (
            membership_getter(active_version_ids=active_version_ids)
            if membership_getter is not None
            else [concept_record(node) for node in nodes]
        )
        getter = getattr(self._vector, "get_chunk", None)
        if getter is None:
            return ()
        candidates: dict[str, tuple[VectorChunk, float, list[str]]] = {}
        eligible_edges = [
            edge
            for edge in edges
            if graph_edge_is_active(edge)
            and edge.version_id in active_version_ids
            and permits(
                assess_assertion_applicability(
                    edge.properties,
                    self._version_metadata(edge.version_id)
                    if self._version_metadata
                    else edge.properties.get("document_metadata"),
                    command.knowledge_context,
                ),
                command.knowledge_context,
            )
        ]
        for edge, signal, _concept_ids in guided_edges(query, nodes, eligible_edges, memberships):
            if not graph_edge_is_active(edge) or edge.version_id not in active_version_ids:
                continue
            if not permits(
                assess_assertion_applicability(
                    edge.properties,
                    self._version_metadata(edge.version_id)
                    if self._version_metadata
                    else edge.properties.get("document_metadata"),
                    command.knowledge_context,
                ),
                command.knowledge_context,
            ):
                continue
            chunk = getter(edge.properties.get("chunk_id", ""))
            if (
                chunk is None
                or chunk.version_id not in active_version_ids
                or chunk.version_id != edge.version_id
            ):
                continue
            quote = edge.properties.get("evidence_text", "")
            if not quote or quote not in chunk.content:
                continue
            score = relevance(query, str(chunk.metadata.get("display_content_zh", chunk.content)))
            if score <= 0:
                continue
            score = min(1.0, 0.85 * score + 0.15 * signal)
            current = candidates.get(chunk.chunk_id)
            if current is None:
                candidates[chunk.chunk_id] = (chunk, score, [edge.edge_id])
            else:
                candidates[chunk.chunk_id] = (chunk, max(score, current[1]), current[2])
                current[2].append(edge.edge_id)
        output = []
        for chunk, score, edge_ids in sorted(
            candidates.values(), key=lambda row: (-row[1], row[0].chunk_id)
        ):
            item = self._manual_item(VectorMatch(chunk, score), 0)
            if item is None:
                continue
            item = item.model_copy(
                update={
                    "locator": {
                        **item.locator,
                        "graph_guidance": {
                            "edge_ids": sorted(set(edge_ids)),
                            "policy": "source-scoped-concepts@1",
                            "version_id": chunk.version_id,
                        },
                    }
                }
            )
            scoped = self._scope_item(item, command, active_version_ids=active_version_ids)
            if scoped is not None:
                output.append(scoped)
        return tuple(output[: command.top_k])

    def _graph_evidence(
        self,
        command: DiagnosisCommand,
        query: str,
        active_version_ids: frozenset[str],
    ) -> tuple[EvidenceItem, ...]:
        active_version_ids = self._eligible_versions(command, active_version_ids)
        if self._source_guided_graph:
            return self._guided_graph_evidence(command, query, active_version_ids)
        causal_paths = [
            path
            for path in FaultGraphReasoner(self._graph).reason(
                query,
                active_version_ids=active_version_ids,
                limit=200 if command.knowledge_context.has_constraints else command.top_k,
            )
            if (path.causes or path.solutions or path.checks)
            and self._scope_item(self._causal_path_item(path), command) is not None
        ]
        if causal_paths:
            return tuple(
                scoped
                for path in causal_paths[: command.top_k]
                if (scoped := self._scope_item(self._causal_path_item(path), command)) is not None
            )

        nodes: dict[str, GraphNode] = {}
        for keyword in _keywords(query):
            for node in self._graph.search_nodes(
                keyword, limit=200 if command.knowledge_context.has_constraints else command.top_k
            ):
                raw_metadata = node.properties.get("document_metadata")
                metadata = raw_metadata if isinstance(raw_metadata, Mapping) else None
                if self._version_metadata is not None:
                    metadata = self._version_metadata(node.version_id)
                if node.version_id in active_version_ids and permits(
                    assess_applicability(metadata, command.knowledge_context),
                    command.knowledge_context,
                ):
                    nodes.setdefault(node.node_id, node)
            if len(nodes) >= command.top_k:
                break
        evidence = []
        for node in list(nodes.values())[: command.top_k]:
            paths = self._graph.neighbors(
                node.node_id,
                max_depth=2,
                active_version_ids=active_version_ids,
            )
            paths = [
                path
                for path in paths
                if graph_edge_is_active(path.edge)
                and permits(
                    assess_assertion_applicability(
                        path.edge.properties,
                        self._version_metadata(path.edge.version_id)
                        if self._version_metadata
                        else path.edge.properties.get("document_metadata"),
                        command.knowledge_context,
                    ),
                    command.knowledge_context,
                )
                and path.edge.version_id in active_version_ids
                and path.source.version_id in active_version_ids
                and path.target.version_id in active_version_ids
            ]
            if not paths and (
                node.properties.get("extraction") == "lexical@2"
                or str(node.properties.get("extraction", "")).startswith("langextract@")
            ):
                # A mention from a negated/ambiguous sentence is not a supported claim.
                continue
            if paths:
                path = paths[0]
                endpoints = {path.source.node_id: path.source, path.target.node_id: path.target}
                edge_source = endpoints[path.edge.source_id]
                edge_target = endpoints[path.edge.target_id]
                excerpt = (
                    f"{edge_source.name} --{path.edge.relation}--> {edge_target.name}: "
                    f"{edge_target.description}"
                )
                source_ref = path.edge.edge_id
                locator = {
                    "kind": "graph_path",
                    "coordinates": {
                        "source": path.edge.source_id,
                        "edge": path.edge.edge_id,
                        "target": path.edge.target_id,
                        "depth": path.depth,
                        "version_id": path.edge.version_id,
                        "properties": dict(path.edge.properties),
                    },
                }
            else:
                excerpt = f"{node.name}: {node.description}"
                source_ref = node.node_id
                locator = {"kind": "graph_node", "coordinates": {"node_id": node.node_id}}
            evidence_id = make_evidence_id(
                SourceKind.KNOWLEDGE_GRAPH_PATH,
                source_ref,
                locator,
            )
            evidence.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    source_kind=SourceKind.KNOWLEDGE_GRAPH_PATH,
                    source_ref=source_ref,
                    document_id=node.source_ref,
                    version_id=node.version_id,
                    content_hash=_hash(excerpt),
                    excerpt=excerpt[:8000],
                    excerpt_truncated=len(excerpt) > 8000,
                    locator=locator,
                    score=1.0,
                    document_metadata=dict(node.properties.get("document_metadata", {}))
                    if isinstance(node.properties.get("document_metadata"), Mapping)
                    else {},
                )
            )
        return tuple(
            scoped for item in evidence if (scoped := self._scope_item(item, command)) is not None
        )

    @staticmethod
    def _causal_path_item(path: FaultGraphPath) -> EvidenceItem:
        parts = [f"Symptom term: {path.symptom.name}"]
        if path.component is not None:
            parts.append(f"{path.component.name} has symptom")
        for cause in path.causes:
            parts.append(f"Candidate cause: {cause.name}")
        for solution in path.solutions:
            parts.append(f"Possible mitigation: {solution.name}")
        for check in path.checks:
            parts.append(f"Suggested inspection: {check.name}")
        excerpt = "; ".join(parts)
        source_ref = path.symptom.node_id
        locator = {
            "kind": "causal_graph_path",
            "coordinates": {
                "symptom": path.symptom.node_id,
                "component": path.component.node_id if path.component else None,
                "equipment": path.equipment.node_id if path.equipment else None,
                "causes": [cause.node_id for cause in path.causes],
                "solutions": [solution.node_id for solution in path.solutions],
                "checks": [check.node_id for check in path.checks],
                "hops": path.hops,
                "evidence_edges": [
                    {
                        "edge_id": edge.edge_id,
                        "version_id": edge.version_id,
                        "relation": edge.relation,
                        "properties": dict(edge.properties),
                    }
                    for edge in path.evidence_edges
                ],
            },
        }
        return EvidenceItem(
            evidence_id=make_evidence_id(
                SourceKind.KNOWLEDGE_GRAPH_PATH,
                source_ref,
                locator,
            ),
            source_kind=SourceKind.KNOWLEDGE_GRAPH_PATH,
            source_ref=source_ref,
            document_id=path.symptom.source_ref,
            version_id=path.symptom.version_id,
            content_hash=_hash(excerpt),
            excerpt=excerpt[:8000],
            excerpt_truncated=len(excerpt) > 8000,
            locator=locator,
            score=path.score,
            document_metadata=dict(path.symptom.properties.get("document_metadata", {}))
            if isinstance(path.symptom.properties.get("document_metadata"), Mapping)
            else {},
        )

    def _case_evidence(
        self,
        command: DiagnosisCommand,
        query: str,
    ) -> tuple[EvidenceItem, ...]:
        evidence = []
        scan_limit = 200 if command.knowledge_context.has_constraints else command.top_k
        for match in self._cases.search(query, limit=scan_limit):
            if len(match.case.summary) > 8000:
                continue
            locator = {
                "kind": "case_version",
                "coordinates": {"case_id": match.case.case_id, "version": match.case.version},
            }
            source_ref = f"{match.case.case_id}:{match.case.version}"
            evidence.append(
                EvidenceItem(
                    evidence_id=make_evidence_id(SourceKind.CASE, source_ref, locator),
                    source_kind=SourceKind.CASE,
                    source_ref=source_ref,
                    document_id=match.case.case_id,
                    version_id=str(match.case.version),
                    content_hash=_hash(match.case.summary),
                    excerpt=match.case.summary,
                    locator=locator,
                    score=match.score,
                    document_metadata=dict(match.case.attributes.get("document_metadata", {}))
                    if isinstance(match.case.attributes.get("document_metadata"), Mapping)
                    else {},
                )
            )
        return tuple(
            scoped for item in evidence if (scoped := self._scope_item(item, command)) is not None
        )[: command.top_k]

    @staticmethod
    def _parameter_evidence(command: DiagnosisCommand) -> tuple[EvidenceItem, ...]:
        evidence = []
        for observation in command.parameters:
            if observation.value < observation.expected_min:
                state = "below"
            elif observation.value > observation.expected_max:
                state = "above"
            else:
                state = "within"
            payload = observation.model_dump(mode="json")
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            source_ref = _hash(canonical)
            locator = {
                "kind": "parameter_observation",
                "coordinates": {"name": observation.name},
            }
            excerpt = (
                f"{observation.name}={observation.value:g}{observation.unit} is {state} "
                f"the supplied range [{observation.expected_min:g}, "
                f"{observation.expected_max:g}]{observation.unit}."
            )
            evidence.append(
                EvidenceItem(
                    evidence_id=make_evidence_id(
                        SourceKind.PARAMETER_ANALYSIS,
                        source_ref,
                        locator,
                    ),
                    source_kind=SourceKind.PARAMETER_ANALYSIS,
                    source_ref=source_ref,
                    document_id="user-supplied-parameters",
                    version_id="range_check@1",
                    content_hash=_hash(excerpt),
                    excerpt=excerpt,
                    locator=locator,
                    score=1.0,
                )
            )
        return tuple(evidence)
