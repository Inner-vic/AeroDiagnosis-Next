# AeroDiagnosis Project Polish Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn AeroDiagnosis from a working diagnostic Agent into a technically richer, evidence-first, observable, and evaluable system without breaking the existing v3 architecture.

**Architecture:** Each phase adds an adapter or use case behind existing ports. Ingestion adds document parsers; retrieval adds embedding/rerank adapters; frontend consumes existing SSE and audit APIs; evaluation adds domain metrics; graph and production phases add observability and governance.

**Tech Stack:** Python 3.13, Pydantic, FastAPI, Docker Compose, pytest, Neo4j, ChromaDB, DeepSeek-compatible LLM.

## Global Constraints

- Preserve `external-primary` production mode.
- Keep SQLite as metadata/case/session/checkpoint store unless a phase explicitly changes it.
- Do not expose API keys or operator tokens.
- Every phase must pass `pytest`, `ruff`, and `mypy`.
- Keep commits independent and recoverable.

---

## Phase 1: Structured Document Ingestion

Support PDF, DOCX, XLSX, and HTML in the same immutable-version pipeline as TXT/Markdown/CSV.

**Tasks:**

- [ ] Add `PdfParser` using `pypdf`, extracting page-scoped chunks.
- [ ] Add `DocxParser` using `python-docx`, preserving headings and tables.
- [ ] Add `XlsxParser` using `openpyxl`, preserving sheet and row locators.
- [ ] Add `HtmlParser`, preserving headings, paragraphs, and tables without unsafe script content.
- [ ] Register new parsers in `ParserRegistry` and update public API.
- [ ] Add parser tests and update supported-extension assertions.
- [ ] Rebuild Docker image and verify upload workflow.

## Phase 2: Retrieval Quality

Replace hashing-only baseline with optional semantic and hybrid retrieval.

**Tasks:**

- [ ] Add configurable embedding provider port.
- [ ] Add dense embedding adapter with local sentence-transformers-compatible model.
- [ ] Add sparse lexical + dense vector hybrid retrieval.
- [ ] Add cross-encoder rerank adapter.
- [ ] Persist embedding identity in run snapshots.
- [ ] Add evaluation for retrieval quality and embedding reproducibility.

## Phase 3: Agent Experience and Frontend Audit

Expose backend Agent events and audit traces in the UI.

**Tasks:**

- [ ] Consume `/api/diagnoses/stream` in the diagnosis UI.
- [ ] Render tool-call and verifier timeline.
- [ ] Add run operation history panel.
- [ ] Add case verification controls.
- [ ] Render capability boundaries.

## Phase 4: Evaluation and Domain Benchmarks

Make diagnosis quality measurable.

**Tasks:**

- [ ] Add RAGAS-style faithfulness, context precision, and recall metrics.
- [ ] Add PDF/table parsing quality cases.
- [ ] Add frozen cross-document causal-chain cases.
- [ ] Record latency, token, and evidence-coverage metrics.
- [ ] Add LLM-as-judge with human-review markers.

## Phase 5: Knowledge Graph Enhancement

Move Neo4j from visualization to causal reasoning.

**Tasks:**

- [ ] Add automatic entity/relation extraction adapter.
- [ ] Add entity alignment and synonym normalization.
- [ ] Add temporal graph attributes.
- [ ] Add multi-hop causal-path scoring.
- [ ] Preserve provenance and active-version filtering.

## Phase 6: Production Governance and Observability

Make the system auditable and operable.

**Tasks:**

- [ ] Add OpenTelemetry traces and metrics.
- [ ] Add rate limiting and concurrent run controls.
- [ ] Add model-call budget and deadline enforcement.
- [ ] Add backup/restore verification scripts.
- [ ] Add role-based read/write boundaries.

## Execution Note

Phases are independent enough to be reviewed separately. Implement from Phase 1 forward, committing after every completed phase. Do not begin a later phase while an earlier phase has failing tests.
