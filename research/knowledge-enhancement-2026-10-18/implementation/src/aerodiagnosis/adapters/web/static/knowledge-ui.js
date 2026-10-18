/* Shared browser and Node contracts for knowledge forms. */
(function(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.KnowledgeUI = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  'use strict';
  const normalize = value => String(value ?? '').normalize('NFKC').trim().replace(/\s+/g, ' ').toLowerCase();
  function identifiers(value) {
    const parts = String(value ?? '').split(/[,，;；\n]+/).map(normalize).filter(Boolean);
    if (parts.some(part => part.length > 120)) throw new Error('型号与阶段标识最长 120 字符。');
    return [...new Set(parts)].sort();
  }
  function context(fields) {
    const engine = normalize(fields.engine_model), phase = normalize(fields.flight_phase);
    const revision = normalize(fields.source_revision);
    if (engine.length > 120 || phase.length > 120 || revision.length > 120) throw new Error('型号与阶段标识最长 120 字符。');
    if (fields.require_known && !engine && !phase && !revision) throw new Error('严格筛选至少需要填写型号、飞行阶段或来源修订。');
    const result = {engine_model: engine || null, flight_phase: phase || null, require_known: Boolean(fields.require_known)};
    if (revision) result.source_revision = revision;
    const documentId = String(fields.document_id ?? "").trim();
    if (documentId.length > 200) throw new Error("资料标识过长。");
    if (documentId) result.document_id = documentId;
    return result;
  }
  function metadata(fields) {
    const status = fields.scope_status || 'unknown';
    if (!['unknown', 'known', 'generic'].includes(status)) throw new Error('无效的适用范围。');
    const engines = identifiers(fields.engine_models), phases = identifiers(fields.flight_phases);
    if (status === 'known' && !engines.length && !phases.length) throw new Error('明确限定范围至少需要填写一个型号或阶段。');
    if (status !== 'known' && (engines.length || phases.length)) throw new Error('未知或通用范围需要清空型号与阶段限定；不会自动更改范围。');
    const record = {schema_version: '1', source_type: String(fields.source_type || '').trim() || 'unknown',
      applicability: {scope_status: status, engine_models: engines, flight_phases: phases}};
    for (const key of ['source_url', 'source_revision', 'language', 'license_or_basis']) {
      const value = String(fields[key] || '').trim();
      if (value) record[key] = value;
    }
    if (record.source_url && !safeSourceUrl(record.source_url)) throw new Error('来源地址需要完整的 HTTP(S) URL。');
    return record;
  }
  function safeSourceUrl(value) {
    try { const url = new URL(value); return ['http:', 'https:'].includes(url.protocol) && Boolean(url.hostname) ? url.href : null; }
    catch { return null; }
  }
  function applicability(record, query) {
    if (!query.engine_model && !query.flight_phase && !query.source_revision) return 'not_checked';
    const revisionMissing = Boolean(query.source_revision && !record?.source_revision);
    if (query.source_revision && record?.source_revision && normalize(record.source_revision) !== normalize(query.source_revision)) return 'conflict';
    const scope = record?.applicability || {};
    if (scope.scope_status === 'generic') return revisionMissing ? 'unknown' : 'generic';
    if (scope.scope_status !== 'known') return 'unknown';
    let incomplete = revisionMissing;
    for (const [value, allowed] of [[query.engine_model, scope.engine_models || []], [query.flight_phase, scope.flight_phases || []]]) {
      if (!value) { if (allowed.length) incomplete = true; continue; }
      if (!allowed.length) incomplete = true;
      else if (!allowed.map(normalize).includes(value)) return 'conflict';
    }
    return incomplete ? 'unknown' : 'matched';
  }
  function filterDocuments(documents, query) {
    return documents.filter(doc => {
      if (query.document_id && doc.document_id !== query.document_id) return false;
      const active = doc.versions.find(version => version.version_id === doc.active_version_id);
      const status = applicability(active?.metadata, query);
      return status !== 'conflict' && (!query.require_known || status === 'matched');
    });
  }
  function activeMetadata(document) {
    return document?.versions.find(version => version.version_id === document.active_version_id)?.metadata ||
      {schema_version: '1', source_type: 'unknown', applicability: {scope_status: 'unknown', engine_models: [], flight_phases: []}};
  }
  const labels = {not_checked: '未提供查询条件', matched: '适用条件匹配', generic: '来源声明通用', unknown: '适用性未知', conflict: '适用条件冲突'};
  const statusLabel = status => labels[status] || labels.unknown;
  return {normalize, identifiers, context, metadata, safeSourceUrl, applicability, filterDocuments, activeMetadata, statusLabel};
});
