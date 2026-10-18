const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const staticRoot = path.resolve(process.argv[2]);
const graphView = require(path.join(staticRoot, 'graph-view.js'));
const ui = require(path.join(staticRoot, 'knowledge-ui.js'));

async function main() {
  const html = fs.readFileSync(path.join(staticRoot, 'index.html'), 'utf8');
  assert.match(html, /仅使用适用条件明确且匹配的证据/);
  assert.match(html, /仅保留适用条件明确且匹配的活动版本/);
  assert.doesNotMatch(html, /所有限定条件已核验/);
  assert.equal(ui.statusLabel('not_checked'), '未提供查询条件');
  assert.equal(ui.statusLabel('matched'), '适用条件匹配');
  assert.deepEqual(ui.metadata({}), {schema_version: '1', source_type: 'unknown',
    applicability: {scope_status: 'unknown', engine_models: [], flight_phases: []}});
  assert.throws(() => ui.metadata({scope_status: 'known'}), /至少/);
  assert.throws(() => ui.metadata({scope_status: 'generic', engine_models: 'engine-a'}), /清空/);
  assert.throws(() => ui.metadata({scope_status: 'unknown', flight_phases: 'cruise'}), /清空/);
  assert.throws(() => ui.metadata({source_url: 'javascript:alert(1)'}), /HTTP/);
  assert.throws(() => ui.context({require_known: true}), /至少/);
  assert.equal(ui.context({document_id:' Case-Sensitive-ID '}).document_id,'Case-Sensitive-ID');
  assert.deepEqual(ui.filterDocuments([{document_id:'A',versions:[]},{document_id:'B',versions:[]}],ui.context({document_id:'B'})).map(d=>d.document_id),['B']);
  assert.match(html,/id="qaDocument"/);
  assert.match(html,/id="catalogDocument"/);
  assert.deepEqual(ui.identifiers(' Engine-Ａ , engine-a，engine-b;engine-c'), ['engine-a', 'engine-b', 'engine-c']);
  const scoped = ui.metadata({scope_status: 'known', engine_models: 'engine-a', flight_phases: 'cruise'});
  assert.equal(ui.applicability(scoped, ui.context({engine_model: 'engine-a'})), 'unknown');
  assert.equal(ui.applicability(scoped, ui.context({engine_model: 'engine-b'})), 'conflict');

  const revised = {...scoped, source_revision:'R2'};
  assert.equal(ui.applicability(revised, ui.context({engine_model:'engine-a', flight_phase:'cruise', source_revision:' r2 '})), 'matched');
  assert.equal(ui.applicability(revised, ui.context({engine_model:'engine-a', source_revision:'R1'})), 'conflict');
  assert.equal(ui.applicability(scoped, ui.context({engine_model:'engine-a', flight_phase:'cruise', source_revision:'R2'})), 'unknown');
  assert.equal(ui.applicability({applicability:{scope_status:'generic'},source_revision:'R1'}, ui.context({source_revision:'R2'})), 'conflict');
  assert.match(html, /id="qaRevision"/);
  assert.match(html, /id="catalogRevision"/);

  const nodes = new Map();
  function node(selector) {
    if (!nodes.has(selector)) nodes.set(selector, {value: '', checked: false, files: [], innerHTML: '', textContent: '',
      classList: {add(){}, remove(){}, toggle(){}}, addEventListener(){}, append(){}, remove(){},
      scrollIntoView(){}, querySelectorAll(){return []}, focus(){}});
    return nodes.get(selector);
  }
  const requests = [];
  const docs = [{document_id: 'logical-doc', display_name: '<manual>', active_version_id: 'active', desired_revision: 3,
    versions: [{version_id:'failed', revision:3, status:'failed', content_hash:'f', parser_version:'p',
      metadata:ui.metadata({source_revision:'failed-r3'})},
      {version_id:'active', revision:2, status:'active', content_hash:'a', parser_version:'p',
       metadata:{...scoped, source_url:'https://example.invalid/manual', source_revision:'R2'}},
      {version_id:'old', revision:1, status:'superseded', content_hash:'o', parser_version:'p', metadata:ui.metadata({})}]},
    {document_id:'unpublished',display_name:'not published',active_version_id:null,desired_revision:1,
      versions:[{version_id:'failed-only',revision:1,status:'failed',content_hash:'z',parser_version:'p',metadata:ui.metadata({})}]}];
  const sandbox = {KnowledgeUI: ui, GraphView: graphView, console, URL, TextDecoder, setTimeout, capturedDiagnosis: null,
    window: {}, document: {querySelector:node, querySelectorAll(){return []},createElement(){return node('created-'+Math.random())},documentElement:{dataset:{}}},
    localStorage:{getItem(){return null},setItem(){}},sessionStorage:{getItem(){return null},setItem(){}},
    fetch: async (url, options = {}) => {
      requests.push({url, options, body:options.body ? JSON.parse(options.body) : null});
      const result = url === '/api/system' ? {default_provider:{available:false}} :
        url === '/api/documents' && options.method === 'POST' ? {revision:4,chunk_count:1,reused:false} :
        url === '/api/documents' ? docs : url === '/api/evidence/search' ? [{document_id:'logical-doc',version_id:'active',
          evidence_id:'evidence',content:'<img src=x onerror=alert(1)>',applicability_status:'matched'}] : {};
      return {ok:true,status:200,json:async()=>result};
    }};
  vm.createContext(sandbox);
  const run = code => vm.runInContext(code, sandbox);
  run(fs.readFileSync(path.join(staticRoot, 'app.js'), 'utf8'));
  await new Promise(resolve => setImmediate(resolve));

  assert.equal(run('graphName({name:"compressor",properties:{name_zh:"压气机"}})'), '压气机');

  assert.equal(run('graphRelation("CAUSES")'), '导致');
  assert.equal(run('kindLabel("Condition")'), '条件');
  assert.doesNotMatch(html, /graphLanguage|graphPrevious|graphNext/);
  const graphFixture={nodes:[{node_id:'e',name:'发动机',kind:'Engine'},{node_id:'c',name:'压气机',kind:'Component'},{node_id:'s',name:'效率下降',kind:'Symptom'}],edges:[{source_id:'e',target_id:'c'},{source_id:'c',target_id:'s'}]};
  const hidden=graphView.filter(graphFixture,new Set(['engine']));
  assert.deepEqual(hidden.nodes.map(n=>n.node_id),['c','s']);
  assert.equal(hidden.edges.length,1);
  assert.equal(graphFixture.nodes.length,3);
  assert.equal(graphView.filter(graphFixture,new Set()).nodes.length,3);
  assert.equal(graphView.shortLabel('压气机检查与修理措施以及完整说明').length,13);
  run('renderKnowledgeGraph('+JSON.stringify(graphFixture)+')');
  run('toggleGraphKind("engine")');
  assert.equal(node('#graphNodeCount').textContent,2);
  assert.equal(node('#graphEdgeCount').textContent,1);
  assert.match(node('#graphLegend').innerHTML,/aria-pressed="false"/);
  run('toggleGraphKind("engine")');
  assert.equal(node('#graphNodeCount').textContent,3);
  assert.equal(node('#graphEdgeCount').textContent,2);

  await run('loadKnowledge()');
  assert.match(node('#documentList').innerHTML, /尚无活动版本/);
  assert.match(node('#documentList').innerHTML, /r3 · failed/);
  assert.match(node('#documentList').innerHTML, /r1 · superseded/);
  assert.match(node('#documentList').innerHTML, /&lt;manual&gt;/);
  node('#documentTarget').value = 'logical-doc';
  node('#documentTarget').onchange();
  assert.equal(node('#sourceRevision').value, 'R2');
  assert.equal(node('#documentScope').value, 'known');
  assert.equal(node('#documentPhases').value, 'cruise');
  assert.match(node('#documentTargetHint').textContent, /活动版本 r2/);
  assert.match(node('#documentTargetHint').textContent, /形成新版本/);

  node('#documentScope').value = 'generic';
  node('#documentScope').onchange();
  assert.equal(node('#documentEngines').value, 'engine-a'); // Scope changes never erase conditions silently.
  assert.throws(() => run('readDocumentMetadata()'), /清空/);
  node('#documentScope').value = 'known';
  node('#operatorToken').value = 'test-token';
  node('#sourceRevision').value = 'R4';
  node('#knowledgeFile').files = [{name:'manual.md',size:20,text:async()=> 'synthetic manual'}];
  await run('uploadKnowledge()');
  const upload = requests.find(request => request.url === '/api/documents' && request.options.method === 'POST');
  assert.equal(upload.body.document_id, 'logical-doc');
  assert.equal(upload.body.metadata.source_revision, 'R4');
  assert.deepEqual(upload.body.metadata.applicability, scoped.applicability);
  assert.equal(upload.options.headers['X-AeroDiagnosis-Operator-Token'], 'test-token');
  assert.equal(upload.options.headers['Content-Type'], 'application/json');
  const count = requests.length;
  node('#documentScope').value = 'known';
  node('#documentEngines').value = '';
  node('#documentPhases').value = '';
  await run('uploadKnowledge()');
  assert.equal(requests.length, count); // Invalid metadata never reaches the server.
  assert.match(node('#uploadResult').textContent, /至少/);

  node('#catalogEngine').value = 'engine-a';
  node('#catalogPhase').value = 'cruise';
  node('#catalogStrict').checked = true;
  node('#catalogEngine').oninput();
  assert.match(node('#documentList').innerHTML, /logical-doc/);
  assert.doesNotMatch(node('#documentList').innerHTML, /unpublished/);
  node('#knowledgeSearchQuery').value = 'EGT rise';
  await run('searchKnowledgeEvidence({preventDefault(){}})');
  const search = requests.find(request => request.url === '/api/evidence/search');
  assert.deepEqual(search.body, {query:'EGT rise',top_k:5,knowledge_context:{engine_model:'engine-a',flight_phase:'cruise',require_known:true}});
  assert.match(node('#knowledgeSearchResults').innerHTML, /适用条件匹配/);
  assert.doesNotMatch(node('#knowledgeSearchResults').innerHTML, /<img/);

  run("state.providerMode='default';state.defaultProvider=true;state.sessionId='session';submitGroundedQA=async body=>{capturedDiagnosis=body}");
  node('#question').value = 'Explain EGT rise';
  node('#qaEngine').value = 'engine-a';
  node('#qaPhase').value = 'cruise';
  node('#qaStrict').checked = true;
  await run('diagnose({preventDefault(){}})');
  assert.deepEqual(JSON.parse(JSON.stringify(sandbox.capturedDiagnosis.knowledge_context)),
    {engine_model:'engine-a',flight_phase:'cruise',require_known:true});
  assert.equal(sandbox.capturedDiagnosis.session_id, 'session');
  run("renderAgentEvidence({evidence:[{source_kind:'document_chunk',score:1,evidence_id:'e',excerpt:'text',applicability_status:'unknown'}]})");
  assert.match(node('#agentEvidence').innerHTML, /适用性未知/);
  run('capturedDiagnosis=null');
  node('#question').value = 'second question';
  node('#qaEngine').value = '';
  node('#qaPhase').value = '';
  await run('diagnose({preventDefault(){}})');
  assert.equal(sandbox.capturedDiagnosis, null);
  assert.match(node('#composerHint').textContent, /至少/);
  console.log('Knowledge UI behavior and API payload checks passed');
}
main().catch(error => {console.error(error);process.exitCode=1});

