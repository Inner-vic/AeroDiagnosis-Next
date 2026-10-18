const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const staticRoot=path.resolve(process.argv[2]);
let output='';
const sandbox={esc:v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),
  addMessage:(role,html)=>{output=html},$:()=>({lastElementChild:{querySelectorAll:()=>[]}})};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(staticRoot,'grounded-qa.js'),'utf8'),sandbox);
const response={evidence:[{id:'c1',title:'<img src=x>',text:'引用💡片段。',version_id:'v1',text_kind:'canonical_source'}],
  claims:[{text:'<script>fake</script>',citations:[{chunk_id:'c1',start:0,end:5,text_sha256:'h'}]}],
  paths:[{entities:['上游实体','<img src=x>'],relations:['导致'],hops:2,source_chunks:['c1']}],
  retrieval:{graph_enabled:true},insufficient_evidence:false};
sandbox.response=response;
vm.runInContext('renderGroundedQA(response)',sandbox);
assert.match(output,/查看来源支持的图谱路径/);
assert.match(output,/data-qa-source="c1"/);
assert.match(output,/data-qa-start="0"/);
assert.match(output,/&lt;script&gt;fake&lt;\/script&gt;/);
assert.doesNotMatch(output,/<script>|<img/);
response.paths=[];response.insufficient_evidence=true;response.claims=[];
response.reason='缺少中间环节证据';response.answer='现有证据不足';response.retrieval={graph_enabled:false,missing_path_context:true};
vm.runInContext('renderGroundedQA(response)',sandbox);
assert.match(output,/当前证据不足/);assert.match(output,/缺少中间环节证据/);
assert.match(output,/未完整进入本次证据/);
assert.doesNotMatch(output,/查看来源支持的图谱路径/);
console.log('Grounded QA citation, path, insufficient-evidence and HTML escaping checks passed');
