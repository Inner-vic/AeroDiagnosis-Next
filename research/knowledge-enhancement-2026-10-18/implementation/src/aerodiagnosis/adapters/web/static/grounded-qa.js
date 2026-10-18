/* Span-bound knowledge answers. All source/model content is escaped or textContent. */
async function submitGroundedQA(body){
  appendAgentTrace('检索知识','正在检索文档与来源约束图谱','active');
  const result=await api('/qa/answer',{method:'POST',body:JSON.stringify(body)});
  renderGroundedQA(result);
  const t=result.retrieval;
  appendAgentTrace('检索完成',t.graph_enabled?'有可靠因果路径，已启用三路融合':({noncausal_or_boundary:'当前问题不适合因果路径，已使用文本检索',graph_unavailable:'图谱暂不可用，已回退文本检索',no_active_sources:'未找到符合筛选条件的资料',no_verified_path:'未找到通过来源校验的完整路径，已回退文本检索'}[t.reason]||'已回退文本检索'),'ok',result.run_id);
  $('#generationQuality').textContent=result.generation.verification?'已检查引用片段、不确定性和模型支持性；这不等于专家验证。':result.reason;
  $('#agentEvidence').innerHTML=result.evidence.map((x,i)=>`<article class="evidence-card"><b>证据 ${i+1} · ${esc(x.title)}</b><p>${esc(x.text)}</p><small>${esc(x.version_id)} · ${esc(x.applicability_status)}</small></article>`).join('')||'<p>暂无证据。</p>';
}
function renderGroundedQA(r){
  const numbers=new Map(r.evidence.map((e,i)=>[e.id,i+1]));
  const cite=(id,quote=null)=>`<button type="button" class="qa-citation" data-qa-source="${esc(id)}" ${quote?`data-qa-start="${quote.start}" data-qa-end="${quote.end}" data-qa-sha="${esc(quote.text_sha256)}"`:''}>[${numbers.get(id)||'来源'}] 查看片段</button>`;
  const claims=r.claims.map(c=>`<div class="claim"><p>${esc(c.text)}</p>${c.citations.map(q=>cite(q.chunk_id,q)).join(' ')}</div>`).join('');
  const paths=r.paths.map(p=>`<article class="qa-path"><p>${p.entities.map((n,i)=>`${i?`<span class="qa-path-arrow"> → ${esc(p.relations[i-1])} → </span>`:''}<b>${esc(n)}</b>`).join('')}</p><small>来源已核查 · ${p.hops} 条关系 · 本次证据包含完整路径</small><p>${p.source_chunks.map(id=>cite(id)).join(' ')}</p></article>`).join('');
  const evidence=r.evidence.map((e,i)=>`<article class="evidence-card"><b>[${i+1}] ${esc(e.title)}</b><p>${esc(e.text)}</p><small>版本 ${esc(e.version_id)} · ${esc(e.text_kind==='chinese_translation'?'原资料中文译文':'资料原文片段')}</small><p>${cite(e.id)}</p></article>`).join('');
  const html=`<div class="diagnosis-summary ${r.insufficient_evidence?'refusal':''}"><b>${r.insufficient_evidence?'当前证据不足':'知识回答'}</b>${claims||`<p>${esc(r.answer)}</p>`}${r.insufficient_evidence?`<p>${esc(r.reason||'现有证据不足以完整回答该问题。')}</p>`:''}<small>${r.retrieval.graph_enabled?'本次图谱已参与检索':'本次使用文本检索'}</small>${paths?`<details class="answer-evidence" open><summary>查看来源支持的图谱路径</summary>${paths}</details>`:r.retrieval.missing_path_context?'<p>图谱候选路径未完整进入本次证据，不展示为完整推理依据。</p>':''}<details class="answer-evidence"><summary>查看全部 ${r.evidence.length} 条检索证据</summary>${evidence||'<p>暂无符合条件的证据。</p>'}</details></div>`;
  addMessage('assistant',html,true);
  const message=$('#messages').lastElementChild;
  message.querySelectorAll('[data-qa-source]').forEach(button=>button.onclick=()=>openQASource(button.dataset));
}
async function openQASource(data){
  let dialog=document.querySelector('#qaSourceDialog');
  if(!dialog){dialog=document.createElement('dialog');dialog.id='qaSourceDialog';dialog.className='qa-source-dialog';dialog.innerHTML='<button type="button" class="ghost-button qa-source-close">关闭</button><h2></h2><p class="qa-source-location"></p><pre class="qa-source-text"></pre><p class="qa-source-notice"></p>';document.body.append(dialog);dialog.querySelector('button').onclick=()=>dialog.close()}
  dialog.querySelector('h2').textContent='正在读取来源片段…';dialog.querySelector('pre').textContent='';dialog.querySelector('.qa-source-notice').textContent='';if(!dialog.open)dialog.showModal();
  const requestId=Symbol();dialog.qaRequest=requestId;
  try{
    const source=await api('/qa/evidence/'+encodeURIComponent(data.qaSource));if(dialog.qaRequest!==requestId)return;
    dialog.querySelector('h2').textContent=source.title||'来源资料';
    const pages=source.locator?.coordinates?.pages||[];
    dialog.querySelector('.qa-source-location').textContent=`${source.text_kind==='chinese_translation'?'原资料中文译文':'原文片段'}${pages.length?' · 第 '+pages.join('、')+' 页':''} · 版本 ${source.version_id.slice(0,12)}`;
    const pre=dialog.querySelector('pre'),chars=Array.from(source.text),start=Number(data.qaStart),end=Number(data.qaEnd);
    if(data.qaSha&&data.qaSha!==source.text_sha256){pre.textContent=source.text;dialog.querySelector('.qa-source-notice').textContent='片段内容与回答时不一致，未高亮旧引用。';return}
    if(data.qaStart!==undefined&&Number.isInteger(start)&&Number.isInteger(end)&&start>=0&&end>start&&end<=chars.length){pre.append(document.createTextNode(chars.slice(0,start).join('')));const mark=document.createElement('mark');mark.textContent=chars.slice(start,end).join('');pre.append(mark,document.createTextNode(chars.slice(end).join('')))}else pre.textContent=source.text;
    dialog.querySelector('.qa-source-notice').textContent='引用位置对应上方片段内的文字；中文译文位置不等于原文件页内位置。';
  }catch(err){if(dialog.qaRequest===requestId)dialog.querySelector('h2').textContent=err.message}
}
