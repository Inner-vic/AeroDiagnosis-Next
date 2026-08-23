/* ============================================================
   AeroDiagnosis v4 — App Logic (Light Theme)
   ============================================================ */
const API='/api',$=s=>document.querySelector(s),$$=s=>document.querySelectorAll(s);
let token=getCookie('aero_token'),username='';
function getCookie(n){const m=document.cookie.match('(^|;)\\s*'+n+'\\s*=\\s*([^;]+)');return m?m[2]:''}
function setCookie(n,v,d){const e=new Date();e.setTime(e.getTime()+(d||7)*864e5);document.cookie=n+'='+v+';expires='+e.toUTCString()+';path=/;SameSite=Lax'}
function delCookie(n){document.cookie=n+'=;expires=Thu,01 Jan 1970 00:00:00 UTC;path=/;'}

/* Boot */
(async function(){
  if(!token){showLogin();return}
  const ok=await checkAuth();ok?showApp():showLogin()
})();
async function checkAuth(){
  try{const r=await fetch(API+'/auth/check?token='+encodeURIComponent(token));const d=await r.json();if(d.valid){username=d.username;return true}}catch(e){}
  return false
}
function showLogin(){$('#app').classList.add('hidden');$('#loginScreen').style.display=''}
function showApp(){
  $('#loginScreen').style.display='none';$('#app').classList.remove('hidden');
  $('#sidebarUser').textContent=username;$('#sidebarAvatar').textContent=username.charAt(0).toUpperCase();
  initNav();initChat();initGraph();initDocs();initCases();initDash()
}

/* Login */
$('#loginForm').addEventListener('submit',async e=>{
  e.preventDefault();
  const u=$('#loginUser').value.trim(),p=$('#loginPass').value.trim();
  if(!u||!p)return;$('#loginError').style.display='none';
  try{
    const r=await fetch(API+'/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:u,password:p})});
    const d=await r.json();
    if(r.ok&&d.token){token=d.token;username=d.username;setCookie('aero_token',token,7);showApp()}
    else{$('#loginError').style.display='block';$('#loginError').textContent=d.detail||'Invalid credentials'}
  }catch(ex){$('#loginError').style.display='block';$('#loginError').textContent='Service unavailable'}
});
$('#logoutBtn').addEventListener('click',()=>{
  fetch(API+'/auth/logout',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})}).catch(()=>{});
  token='';delCookie('aero_token');window.location.reload()
});

/* Nav */
function initNav(){
  $$('.sb-item').forEach(el=>el.addEventListener('click',function(){
    $$('.sb-item').forEach(n=>n.classList.remove('active'));
    this.classList.add('active');
    $$('.page').forEach(p=>p.classList.remove('active'));
    const pg=$('#page-'+this.dataset.page);if(pg)pg.classList.add('active');
    if(this.dataset.page==='dashboard')loadDash();
    if(this.dataset.page==='graph')loadGraph();
    if(this.dataset.page==='docs')loadDocs();
    if(this.dataset.page==='cases')loadCases()
  }))
}

/* ============================================================
   CHAT
   ============================================================ */
function initChat(){
  const input=$('#chatInput'),btn=$('#chatSendBtn');let asking=false;
  btn.addEventListener('click',send);
  input.addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();send()}});
  input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,130)+'px'});
  $$('.hero-chip').forEach(c=>c.addEventListener('click',function(){input.value=this.dataset.q;send()}));
  async function send(){
    const q=input.value.trim();if(!q||asking)return;
    const hero=document.querySelector('.chat-hero');if(hero)hero.remove();
    asking=true;btn.disabled=true;addMsg('user',q);
    input.value='';input.style.height='auto';
    const typ=addTyping();
    try{
      const r=await fetch(API+'/qa/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
      typ.remove();
      if(!r.ok){const e=await r.json().catch(()=>({detail:'Server error'}));throw new Error(e.detail||'HTTP '+r.status)}
      const d=await r.json();addMsg('agent',d.answer,{intent:d.intent,sources:d.sources,hypotheses:d.hypotheses})
    }catch(err){typ.remove();addMsg('agent','**Error:** '+err.message)}
    finally{asking=false;btn.disabled=false}
  }
}
function addMsg(role,content,meta){
  const div=document.createElement('div');div.className='c-msg '+role;
  const av=document.createElement('div');av.className='cm-avatar';av.textContent=role==='user'?username.charAt(0).toUpperCase():'AI';
  const body=document.createElement('div');body.className='cm-body';
  const bubble=document.createElement('div');bubble.className='cm-bubble';
  if(role==='agent'){try{bubble.innerHTML=marked.parse(content||'');bubble.querySelectorAll('table').forEach(t=>{t.style.cssText='width:100%;border-collapse:collapse;margin:8px 0;font-size:13px;';t.querySelectorAll('td,th').forEach(c=>c.style.cssText='padding:6px 10px;border:1px solid #e2e6ed;')})}catch(e){bubble.textContent=content}}
  else bubble.textContent=content;
  body.appendChild(bubble);
  if(meta&&Object.values(meta).some(v=>Array.isArray(v)?v.length:v)){
    if(meta.intent){const b=document.createElement('span');b.className='cm-intent';b.textContent=meta.intent==='fault_diagnosis'?'故障诊断':meta.intent==='knowledge_qa'?'知识问答':meta.intent==='procedure'?'操作流程':meta.intent;body.appendChild(b)}
    if(meta.hypotheses&&meta.hypotheses.length){const h=document.createElement('div');h.className='cm-hyp';h.innerHTML='<strong>诊断假设:</strong><br>'+meta.hypotheses.map((h,i)=>`${i+1}. <strong>${h.cause||'?'}</strong> (${Math.round((h.confidence||0)*100)}%)<br><span style="opacity:.7;">${h.path||''}</span>`).join('<br>');body.appendChild(h)}
    if(meta.sources&&meta.sources.length){const s=document.createElement('div');s.className='cm-src';s.textContent='参考来源: '+meta.sources.map((s,i)=>`[${i+1}] ${s.source||s.type||'?'}`).join(' · ');body.appendChild(s)}
  }
  div.appendChild(av);div.appendChild(body);
  const c=$('#chatMessages');c.appendChild(div);c.scrollTop=c.scrollHeight
}
function addTyping(){
  const d=document.createElement('div');d.className='c-msg agent';d.id='_typing';
  d.innerHTML='<div class="cm-avatar">AI</div><div class="cm-body"><div class="cm-bubble cm-typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span> 推理中...</div></div>';
  $('#chatMessages').appendChild(d);$('#chatMessages').scrollTop=$('#chatMessages').scrollHeight;
  return{remove:()=>d.remove()}
}

/* ============================================================
   GRAPH
   ============================================================ */
let kgChart=null,gData={nodes:[],edges:[]},ctxNode=null;
const NC={EngineComponent:'#2563eb',EngineModel:'#2563eb',FaultMode:'#ef4444',Cause:'#ef4444',Symptom:'#ef4444',MaintenanceProcedure:'#16a34a',Algorithm:'#16a34a',Technology:'#16a34a',SensorParameter:'#9333ea',Parameter:'#9333ea'};
const EC={caused_by:'#ef4444',indicated_by:'#9333ea',repaired_by:'#16a34a',has_component:'#8a92a4',has_fault_mode:'#8a92a4',measured_by:'#9333ea',affects:'#ef4444'};
function initGraph(){
  $('#graphRefreshBtn').addEventListener('click',loadGraph);
  $('#graphFitBtn').addEventListener('click',()=>{if(kgChart)kgChart.dispatchAction({type:'restore'})});
  $('#graphSearch').addEventListener('input',debounce(function(){filterGraph(this.value)},300));
  document.addEventListener('click',()=>$('#contextMenu').classList.remove('open'));
  $$('.ctx-row').forEach(el=>el.addEventListener('click',function(){
    if(this.dataset.action==='detail')showNodeDetail(ctxNode);else if(this.dataset.action==='delete')delNode(ctxNode);
    $('#contextMenu').classList.remove('open')
  }));
  $('#drawerOverlay').addEventListener('click',closeDrawer);$('#drawerClose').addEventListener('click',closeDrawer);
  document.addEventListener('click',e=>{if(e.target.classList.contains('dr-tab'))renderDrawerTab(e.target.dataset.tab)})
}
async function loadGraph(){
  const l=$('#graphLoading'),e=$('#graphEmpty');l.style.display='';e.classList.add('hidden');
  try{const r=await fetch(API+'/admin/graph?limit=200');gData=await r.json();l.style.display='none';if(!gData.nodes.length){e.classList.remove('hidden');if(kgChart)kgChart.dispose();kgChart=null;return}e.classList.add('hidden');renderGraph(gData,'')}
  catch(x){l.style.display='none';e.classList.remove('hidden')}
}
function renderGraph(data,filter){
  if(!kgChart){kgChart=echarts.init($('#graphCanvas'));bindKgEvt()}kgChart.resize();
  let nodes=data.nodes;if(filter){const q=filter.toLowerCase();nodes=nodes.filter(n=>(n.name||'').toLowerCase().includes(q))}
  const nids=new Set(nodes.map(n=>n.name)),edges=data.edges.filter(e=>nids.has(e.source)&&nids.has(e.target));
  const cats=[...new Set(nodes.map(n=>n.type||'Unknown'))].map(t=>({name:t,itemStyle:{color:NC[t]||'#64748b'}}));

  // Merge duplicate node names: sum symbolSize for dupes
  const nodeMap={};nodes.forEach(n=>{
    const k=n.name;
    if(!nodeMap[k]){nodeMap[k]={...n,symbolSize:Math.min(60,14+(n.name||'').length*1.6),dupCount:1}}
    else{nodeMap[k].symbolSize=Math.min(70,nodeMap[k].symbolSize+4);nodeMap[k].dupCount++}
  });
  const mergedNodes=Object.values(nodeMap);

  kgChart.setOption({
    tooltip:{trigger:'item',backgroundColor:'#fff',borderColor:'#e2e6ed',textStyle:{color:'#10141e',fontSize:13},
      formatter:p=>p.dataType==='node'?`<div style="padding:4px"><b style="font-size:14px">${p.name}</b><br/><span style="color:#8a92a4">${p.data?.category||''}</span></div>`:`${p.data.source} <span style="color:#2563eb">→</span> ${p.data.target}<br/><span style="font-size:11px;color:#8a92a4">${p.data.relation||''}</span>`},
    legend:[{data:cats.map(c=>c.name),type:'scroll',orient:'vertical',right:10,top:10,textStyle:{fontSize:11,color:'#4a5368'}}],
    series:[{type:'graph',layout:'force',zoom:1.2,
      force:{repulsion:450,edgeLength:[120,300],gravity:.12,friction:.5},
      roam:true,draggable:true,
      data:mergedNodes.map(n=>({id:n.name,name:n.name,category:n.type||'Unknown',symbolSize:n.symbolSize,
        itemStyle:{color:NC[n.type]||'#64748b',shadowBlur:4,shadowColor:'rgba(0,0,0,.06)'}
      })),
      links:edges.map(e=>({source:e.source,target:e.target,
        label:{show:true,formatter:e.relation||'',fontSize:9,color:'#8a92a4',distance:8},
        lineStyle:{color:EC[e.relation]||'#d0d5de',width:EC[e.relation]?2.2:1.2,curveness:.25,opacity:.8},
      })),
      categories:cats,
      label:{show:true,fontSize:11,fontWeight:500,color:'#4a5368',
        formatter:p=>(p.name||'').length>14?p.name.slice(0,13)+'…':p.name,
        position:'right',distance:6
      },
      emphasis:{
        focus:'adjacency',
        label:{fontSize:14,fontWeight:700,color:'#10141e'},
        itemStyle:{shadowBlur:20,shadowColor:'rgba(37,99,235,.35)',borderWidth:2,borderColor:'#2563eb'},
        lineStyle:{width:3}
      },
      lineStyle:{color:'#d0d5de',curveness:.25,opacity:.7},
    }]
  },true)
}
function bindKgEvt(){
  kgChart.on('click',p=>{if(p.dataType==='node')showNodeDetail(p.name)});
  kgChart.on('contextmenu',p=>{p.event.event.preventDefault();if(p.dataType==='node'){ctxNode=p.name;const m=$('#contextMenu');m.style.left=p.event.event.clientX+'px';m.style.top=p.event.event.clientY+'px';m.classList.add('open')}})
}
function filterGraph(q){if(gData.nodes.length)renderGraph(gData,q)}

/* Drawer */
async function showNodeDetail(name){
  if(!name)return;
  try{const r=await fetch(API+'/admin/graph/node/'+encodeURIComponent(name));if(!r.ok)throw new Error('Not found');window._nd=await r.json();$('#drawerOverlay').classList.add('open');$('#drawer').classList.add('open');$$('.dr-tab').forEach(t=>t.classList.remove('active'));document.querySelector('.dr-tab[data-tab="props"]').classList.add('active');renderDrawerTab('props')}
  catch(e){alert('Failed to load entity: '+e.message)}
}
function closeDrawer(){$('#drawerOverlay').classList.remove('open');$('#drawer').classList.remove('open')}
function renderDrawerTab(tab){
  const d=window._nd;if(!d)return;const ct=$('#drawerContent');
  if(tab==='props'){const e=(d.entity?.e)||d.entity||{};ct.innerHTML=`<h3 style="margin-bottom:14px;font-size:16px;color:#10141e;">${e.name||'Unknown'}</h3><table><tr><td>类型</td><td>${e.type||'--'}</td></tr><tr><td>描述</td><td>${e.description||'--'}</td></tr></table>`}
  else if(tab==='meta'){const m=d.metadata||{};ct.innerHTML=`<h3 style="margin-bottom:14px;font-size:16px;color:#10141e;">元数据</h3><table><tr><td>版本</td><td>${m.version||'--'}</td></tr><tr><td>来源</td><td>${m.source||'--'}</td></tr><tr><td>创建</td><td>${m.created_at?new Date(m.created_at*1000).toLocaleString():'--'}</td></tr><tr><td>更新</td><td>${m.updated_at?new Date(m.updated_at*1000).toLocaleString():'--'}</td></tr></table>`}
  else if(tab==='neighbors'){const nb=d.neighbors||[];ct.innerHTML=`<h3 style="margin-bottom:14px;font-size:16px;color:#10141e;">关联节点 (${nb.length})</h3>`+(nb.length?nb.map(n=>`<div class="dr-neighbor"><div><span class="dr-nb-name">${n.name||'?'}</span><span class="dr-nb-type">${n.type||''}</span></div><span class="dr-nb-rel">${n.relation||''}</span></div>`).join(''):'<p style="color:#8a92a4;">暂无关联节点</p>')}
}
async function delNode(name){
  if(!confirm('确认软删除实体 "'+name+'"？'))return;
  try{const r=await fetch(API+'/admin/graph/node/'+encodeURIComponent(name),{method:'DELETE'});if(r.ok){closeDrawer();loadGraph()}else throw new Error((await r.json().catch(()=>({}))).detail||'Failed')}
  catch(e){alert('删除失败: '+e.message)}
}

/* ============================================================
   DOCS
   ============================================================ */
function initDocs(){
  const a=$('#uploadArea'),i=$('#fileInput');
  $('#uploadBtn').addEventListener('click',()=>i.click());
  a.addEventListener('click',e=>{if(e.target!==$('#uploadBtn')&&!e.target.closest('button'))i.click()});
  a.addEventListener('dragover',e=>{e.preventDefault();a.classList.add('drag-over')});
  a.addEventListener('dragleave',()=>a.classList.remove('drag-over'));
  a.addEventListener('drop',e=>{e.preventDefault();a.classList.remove('drag-over');if(e.dataTransfer.files.length)doUpload(e.dataTransfer.files)});
  i.addEventListener('change',()=>{if(i.files.length)doUpload(i.files)});
  $('#docsRefreshBtn').addEventListener('click',loadDocs)
}
async function doUpload(files){
  const list=Array.from(files);if(!list.length)return;
  const p=$('#uploadProgress'),bar=$('#progressFill'),txt=$('#progressText'),lbl=$('#progressLabel');
  p.classList.remove('hidden');p.style.background='';
  for(let i=0;i<list.length;i++){
    const f=list[i];lbl.textContent='处理: '+f.name;txt.textContent=Math.round((i/list.length)*100)+'%';bar.style.width=((i/list.length)*100)+'%';
    try{const fd=new FormData();fd.append('file',f);const r=await fetch(API+'/ingest/upload',{method:'POST',body:fd});if(!r.ok)throw new Error((await r.json().catch(()=>({detail:'Upload failed'}))).detail);const d=await r.json();lbl.textContent=f.name+' — '+d.chunks_count+' 片段, '+d.entities_count+' 实体';txt.textContent=Math.round(((i+1)/list.length)*100)+'%';bar.style.width=(((i+1)/list.length)*100)+'%'}
    catch(e){lbl.textContent=f.name+': '+e.message;p.style.background='#fef2f2'}
  }
  setTimeout(()=>{p.classList.add('hidden');p.style.background=''},8000);i.value='';loadDocs()
}
async function loadDocs(){
  const t=$('#docTableBody');
  try{const r=await fetch(API+'/documents');const d=await r.json();const docs=d.documents||[];
    if(!docs.length){t.innerHTML='<tr><td colspan="4" class="empty-cell"><div class="empty-msg"><svg width="36" height="36" opacity=".3"><use href="#i-file"/></svg><p>暂无文档</p><span>上传航空发动机知识文档以构建知识库</span></div></td></tr>';return}
    t.innerHTML=docs.map(doc=>`<tr><td style="font-weight:600">${esc(doc.name)}</td><td class="r">${doc.size_kb} KB</td><td class="r">${doc.uploaded_at||'--'}</td><td class="r"><span class="tag-ok">已入库</span></td></tr>`).join('')
  }catch(e){console.error('loadDocs failed:',e)}
}

/* ============================================================
   CASES
   ============================================================ */
let casePage=1,caseSearch='',caseTag='';
function initCases(){
  $('#caseRefreshBtn').addEventListener('click',()=>{casePage=1;loadCases()});
  $('#caseSearch').addEventListener('input',debounce(function(){
    caseSearch=this.value;casePage=1;loadCases()
  },300))
}
async function loadCases(){
  const g=$('#caseGrid'),t=$('#caseTags');
  try{
    const tr=await fetch(API+'/cases/tags');const td=await tr.json();
    t.innerHTML=(td.tags||[]).map(tag=>
      `<button class="case-tag${caseTag===tag?' active':''}" data-tag="${esc(tag)}">${esc(tag)}</button>`
    ).join('');
    t.querySelectorAll('.case-tag').forEach(btn=>btn.addEventListener('click',function(){
      caseTag=caseTag===this.dataset.tag?'':this.dataset.tag;casePage=1;loadCases()
    }));
    const params=new URLSearchParams({page:casePage,page_size:20,search:caseSearch,tag:caseTag});
    const r=await fetch(API+'/cases/list?'+params);const d=await r.json();
    if(!d.items||!d.items.length){
      g.innerHTML='<div style="grid-column:1/-1;text-align:center;padding:60px"><svg width="36" height="36" opacity=".3"><use href="#i-file"/></svg><p style="font-size:15px;font-weight:600;color:#4a5368;margin-top:10px">暂无案例</p><span style="font-size:13px;color:#8a92a4">使用诊断对话功能后, 案例将自动记录在此</span></div>';
      $('#casePagination').innerHTML='';return
    }
    g.innerHTML=d.items.map(c=>`<div class="case-card" onclick="showCaseDetail('${c.id}')">
      <div class="cc-head"><span class="cc-title">${esc(c.title)}</span><span class="cc-severity ${c.severity||'待评估'}">${c.severity||'待评估'}</span></div>
      ${c.engine_model?`<div class="cc-model">${esc(c.engine_model)}</div>`:''}
      <div class="cc-symptoms">${esc(c.symptoms||'')}</div>
      <div class="cc-footer">
        ${(c.tags||[]).map(t=>`<span class="cc-tag">${esc(t)}</span>`).join('')}
        ${c.outcome?`<span class="cc-outcome ${c.outcome}">${c.outcome==='confirmed'?'已确认':c.outcome==='partial'?'部分解决':'待评估'}</span>`:''}
      </div>
    </div>`).join('');
    const tp=Math.max(1,Math.ceil(d.total/d.page_size));
    let ph='';
    ph+=`<button ${casePage<=1?'disabled':''} onclick="casePage=1;loadCases()">首页</button>`;
    for(let p=Math.max(1,casePage-2);p<=Math.min(tp,casePage+2);p++)
      ph+=`<button class="${p===casePage?'active':''}" onclick="casePage=${p};loadCases()">${p}</button>`;
    ph+=`<button ${casePage>=tp?'disabled':''} onclick="casePage=${tp};loadCases()">末页</button>`;
    $('#casePagination').innerHTML=ph
  }catch(e){console.error(e)}
}
async function showCaseDetail(cid){
  try{
    const r=await fetch(API+'/cases/'+encodeURIComponent(cid));if(!r.ok)throw new Error('Not found');
    const c=await r.json();
    const overlay=document.createElement('div');
    overlay.className='case-modal-overlay open';
    overlay.innerHTML=`<div class="case-modal">
      <button class="cm-close" onclick="this.closest('.case-modal-overlay').remove()">✕</button>
      <h2>${esc(c.title)}</h2>
      <div class="cm-meta">
        <span class="cc-severity ${c.severity||'待评估'}">${c.severity||'待评估'}</span>
        ${c.outcome?`<span class="cc-outcome ${c.outcome}">${c.outcome==='confirmed'?'已确认':c.outcome==='partial'?'部分解决':'待评估'}</span>`:''}
        ${c.engine_model?`<span class="cc-tag">${esc(c.engine_model)}</span>`:''}
      </div>
      <div class="cm-section"><h4>故障现象</h4><p>${esc(c.symptoms||'--')}</p></div>
      <div class="cm-section"><h4>诊断结论</h4><p>${esc(c.diagnosis||'--')}</p></div>
      <div class="cm-section"><h4>故障类型</h4><p>${esc(c.fault_type||'--')}</p></div>
      <div class="cm-section"><h4>维修措施</h4><p>${esc(c.actions||'--')}</p></div>
      <div class="cm-section"><h4>处理结果</h4><p>${esc(c.result||'--')}</p></div>
      <div class="cm-section"><h4>标签</h4><div style="display:flex;gap:4px;flex-wrap:wrap">${(c.tags||[]).map(t=>`<span class="cc-tag">${esc(t)}</span>`).join('')}</div></div>
      <div style="font-size:11px;color:#8a92a4;margin-top:16px">案例ID: ${c.id} | 记录时间: ${c.created_at||'--'}</div>
    </div>`;
    overlay.addEventListener('click',e=>{if(e.target===overlay)overlay.remove()});
    document.body.appendChild(overlay)
  }catch(e){alert('Failed to load case: '+e.message)}
}

/* ============================================================
   DASHBOARD
   ============================================================ */
function initDash(){$('#dashRefreshBtn').addEventListener('click',loadDash)}
async function loadDash(){
  try{const[s,y]=await Promise.all([fetch(API+'/admin/stats').then(r=>r.json()),fetch(API+'/system').then(r=>r.json())]);$('#statChunks').textContent=(s.vector_store?.total_vectors||0).toLocaleString();$('#statEntities').textContent=(s.knowledge_graph?.total_entities||0).toLocaleString();$('#statRelations').textContent=(s.knowledge_graph?.total_relations||0).toLocaleString();$('#statUptime').textContent=y.uptime||'--';$('#sysBackend').textContent=y.backend||'--';$('#sysPython').textContent=y.python_version||'--'}catch(e){['statChunks','statEntities','statRelations','statUptime'].forEach(id=>$('#'+id).textContent='--')}
}

/* Utils */
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function debounce(fn,ms){let t;return function(...a){clearTimeout(t);t=setTimeout(()=>fn.apply(this,a),ms)}}
