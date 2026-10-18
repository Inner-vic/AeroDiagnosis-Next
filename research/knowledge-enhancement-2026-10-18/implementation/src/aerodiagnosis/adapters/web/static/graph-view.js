(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.GraphView=api})(typeof globalThis!=='undefined'?globalThis:this,function(){
  function filter(data,hidden){const nodes=(data.nodes||[]).filter(n=>!hidden.has(String(n.kind).toLowerCase().replaceAll('_','-')));const ids=new Set(nodes.map(n=>n.node_id));return {nodes,edges:(data.edges||[]).filter(e=>ids.has(e.source_id)&&ids.has(e.target_id))}}
  function shortLabel(value,limit=12){const chars=Array.from(String(value||''));return chars.length>limit?chars.slice(0,limit).join('')+'…':chars.join('')}
  return {filter,shortLabel};
});
