/* Original CNN laboratory, now powered by local ONNX inference. */
window.initLive=function(){
 const $=id=>document.getElementById(id),root=$('live');if(!root||root.dataset.ready)return;root.dataset.ready='1';
 const phases=['PRE','EARLY','LATE'];let generation=0,busy=false,controller=null;const previews=new Map();
 const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 $('live-inputs').innerHTML=phases.map(p=>`<div class="live-drop" data-phase="${p}"><h3>${p}<small>PNG · 256 × 256</small></h3><label>Seleccionar cortes de ${p}<input type="file" id="live-file-${p}" multiple accept="image/png,.png"></label><p class="live-names" id="live-names-${p}">Selecciona los cortes.</p><img id="live-preview-${p}" alt="Vista previa del primer corte ${p}" hidden><button id="live-remove-${p}" class="live-remove" type="button">Retirar fase</button></div>`).join('');
 const files=()=>Object.fromEntries(phases.map(p=>[p,Array.from($('live-file-'+p).files).sort((a,b)=>a.name.localeCompare(b.name,undefined,{numeric:true}))]));
 const matching=()=>{const f=files(),n=f.PRE.length;return n>0&&n<=64&&phases.every(p=>f[p].length===n)};
 function state(kind,message){root.dataset.state=kind;$('live-progress').textContent=message}
 function controls(){const valid=matching();$('live-confirm').disabled=!valid;$('live-run').disabled=busy||!valid||!$('live-confirm').checked;$('live-reconnect').disabled=busy;$('live-demo').hidden=true;$('live-demo').disabled=true}
 function invalidate(message='Entrada modificada. Comprueba la correspondencia y confirma antes de ejecutar.'){
  generation++;controller?.abort();$('live-confirm').checked=false;$('live-result').innerHTML='<h3>Resultado pendiente</h3><p>Sin inferencia para la selección actual.</p>';state('ready',message);controls();
 }
 function pairing(){
  const f=files(),count=Math.max(...phases.map(p=>f[p].length));$('live-input-count').textContent=matching()?`${f.PRE.length} cortes completos`:'Fases pendientes o no coincidentes';
  $('live-pair-summary').textContent=matching()?'Comprueba cada fila: mismo corte, paciente y examen.':'Selecciona el mismo número de cortes en las tres fases (1–64).';
  $('live-pairs').innerHTML=count?`<table><thead><tr><th>Corte</th>${phases.map(p=>`<th>${p}</th>`).join('')}</tr></thead><tbody>${Array.from({length:Math.min(count,64)},(_,i)=>`<tr><td>${i+1}</td>${phases.map(p=>`<td>${escape(f[p][i]?.name||'Falta archivo')}</td>`).join('')}</tr>`).join('')}</tbody></table>`:'Sin imágenes seleccionadas.';
  for(const p of phases){const list=f[p];$('live-names-'+p).textContent=list.length?`${list.length} archivos: ${list.map(x=>x.name).join(', ')}`:'Selecciona los cortes.';const img=$('live-preview-'+p);if(previews.has(p)){URL.revokeObjectURL(previews.get(p));previews.delete(p)}img.removeAttribute('src');img.hidden=true;const current=generation;if(list[0])window.DCEBrowserInference.decodePNG(list[0]).then(()=>{if(current!==generation)return;const url=URL.createObjectURL(list[0]);previews.set(p,url);img.src=url;img.hidden=false}).catch(()=>{if(current===generation){img.removeAttribute('src');img.hidden=true}})}
  controls();
 }
 for(const p of phases){$('live-file-'+p).addEventListener('change',()=>{invalidate();pairing()});$('live-remove-'+p).onclick=()=>{$('live-file-'+p).value='';invalidate();pairing()}}
 $('live-confirm').onchange=controls;$('live-threshold').oninput=()=>invalidate('Umbral modificado. El resultado anterior queda invalidado; confirma la entrada de nuevo.');
 $('live-clear').onclick=()=>{for(const p of phases)$('live-file-'+p).value='';invalidate('Archivos y resultados retirados de esta interfaz.');pairing()};
 $('live-reconnect').textContent='Preparar motor local';$('live-reconnect').onclick=async()=>{if(busy)return;busy=true;controls();state('loading','Descargando y comprobando el motor local…');try{await window.DCEBrowserInference.load();state('ready','Motor preparado. Selecciona las tres fases y confirma sus cortes.')}catch(error){state('error','No se pudo preparar el motor: '+error.message)}finally{busy=false;controls()}};
 $('live-run').onclick=async()=>{
  if(busy||!matching()||!$('live-confirm').checked)return;const threshold=Number($('live-threshold').value);if(!$('live-threshold').value.trim()||!Number.isFinite(threshold)||threshold<0||threshold>1){state('error','El umbral debe ser un número entre 0 y 1.');return}
  const current=++generation;controller=new AbortController();busy=true;controls();state('working','Preparando inferencia local…');
  try{const result=await window.DCEBrowserInference.predict({phases:files(),mode:'image',signal:controller.signal,onProgress:message=>{if(current===generation)state('working',message)}});if(current!==generation)return;$('live-result').innerHTML=window.DCEBrowserInference.resultHTML(result,threshold);state('complete','Inferencia real completada en este navegador. No se han enviado las imágenes.')}catch(error){if(current===generation&&error.name!=='AbortError')state('error','No se pudo ejecutar la inferencia: '+error.message)}finally{busy=false;controls()}
 };
 $('live-connection').textContent='CNN EMA original ejecutable en este navegador. El motor se descarga una vez; las imágenes se procesan localmente y no se envían a un servidor.';
 state('ready','Selecciona PRE, EARLY y LATE del mismo paciente. El motor se cargará al ejecutar.');pairing();
};document.addEventListener('DOMContentLoaded',()=>window.initLive());
