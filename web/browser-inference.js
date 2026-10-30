/* Local-only PNG -> original EMA CNN / frozen-image clinical fusion. No uploads. */
(()=>{'use strict';
const PHASES=['PRE','EARLY','LATE'],SIZE=256,PIXELS=SIZE*SIZE,MAX_FILE=2*1024*1024,MAX_TOTAL=64*1024*1024;
let loading=null,resources=null,active=false;
const base=()=>new URL('inference-assets/',document.baseURI);
const abort=signal=>{if(signal?.aborted)throw new DOMException('Operación cancelada.','AbortError')};
const failure=message=>{throw Error(message)};
const f32=Math.fround;
const crcTable=Uint32Array.from({length:256},(_,n)=>{let c=n;for(let k=0;k<8;k++)c=(c&1)?0xedb88320^(c>>>1):c>>>1;return c>>>0});
function crc(bytes,start,end){let c=0xffffffff;for(let i=start;i<end;i++)c=crcTable[(c^bytes[i])&255]^(c>>>8);return(c^0xffffffff)>>>0}
async function decodePNG(file,signal){
 abort(signal);if(!file||!Number.isFinite(file.size)||file.size>MAX_FILE||file.size<57)failure('Cada archivo debe ser un PNG válido de hasta 2 MiB.');
 const bytes=new Uint8Array(await file.arrayBuffer());abort(signal);const view=new DataView(bytes.buffer);
 if(![137,80,78,71,13,10,26,10].every((v,i)=>bytes[i]===v))failure('Formato incorrecto: solo se admite PNG.');
 let position=8,header=false,ended=false,sawData=false,dataEnded=false,compressedLength=0;const chunks=[];
 while(position<bytes.length){
  if(position+12>bytes.length)failure('PNG truncado.');
  const length=view.getUint32(position),start=position+8,end=start+length;
  if(end+4>bytes.length)failure('PNG truncado o con longitud inválida.');
  const type=String.fromCharCode(...bytes.subarray(position+4,position+8));
  if(!/^[A-Za-z]{4}$/.test(type)||crc(bytes,position+4,end)!==view.getUint32(end))failure('PNG dañado: comprobación CRC incorrecta.');
  if(!header&&type!=='IHDR')failure('Cabecera PNG inválida.');
  if(type==='IHDR'){
   if(header||length!==13)failure('Cabecera PNG duplicada o inválida.');header=true;
   if(view.getUint32(start)!==SIZE||view.getUint32(start+4)!==SIZE||bytes[start+8]!==8||bytes[start+9]!==0||bytes[start+10]!==0||bytes[start+11]!==0||bytes[start+12]!==0)failure('Se requiere PNG gris de 8 bits, 256 × 256, sin entrelazado. No se redimensiona ni convierte la imagen.');
  }else if(['acTL','fcTL','fdAT','tRNS'].includes(type))failure('No se admiten PNG animados ni transparencias.');
  else if(type==='IDAT'){
   if(dataEnded)failure('Orden de bloques PNG incorrecto.');sawData=true;compressedLength+=length;chunks.push(bytes.subarray(start,end));
  }else if(type==='IEND'){
   if(length!==0||!sawData||end+4!==bytes.length)failure('Final PNG inválido.');ended=true;position=end+4;break;
  }else{
   if(sawData)dataEnded=true;
   if(type[0]===type[0].toUpperCase()&&type!=='PLTE')failure('Bloque PNG crítico no compatible.');
   if(type==='PLTE')failure('Se requiere PNG de escala de grises, sin paleta.');
  }
  position=end+4;
 }
 if(!header||!ended||!compressedLength)failure('PNG incompleto.');
 if(typeof DecompressionStream!=='function')failure('Este navegador no permite descomprimir PNG localmente. Usa una versión actual de Chrome, Edge o Firefox.');
 const compressed=new Uint8Array(compressedLength);let cursor=0;for(const chunk of chunks){compressed.set(chunk,cursor);cursor+=chunk.length}
 const reader=new Blob([compressed]).stream().pipeThrough(new DecompressionStream('deflate')).getReader();
 const expected=(SIZE+1)*SIZE,raw=new Uint8Array(expected);let length=0;
 try{for(;;){abort(signal);const item=await reader.read();if(item.done)break;if(length+item.value.length>expected){await reader.cancel();failure('PNG con tamaño descomprimido incorrecto.')}raw.set(item.value,length);length+=item.value.length}}catch(error){try{await reader.cancel()}catch{}if(error.name==='AbortError')throw error;failure('No se pudo descomprimir el PNG: el contenido está dañado o tiene un tamaño incorrecto.')}finally{reader.releaseLock()}
 if(length!==expected)failure('Datos PNG incompletos.');
 const output=new Uint8Array(PIXELS);
 for(let y=0;y<SIZE;y++){
  const filter=raw[y*(SIZE+1)];if(filter>4)failure('Filtro PNG no válido.');
  for(let x=0;x<SIZE;x++){
   const i=y*SIZE+x,left=x?output[i-1]:0,up=y?output[i-SIZE]:0,upperLeft=x&&y?output[i-SIZE-1]:0;
   let predictor=0;if(filter===1)predictor=left;else if(filter===2)predictor=up;else if(filter===3)predictor=Math.floor((left+up)/2);else if(filter===4){const p=left+up-upperLeft,a=Math.abs(p-left),b=Math.abs(p-up),c=Math.abs(p-upperLeft);predictor=a<=b&&a<=c?left:b<=c?up:upperLeft}
   output[i]=(raw[y*(SIZE+1)+1+x]+predictor)&255;
  }
 }
 abort(signal);return output;
}
function ordered(files){return Array.from(files||[]).sort((a,b)=>a.name.localeCompare(b.name,undefined,{numeric:true}))}
function validatePhases(phases){
 const files=Object.fromEntries(PHASES.map(p=>[p,ordered(phases?.[p])]));const count=files.PRE.length;
 if(count<1||count>64||!PHASES.every(p=>files[p].length===count))failure('Selecciona entre 1 y 64 cortes por fase, con el mismo número de PRE, EARLY y LATE.');
 let total=0;for(const phase of PHASES)for(const file of files[phase]){if(file.size>MAX_FILE||file.size<57)failure('Cada PNG debe ocupar como máximo 2 MiB.');total+=file.size}
 if(total>MAX_TOTAL)failure('La selección supera el límite total de 64 MiB.');return files;
}
function encodeClinical(record,state){
 if(!record||!Object.hasOwn(record,'age')||!Object.hasOwn(record,'HR'))failure('Indica edad y HR; puedes dejarlos sin dato para aplicar la imputación original.');
 const values=[record.age,record.HR],out=new Float32Array(4);
 values.forEach((value,index)=>{const missing=value===null;if(!missing&&(typeof value!=='number'||!Number.isFinite(value)))failure('Edad y HR deben ser valores numéricos válidos o ausentes.');if(!missing&&(index===0?(value<18||value>110):![0,1].includes(value)))failure(index===0?'La edad admitida es de 18 a 110 años.':'HR debe ser 0, 1 o no disponible.');out[index]=((missing?state.median[index]:value)-state.mean[index])/state.std[index];out[index+2]=missing?1:0});return out;
}
function normalize(pixels,mean,std){const result=new Float32Array(PIXELS);mean=f32(mean);std=f32(std);for(let i=0;i<PIXELS;i++)result[i]=f32(f32(f32(pixels[i]/255)-mean)/std);return result}
async function checkedFile(path,manifest){
 if(!/^[A-Za-z0-9_.-]+$/.test(path))failure('Ruta de recurso inválida.');const entry=manifest.files.find(v=>v.path===path);if(!entry||!Number.isFinite(entry.size)||!/^[a-f0-9]{64}$/.test(entry.sha256))failure('Recurso no incluido en el manifiesto de inferencia.');
 const response=await fetch(new URL(path,base()),{credentials:'omit'});if(!response.ok)failure('No se pudo descargar el recurso de inferencia. Comprueba la conexión y vuelve a intentarlo.');
 const data=await response.arrayBuffer();if(data.byteLength!==entry.size)failure('El tamaño del recurso no coincide con su manifiesto.');const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),n=>n.toString(16).padStart(2,'0')).join('');if(hash!==entry.sha256)failure('El recurso de inferencia no supera la comprobación SHA-256.');return data;
}
async function load(){
 if(resources)return resources;if(loading)return loading;
 loading=(async()=>{
  const response=await fetch(new URL('asset-manifest.json',base()),{credentials:'omit'});if(!response.ok)failure('Motor local no disponible: no se encontró el manifiesto de recursos.');const manifest=await response.json();if(!Array.isArray(manifest.files))failure('Manifiesto de inferencia inválido.');
  const config=JSON.parse(new TextDecoder().decode(await checkedFile('model-config.json',manifest)));
  // The only script loaded is a reviewed same-origin runtime. No data is sent.
  if(!window.ort){const scriptData=await checkedFile('ort.min.js',manifest);await new Promise((resolve,reject)=>{const script=document.createElement('script'),url=URL.createObjectURL(new Blob([scriptData],{type:'text/javascript'}));script.src=url;script.onload=()=>{URL.revokeObjectURL(url);resolve()};script.onerror=()=>{URL.revokeObjectURL(url);script.remove();reject(Error('No se pudo iniciar ONNX Runtime.'))};document.head.append(script)})}
  if(!window.ort)failure('ONNX Runtime no se ha iniciado.');ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths=base().href;
  const sessions={};for(const kind of ['encoder','classifier','fusion']){const info=config.models[kind];if(!info)failure('Configuración de modelo incompleta.');sessions[kind]=await ort.InferenceSession.create(await checkedFile(info.file,manifest),{executionProviders:['wasm'],graphOptimizationLevel:'all'})}
  resources={config,sessions};return resources;
 })().catch(error=>{loading=null;throw error});return loading;
}
async function predict({phases,mode='image',clinical=null,onProgress=()=>{},signal}={}){
 if(active)failure('Ya hay una inferencia en curso. Espera a que termine.');if(!['image','fusion'].includes(mode))failure('Modelo no válido.');
 const files=validatePhases(phases);abort(signal);active=true;const start=performance.now();
 try{
  onProgress('Descargando y comprobando los modelos locales…');const {config,sessions}=await load();abort(signal);const count=files.PRE.length,embeddings=[];
  const encoded=mode==='fusion'?encodeClinical(clinical,config.clinical_transform):null;
  for(let slice=0;slice<count;slice++){
   abort(signal);onProgress(`Validando y procesando corte ${slice+1} de ${count} en este navegador…`);const input=new Float32Array(3*PIXELS);
   for(let channel=0;channel<3;channel++){const pixels=await decodePNG(files[PHASES[channel]][slice],signal);input.set(normalize(pixels,config.normalization.means[channel],config.normalization.std[channel]),channel*PIXELS)}
   const info=config.models.encoder,result=await sessions.encoder.run({[info.input]:new ort.Tensor('float32',input,[1,3,SIZE,SIZE])});abort(signal);const embedding=result[info.output].data;if(embedding.length!==64||!embedding.every(Number.isFinite))failure('La CNN devolvió una representación inválida.');embeddings.push(new Float32Array(embedding));Object.values(result).forEach(t=>t.dispose?.());
  }
  onProgress('Agregando los cortes y calculando la puntuación…');const mean=new Float32Array(64);for(let component=0;component<64;component++){let sum=0;for(const embedding of embeddings)sum=f32(sum+embedding[component]);mean[component]=f32(sum/count)}
  let result,info;if(mode==='image'){info=config.models.classifier;result=await sessions.classifier.run({[info.input]:new ort.Tensor('float32',mean,[1,64])})}else{info=config.models.fusion;const normalized=Float32Array.from(mean,(value,i)=>(value-config.embedding_mean[i])/config.embedding_std[i]);result=await sessions.fusion.run({[info.inputs[0]]:new ort.Tensor('float32',normalized,[1,64]),[info.inputs[1]]:new ort.Tensor('float32',encoded,[1,4])})}
  abort(signal);const logit=Number(result[info.output].data[0]);Object.values(result).forEach(t=>t.dispose?.());if(!Number.isFinite(logit))failure('El modelo devolvió una puntuación no válida.');const probability=1/(1+Math.exp(-logit));return {probability,logit,n_slices:count,model:mode==='image'?'CNN EMA 0,8 · época 15':'CNN EMA + clínica · fusión final',mode,elapsed_ms:performance.now()-start};
 }finally{active=false}
}
function resultHTML(result,threshold){const number=value=>value.toLocaleString('es-ES',{maximumFractionDigits:4,minimumFractionDigits:4});return `<h3>${result.model}</h3><output class="live-score mm-score" data-probability="${result.probability}" title="Puntuación sin redondear: ${result.probability}">${number(result.probability)}</output><p>Puntuación de pCR · no es una probabilidad clínica calibrada.</p><div class="live-result-summary"><div><small>Cortes del paciente</small>${result.n_slices}</div><div><small>Tiempo total</small>${(result.elapsed_ms/1000).toLocaleString('es-ES',{maximumFractionDigits:2})} s</div></div><p>Umbral exploratorio: ${number(threshold)}. Clase resultante: ${result.probability>=threshold?'pCR':'no-pCR'}.</p><p class="legend">${result.mode==='fusion'?'La cabeza multimodal final no tiene evaluación independiente.':'La AUC histórica de 0,5524 corresponde a otro protocolo y no estima la certeza de este caso.'} Resultado de investigación, sin recomendación terapéutica. Procesado en este dispositivo.</p>`}
window.DCEBrowserInference=Object.freeze({predict,load,decodePNG,encodeClinical,normalize,validatePhases,resultHTML});
})();
