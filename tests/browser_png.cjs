// Pure preprocessing checks in Node; no browser automation, model fit or network.
const fs=require('node:fs'),vm=require('node:vm'),crypto=require('node:crypto');
global.window={};global.document={baseURI:'https://example.invalid/'};
vm.runInThisContext(fs.readFileSync(process.argv[2],'utf8'));
const api=window.DCEBrowserInference,config=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
(async()=>{
 const names=process.argv.slice(4),files=names.map(name=>new File([fs.readFileSync(name)],name,{type:'image/png'}));
 const pixels=await api.decodePNG(files[0]);
 const normalized=api.normalize(pixels,config.normalization.means[0],config.normalization.std[0]);
 const rejected=[];for(const file of files.slice(1)){try{await api.decodePNG(file);rejected.push(false)}catch{rejected.push(true)}}
 console.log(JSON.stringify({pixel_sha:hash(pixels),normalized_sha:hash(Buffer.from(normalized.buffer)),
  clinical:Array.from(api.encodeClinical({age:48,HR:1},config.clinical_transform)),
  missing:Array.from(api.encodeClinical({age:null,HR:null},config.clinical_transform)),rejected}));
})().catch(error=>{console.error(error);process.exitCode=1});
