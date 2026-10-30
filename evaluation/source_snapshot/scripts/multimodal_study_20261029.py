"""Patient-paired fixed-recipe fusion study; historical TEST never accessed."""
import sys,json,time,hashlib,argparse,importlib.util
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss,roc_curve,precision_recall_curve,confusion_matrix
R=Path(__file__).resolve().parents[1];B=R/'artifacts/compacta_cv15_20261028/prepared';O=R/'artifacts/multimodal_20261029';REPORT=R/'reports/multimodal_20261029'
sys.path.insert(0,str(B/'source_snapshot/src'))
from breastdcedl.university.config import PatientConfig
from breastdcedl.university.training import build_patient_model
from breastdcedl.university.data import standardize
from breastdcedl.data.validation import load_development
from breastdcedl.data.image_cache import ImageCache
spec=importlib.util.spec_from_file_location('fusion_impl',R/'src/breastdcedl/multimodal.py');mm=importlib.util.module_from_spec(spec);spec.loader.exec_module(mm)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def extract(cp_path,rows,cache,out):
 if out.exists():
  z=np.load(out);assert str(z['checkpoint_sha'])==sha(cp_path) and str(z['extractor_version'])=='loaded_state_v2';return z
 cp=torch.load(cp_path,map_location='cpu',weights_only=False);model=build_patient_model(PatientConfig(**cp['config'])).cuda().eval();assert model.n_parameters==24713
 model.load_state_dict(cp['model_state_dict'],strict=True)
 assert all(torch.equal(v.cpu(),cp['model_state_dict'][k].cpu()) for k,v in model.state_dict().items())
 ids=[];emb=[];scores=[];errors=[]
 with torch.inference_mode():
  for pid,g in rows.groupby('patient_id',sort=True):
   x=standardize(torch.stack([cache.get(s) for s in g.sample_id]).cuda(),cp['normalization']);h=torch.cat([model._encode(ch) for ch in x.split(16)]).float().mean(0,keepdim=True);a=model._classify(h).sigmoid().item();b=model.forward_patient(x).sigmoid().item();errors.append(abs(a-b));ids.append(pid);emb.append(h.cpu().numpy()[0]);scores.append(a)
 assert max(errors)<1e-7
 np.savez(out,ids=np.array(ids),embedding=np.array(emb),score=np.array(scores),checkpoint_sha=sha(cp_path),extractor_version='loaded_state_v2',parity_error=max(errors));del model;torch.cuda.empty_cache();return np.load(out)
def train_head(e,c,y,seed=42,epochs=80,evaluation=None):
 torch.manual_seed(seed);net=mm.FusionNet(c.shape[1]);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);x=torch.tensor(e,dtype=torch.float32);t=torch.tensor(c);target=torch.tensor(y,dtype=torch.float32);history=[]
 for epoch in range(epochs):
  net.train();order=torch.randperm(len(y))
  for ix in order.split(64):
   opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(net(x[ix],t[ix]),target[ix]);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
  net.eval()
  with torch.inference_mode():history.append({'epoch':epoch+1,'train_bce':float(torch.nn.functional.binary_cross_entropy_with_logits(net(x,t),target)),'validation_bce':float(torch.nn.functional.binary_cross_entropy_with_logits(net(torch.tensor(evaluation[0],dtype=torch.float32),torch.tensor(evaluation[1])),torch.tensor(evaluation[2],dtype=torch.float32))) if evaluation is not None else None})
 return net,history
def metrics(y,p):return dict(n=len(y),auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),bce=float(log_loss(y,p,labels=[0,1])),brier=float(brier_score_loss(y,p)),threshold=.5,confusion_matrix=confusion_matrix(y,np.array(p)>=.5,labels=[0,1]).tolist())
def boot(y,a,b=None):
 rng=np.random.default_rng(42);ixs=[np.flatnonzero(y==k) for k in [0,1]];values=[]
 for _ in range(2000):
  ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in ixs]);v=roc_auc_score(y[ix],a[ix]);values.append(v if b is None else v-roc_auc_score(y[ix],b[ix]))
 return [float(v) for v in np.quantile(values,[.025,.975])]
def main():
 p=argparse.ArgumentParser();p.add_argument('--extract-only',action='store_true');args=p.parse_args();torch.set_num_threads(4);O.mkdir(exist_ok=True,parents=True);REPORT.mkdir(exist_ok=True,parents=True)
 protocol=json.loads((B/'protocol.json').read_text());ledger=json.loads((B/'ledger.json').read_text());assert ledger['status']=='completed';rows,_=load_development(B/'metadata');cache=ImageCache(Path(protocol['cache_dir']),rows)
 # Extract only development images. No TEST predictions or features are opened.
 for f in ledger['folds']:
  fold=f['outer_fold'];cp=Path(f['checkpoint']);assert sha(cp)==f['checkpoint_sha256'];norm=torch.load(cp,map_location='cpu',weights_only=False)['normalization'];part=pd.read_csv(B/f'fold_{fold}/partition.csv');fit=set(part.loc[part.role=='refit','patient_id']);outer=set(part.loc[part.role=='outer','patient_id']);assert fit.isdisjoint(outer) and set(norm['fit_patient_ids'])==fit
  extract(cp,rows,cache,O/f'embeddings_fold_{fold}.npz');print('Extracted fold',fold,flush=True)
 cp=R/'artifacts/curve_search_20261029/prepared/weight_ema08/epoch_015.pt';assert sha(cp)=='d9fcf647298a88b3106eb5861a866dbab937f22d191d5e5e3804e97fe23677d5';extract(cp,rows,cache,O/'embeddings_original.npz')
 if args.extract_only:return
 schema=json.loads((O/'schema.json').read_text());features=schema.get('primary_features',schema.get('features'));features=[x['name'] if isinstance(x,dict) else x for x in features];clinical=pd.read_csv(O/'clinical.csv');key='patient_id' if 'patient_id' in clinical else 'pid';clinical=clinical.rename(columns={key:'patient_id'});clinical=clinical[clinical.eligible_primary.astype(str).str.lower().isin(['true','1'])];people=rows[['patient_id','pCR','fold','cohort']].drop_duplicates().merge(clinical[['patient_id']+features],on='patient_id',validate='one_to_one');assert len(people)==1092
 outputs=[];losses={};coefs=[]
 for f in ledger['folds']:
  fold=f['outer_fold'];z=np.load(O/f'embeddings_fold_{fold}.npz');lookup={pid:i for i,pid in enumerate(z['ids'])};frame=people.sort_values('patient_id').copy();ee=np.array([z['embedding'][lookup[pid]] for pid in frame.patient_id]);train=frame.fold.to_numpy()!=fold;test=~train;state=mm.fit_clinical(frame[train],features);cc=mm.encode_clinical(frame,state);mu=ee[train].mean(0);sd=ee[train].std(0);sd[sd<1e-6]=1;ee=(ee-mu)/sd;y=frame.pCR.to_numpy();lr=LogisticRegression(C=1,solver='liblinear',max_iter=2000,random_state=42).fit(cc[train],y[train]);linear=LogisticRegression(C=1,solver='liblinear',max_iter=2000,random_state=42).fit(np.concatenate((ee[train],cc[train]),1),y[train]);net,hist=train_head(ee[train],cc[train],y[train],evaluation=(ee[test],cc[test],y[test]));losses[str(fold)]=hist
  with torch.inference_mode():fusion=net(torch.tensor(ee[test],dtype=torch.float32),torch.tensor(cc[test])).sigmoid().numpy()
  val=frame[test][['patient_id','pCR','fold','cohort']].copy();val['image']=[z['score'][lookup[pid]] for pid in val.patient_id];val['tabular']=lr.predict_proba(cc[test])[:,1];val['fusion']=fusion;val['linear_fusion']=linear.predict_proba(np.concatenate((ee[test],cc[test]),1))[:,1];outputs.append(val);coefs.append(lr.coef_[0].tolist());write(O/f'clinical_transform_fold_{fold}.json',state);torch.save({'head_state':net.state_dict(),'clinical_transform':state,'embedding_mean':mu.tolist(),'embedding_std':sd.tolist(),'features':features},O/f'head_fold_{fold}.pt');print('Trained fusion fold',fold,flush=True)
 result=pd.concat(outputs).sort_values('patient_id');result.to_csv(O/'oof_predictions.csv',index=False);y=result.pCR.to_numpy();summary={'protocol':'Five patient-disjoint outer folds; frozen compact CNN checkpoints selected by historical inner BCE. Ordinary CNN, not historical globally selected EMA. Fixed 80-epoch fusion recipe, no outer-based tuning. Historical TEST not accessed.','models':{},'features':features,'fine_tuning':{'status':'not_run','reason':'Conditional experiment: no independent inner fusion improvement gate was established; do not unlock using outer results.'},'losses':losses}
 for name in ['image','tabular','fusion','linear_fusion']:
  pp=result[name].to_numpy();m=metrics(y,pp);m['auc_ci95']=boot(y,pp);fpr,tpr,th=roc_curve(y,pp);pr,re,_=precision_recall_curve(y,pp);m['roc']=[dict(fpr=float(a),tpr=float(b)) for a,b in zip(fpr,tpr)];m['pr']=[dict(recall=float(a),precision=float(b)) for a,b in zip(re,pr)];summary['models'][name]=m
 summary['paired_auc_delta']={name:{'delta':summary['models']['fusion']['auc']-summary['models'][name]['auc'],'ci95':boot(y,result.fusion.to_numpy(),result[name].to_numpy())} for name in ['image','tabular']};summary['clinical_coefficients']={'features':features+[n+':missing' for n in features],'fold_coefficients':coefs,'meaning':'Predictive associations, not causal effects.'}
 # Final training on development only, original frozen EMA extractor retained.
 z=np.load(O/'embeddings_original.npz');keep=np.array([pid in set(people.patient_id) for pid in z['ids']]);frame=people.set_index('patient_id').loc[z['ids'][keep]].reset_index();state=mm.fit_clinical(frame,features);cc=mm.encode_clinical(frame,state);ee=z['embedding'][keep];mu=ee.mean(0);sd=ee.std(0);sd[sd<1e-6]=1;net,hist=train_head((ee-mu)/sd,cc,frame.pCR.to_numpy());bundle={'head_state':net.state_dict(),'clinical_transform':state,'embedding_mean':mu.tolist(),'embedding_std':sd.tolist(),'features':features,'cnn_sha256':str(z['checkpoint_sha']),'image_checkpoint_sha256':str(z['checkpoint_sha']),'config':{'epochs':80,'seed':42,'lr':.001,'weight_decay':.01},'evaluation':'Final head fitted1092eligible development patients. No heldout performance assigned; OOF uses different fold-local CNN weights.'};torch.save(bundle,O/'final.pt');check=torch.load(O/'final.pt',weights_only=False);p1=mm.predict_embedding(ee[0],frame.iloc[0].to_dict(),check)
 with torch.inference_mode():p2=float(net(torch.tensor(((ee[0]-mu)/sd)[None],dtype=torch.float32),torch.tensor(cc[:1])).sigmoid().item())
 assert abs(p1-p2)<1e-6;summary['final_reload_parity']=abs(p1-p2);summary['final_model_evaluation']=bundle['evaluation'];summary['conclusion']='No hay una ventaja demostrada de la fusión sobre las variables clínicas si el intervalo pareado incluye cero; consultar las diferencias estimadas.';summary['historical_label_limit']='Five unknown I-SPY1 outcomes excluded from new clinical/fusion fits and evaluation. Reused historical fold CNNs had included these noisy labels in fitting; residual label noise remains.';write(REPORT/'results.json',summary);write(O/'public_results.json',summary);print(json.dumps({k:v for k,v in summary.items() if k in ['paired_auc_delta','final_reload_parity']}),flush=True)
if __name__=='__main__':main()
