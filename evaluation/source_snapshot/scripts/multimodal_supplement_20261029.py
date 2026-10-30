from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd,torch
from sklearn.linear_model import LogisticRegression
R=Path(__file__).resolve().parents[1];O=R/'artifacts/multimodal_20261029';Q=R/'reports/multimodal_20261029'
sys.path.insert(0,str(R/'src'))
from breastdcedl.multimodal import encode_clinical
from sklearn.metrics import roc_auc_score,log_loss

def main():
 results=json.loads((Q/'results.json').read_text());df=pd.read_csv(O/'oof_predictions.csv');control=np.full(len(df),np.nan)
 for fold in range(5):
  z=np.load(O/f'embeddings_fold_{fold}.npz');look={p:i for i,p in enumerate(z['ids'])};e=np.array([z['embedding'][look[p]] for p in df.patient_id]);train=df.fold.to_numpy()!=fold;test=~train;mu=e[train].mean(0);sd=e[train].std(0);sd[sd<1e-6]=1;e=(e-mu)/sd
  lr=LogisticRegression(C=1,solver='liblinear',max_iter=2000,random_state=42).fit(e[train],df.pCR.to_numpy()[train]);control[test]=lr.predict_proba(e[test])[:,1]
 df['image_refit_control']=control;df.to_csv(O/'oof_predictions_with_control.csv',index=False)
 from sklearn.calibration import calibration_curve
 for name,m in results['models'].items():
  observed,predicted=calibration_curve(df.pCR,df[name],n_bins=10,strategy='uniform');m['calibration']=[{'predicted':float(a),'observed':float(b)} for a,b in zip(predicted,observed)]
 def pair(a,b):
  y=df.pCR.to_numpy();rng=np.random.default_rng(42);groups=[np.flatnonzero(y==k) for k in (0,1)];auc=[];bce=[]
  for _ in range(2000):
   ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);auc.append(roc_auc_score(y[ix],a[ix])-roc_auc_score(y[ix],b[ix]));bce.append(log_loss(y[ix],a[ix])-log_loss(y[ix],b[ix]))
  return {'auc_delta':float(roc_auc_score(y,a)-roc_auc_score(y,b)),'auc_ci95':np.quantile(auc,[.025,.975]).tolist(),'bce_delta':float(log_loss(y,a)-log_loss(y,b)),'bce_ci95':np.quantile(bce,[.025,.975]).tolist()}
 results['image_refit_control']={'auc':float(roc_auc_score(df.pCR,control)),'bce':float(log_loss(df.pCR,control))};results['paired_full']={k:pair(df.fusion.to_numpy(),df[k].to_numpy()) for k in ['image','tabular','image_refit_control']};results['bootstrap_limit']='Patient-stratified bootstrap conditional on these trained folds; does not include model fitting or selection variability.'
 p=results['paired_full']['tabular'];results['conclusion']='La fusión aporta una mejora de AUC frente al modelo clínico con intervalo condicional positivo.' if p['auc_ci95'][0]>0 else 'No se demuestra una ventaja de la fusión sobre las variables clínicas: el intervalo pareado de AUC incluye cero o favorece al modelo clínico.'
 (Q/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8');(O/'public_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
 labels={'image':'CNN por fold (sin EMA global)','tabular':'Regresión logística clínica','fusion':'CNN congelada + rama clínica','linear_fusion':'Fusión lineal regularizada'}
 web={'models':[{'id':k,'label':labels[k],'protocol':'OOF 5 folds emparejados · 1092 pacientes elegibles','n_patients':m['n'],'metrics':{'auc':m['auc'],'ap':m['average_precision'],'bce':m['bce'],'brier':m['brier']},'roc':m['roc'],'pr':m['pr'],'calibration':m['calibration']} for k,m in results['models'].items()],'paired_summary':json.dumps(results['paired_full'],ensure_ascii=False),'conclusion':results['conclusion'],'study_status':'Estudio retrospectivo interno completado. TEST histórico no utilizado. Fine-tuning condicional no ejecutado.'}
 (Q/'web_results.json').write_text(json.dumps(web,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps({'metrics':{k:{n:m[n] for n in ['auc','bce']} for k,m in results['models'].items()},'paired':results['paired_full'],'conclusion':results['conclusion']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
