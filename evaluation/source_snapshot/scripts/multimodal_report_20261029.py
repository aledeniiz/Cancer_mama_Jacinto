"""Render the actual paired OOF study and retain its methodological limits."""
from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];Q=R/'reports/multimodal_20261029';O=R/'artifacts/multimodal_20261029'
def main():
 d=json.loads((Q/'results.json').read_text());p=pd.read_csv(O/'oof_predictions.csv');old=pd.read_csv(R/'artifacts/compacta_cv15_20261028/prepared/oof_predictions.csv');errors=[]
 for fold in range(5):
  z=np.load(O/f'embeddings_fold_{fold}.npz');assert str(z['extractor_version'])=='loaded_state_v2';assert z['embedding'].shape==(1097,64);assert len(set(z['ids']))==1097
  g=old[old.fold==fold];look=dict(zip(z['ids'],z['score']));errors.extend(abs(g.score.to_numpy()-g.patient_id.map(look).to_numpy()))
 assert max(errors)<1e-6
 names={'image':'CNN por fold','tabular':'Solo clínica','fusion':'Fusión neuronal','linear_fusion':'Fusión lineal'}
 for kind in ['roc','pr','calibration']:
  fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
  for key,m in d['models'].items():
   pts=m[kind];x,y=({'roc':('fpr','tpr'),'pr':('recall','precision'),'calibration':('predicted','observed')}[kind]);ax.plot([v[x] for v in pts],[v[y] for v in pts],label=names[key],lw=1.7)
  if kind!='pr':ax.plot([0,1],[0,1],'--',color='.65',lw=1)
  ax.set(xlim=(0,1),ylim=(0,1),xlabel={'roc':'1 − especificidad','pr':'Sensibilidad','calibration':'Puntuación media'}[kind],ylabel={'roc':'Sensibilidad','pr':'Precisión','calibration':'Fracción observada de pCR'}[kind]);ax.legend();ax.grid(alpha=.15);fig.savefig(Q/f'{kind}.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
 for (fold,history),ax in zip(d['losses'].items(),axes.flat):
  ax.plot([v['epoch'] for v in history],[v['train_bce'] for v in history],label='Entrenamiento');ax.plot([v['epoch'] for v in history],[v['validation_bce'] for v in history],label='Fold exterior · diagnóstico');ax.set(title=f'Fold {int(fold)+1}',xlabel='Época',ylabel='BCE');ax.legend(fontsize=8);ax.grid(alpha=.15)
 axes.flat[-1].axis('off');fig.savefig(Q/'losses.png',dpi=160);plt.close(fig)
 rows=['| Estrategia | AUC | AP | BCE | Brier |','|---|---:|---:|---:|---:|']
 for k,m in d['models'].items():rows.append(f"| {names[k]} | {m['auc']:.4f} | {m['average_precision']:.4f} | {m['bce']:.4f} | {m['brier']:.4f} |")
 text='# Estudio multimodal: resultados reales\n\n'+d['conclusion']+'\n\n'+'\n'.join(rows)+'\n\n## Comparaciones pareadas\n\n'
 for k,v in d['paired_full'].items():text+=f"Fusión frente a {k}: ΔAUC {v['auc_delta']:+.4f}, IC95% [{v['auc_ci95'][0]:+.4f}, {v['auc_ci95'][1]:+.4f}]. ΔBCE {v['bce_delta']:+.4f}, IC95% [{v['bce_ci95'][0]:+.4f}, {v['bce_ci95'][1]:+.4f}].\n\n"
 text+='## Protocolo y límites\n\n1.092 pacientes de desarrollo elegibles, cinco folds por paciente. Edad y HR basales, con tres y dos ausencias antes de excluir resultados desconocidos. Medianas, escalado e indicadores de ausencia ajustados solo en entrenamiento de cada fold. CNN compacta de 24.713 parámetros congelada, embedding de 64 componentes. Rama clínica de 8 unidades y cabeza de fusión de 16, 80 épocas fijas, semilla 42, AdamW. No selección mediante resultados exteriores. Las curvas exteriores son diagnósticas y no se usan para seleccionar época.\n\nLos extractores por fold proceden del entrenamiento anidado histórico desde cero. Son CNN ordinarias, no la EMA global seleccionada: la referencia histórica AUC TEST 0,5524 pertenece a otro protocolo. Las cinco etiquetas I-SPY1 ausentes se excluyen de nuevos ajustes y evaluación, pero pudieron estar presentes como ruido en el ajuste histórico de los extractores. Desarrollo ya había sido explorado: esta comparación es retrospectiva, no confirmatoria.\n\nEl bootstrap estratificado por paciente (2.000 repeticiones, semilla 42) es condicional a los modelos ya entrenados, sin capturar toda la incertidumbre del entrenamiento y selección. No se evaluó ni optimizó con TEST histórico. Fine-tuning no ejecutado: no se estableció un criterio interno independiente que justificase desbloquear la CNN. Se incluyó fusión lineal adicional y control lineal de embeddings.\n\nModelo final separado: CNN EMA original conservada y cabeza entrenada con 1.092 pacientes. No se atribuye a ese archivo el AUC OOF de otros extractores. Uso experimental, sin calibración ni utilidad clínica demostrada. HER2, menopausia y volumen quedan excluidos por las razones de la auditoría. Importancias y coeficientes describen asociaciones, no causalidad.\n\nLa carga de pesos omitida en la primera extracción se corrigió. Sus resultados y archivos se conservan en INVALID_random_init y no constituyen evidencia. Las nuevas puntuaciones de los cinco folds reproducen las históricas con error máximo '+str(max(errors))+'.\n\nVer audit.md, audit_source_columns.csv, schema.json, results.json (métricas, coeficientes por fold, matrices, curvas e incertidumbre), y los scripts multimodal_*_20261029.py.\n'
 co=d['clinical_coefficients'];a=np.asarray(co['fold_coefficients']);text+='\n## Aportación de variables\n\nCoeficientes de la regresión logística, sobre entradas estandarizadas dentro de cada fold (rango entre folds, no intervalo de confianza):\n\n'
 for j,name in enumerate(co['features']):text+=f"- {name}: media {a[:,j].mean():+.4f}, rango [{a[:,j].min():+.4f}, {a[:,j].max():+.4f}].\n"
 text+='\nHR presenta una asociación negativa más intensa y consistente que edad en este modelo. No representa un efecto causal ni independencia respecto a tratamiento, subtipo o cohorte. Los indicadores de ausencia se basan en muy pocos casos y no admiten una interpretación estable. La fusión no demuestra valor incremental de MRI sobre edad y HR.\n'
 (Q/'resultado.md').write_text(text,encoding='utf-8')
 import importlib.metadata
 (O/'environment.json').write_text(json.dumps({k:importlib.metadata.version(k) for k in ['torch','numpy','pandas','scikit-learn','Pillow','matplotlib']},indent=2),encoding='utf-8')
 files={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [O/'final.pt',O/'schema.json',O/'clinical.csv',Q/'results.json',R/'src/breastdcedl/multimodal.py',R/'scripts/multimodal_study_20261029.py']}
 (Q/'verification.json').write_text(json.dumps({'original_prediction_parity_max':max(errors),'n':len(p),'unique':bool(p.patient_id.is_unique),'files_sha256':files,'python':sys.version},indent=2),encoding='utf-8');print('Report and figures verified')
if __name__=='__main__':main()
