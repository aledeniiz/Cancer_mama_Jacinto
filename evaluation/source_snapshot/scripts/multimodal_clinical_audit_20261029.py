"""Clinical inventory and exact linkage audit. No fitted transforms or models."""
from pathlib import Path
import json,hashlib,io,urllib.request,re
import pandas as pd
from openpyxl import load_workbook
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/multimodal_20261029';ART=ROOT/'artifacts/multimodal_20261029'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summary(s):
 s=s.replace([-1,'-1'],float('nan'));v=pd.to_numeric(s,errors='coerce')
 return {'n':len(s),'missing':int(s.isna().sum()),'missing_pct':float(s.isna().mean()*100),'dtype':str(s.dtype),'unique':int(s.nunique()),'distribution':{str(k):int(n) for k,n in s.value_counts().items()} if s.nunique()<=25 else {'min':float(v.min()),'q25':float(v.quantile(.25)),'median':float(v.median()),'q75':float(v.quantile(.75)),'max':float(v.max())}}
def decision(name):
 if name in ['age','HR']:return ('include_primary','Baseline after diagnostic biopsy; prior exact source mapping verified','baseline_by_dictionary_and_design')
 if name=='HER2':return ('exclude_primary','ISPY1 central retrospective supplementation creates availability uncertainty','baseline_context_individual_availability_uncertain')
 if name=='menopause':return ('exclude','Harmonization mismatch, cohort-identifying absence and one unresolved availability disagreement','baseline_context_uncertain_harmonization')
 if name in ['pCR','test','split','dataset','pid','sample_id','patient_id','fold']:return ('exclude_predictor','Outcome or identifier/partition/cohort shortcut','not_predictor')
 return ('exclude','No approved variable-level baseline provenance; acquisition/ROI/subtype redundancy or uncertain timing','unverified')
def main():
 OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
 people=pd.read_csv(ROOT/'metadata/patients.csv');samples=pd.read_csv(ROOT/'metadata/samples.csv')
 assert people.pid.notna().all() and not people.pid.duplicated().any()
 linked=samples.merge(people[['pid','split','pCR']],left_on='patient_id',right_on='pid',validate='many_to_one',suffixes=('_image','_patient'),indicator=True)
 assert linked['_merge'].eq('both').all() and linked.split_image.eq(linked.split_patient).all() and linked.pCR_image.eq(linked.pCR_patient).all()
 labels=pd.read_csv(ROOT/'artifacts/duke_source_recovery_20261018/ispy_dev_source_labels.csv')
 missing=labels.loc[~labels.source_label_available.astype(str).str.lower().isin(['true','1']),'patient_id'].astype(str)
 frame=people.rename(columns={'pid':'patient_id','pCR':'label','dataset':'cohort'})[['patient_id','label','cohort','split','age','HR']].copy()
 frame[['age','HR']]=frame[['age','HR']].replace(-1,float('nan'))
 frame['eligible_primary']=~frame.patient_id.isin(missing)
 frame['exclusion_reason']=frame.patient_id.isin(missing).map({True:'source_pCR_missing_do_not_treat_as_negative',False:''})
 assert set(frame.HR.dropna().unique())<=set([0,1]) and frame.age.dropna().between(18,110).all()
 frame.to_csv(ART/'clinical.csv',index=False)
 inventory=[]
 for name in people:
  dec,why,time=decision(name)
  for cohort,g in people.groupby('dataset'):
   inventory.append({'table':'metadata/patients.csv','column':name,'cohort':cohort,'decision':dec,'reason':why,'time':time,**summary(g[name])})
 manifest=json.loads((ROOT/'artifacts/duke_source_recovery_20261018/recovery_manifest.json').read_text())
 raw_inventory=[];sources={}
 for cohort,spec in manifest['sources'].items():
  raw=urllib.request.urlopen(spec['url'],timeout=60).read();assert hashlib.sha256(raw).hexdigest()==spec['sha256']
  book=load_workbook(io.BytesIO(raw),read_only=True,data_only=True)
  sources[cohort]={'url':spec['url'],'sha256':spec['sha256'],'sheets':book.sheetnames}
  for sheet in book:
   rows=list(sheet.iter_rows(values_only=True));h=1 if cohort=='duke' and sheet.title=='Data' else 0
   if not rows:continue
   headers=rows[h];body=rows[h+2:] if cohort=='duke' and sheet.title=='Data' else rows[h+1:]
   dictionary_sheet=any(x in sheet.title.lower() for x in ['dictionary','legend','explanation'])
   for i,name in enumerate(headers):
    if name is None:continue
    vals=pd.Series([r[i] if i<len(r) else None for r in body],dtype=object)
    risk='post_outcome_or_response' if re.search(r'response|pcr|surviv|recurr|residual|patholog|yp[tn]|death|rfs|dfs|os\b',str(name),re.I) else 'unverified_timing_or_cohort_specific'
    raw_inventory.append({'cohort':cohort,'sheet':sheet.title,'original_name':str(name),'role':'dictionary' if dictionary_sheet else 'source_data','decision':'exclude_unless_explicitly_mapped_to_age_or_HR','leakage_risk':risk,**summary(vals)})
  book.close()
 schema={'version':1,'primary_features':['age','HR'],'features':[{'name':'age','label':'Edad basal','type':'number','min':18,'max':110,'unit':'años','missing_allowed':True,'description':'Edad al diagnóstico/screening, según cohorte.'},{'name':'HR','label':'Receptores hormonales basales','type':'category','allowed':[0,1],'categories':[{'value':0,'label':'Negativos'},{'value':1,'label':'Positivos'}],'missing_allowed':True,'description':'ER o PR positivo en biopsia basal.'}],'supported_cohorts':['duke','spy1','spy2'],'timing_caveat':'Predicción posterior a biopsia basal y disponibilidad de receptores, antes de terapia neoadyuvante. No se recuperaron timestamps individuales.','missing_rule':'NA o -1 se convierte en ausencia; imputación y escalado se ajustan exclusivamente en cada fold de entrenamiento.','id_column':'patient_id','label_column':'label','cohort_column':'cohort','split_column':'split','eligibility_column':'eligible_primary','historical_test':'split=test; no selección con test histórico','source_references':['reports/pcr_audit_20261028/clinical_lineage/age_hr_gate.json','reports/pcr_audit_20261028/clinical_lineage/verified_mapping/audit.json'],'excluded_primary':{'HER2':'Disponibilidad individual de suplemento central retrospectivo incierta','menopause':'Harmonización y disponibilidad discordantes','tum_vol':'Proveniencia temporal/ROI no cerrada','race':'No necesaria para hipótesis primaria','subtypes':'Redundantes con receptores y no autorizados en esquema principal'}}
 (ART/'schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding='utf-8')
 report={'prediction_unit':'patient','contract':schema['timing_caveat'],'patients':len(people),'cuts':len(samples),'duplicate_patient_ids':0,'unmatched_image_rows':0,'label_split_disagreements':0,'clinical_without_images':int((~people.pid.isin(samples.patient_id)).sum()),'cohort_counts':people.dataset.value_counts().to_dict(),'official_split_counts':people.split.value_counts().to_dict(),'source_missing_outcomes_excluded':len(missing),'primary_eligible_dev':int((frame.eligible_primary & frame.split.eq('train')).sum()),'primary_eligible_total':int(frame.eligible_primary.sum()),'features':schema['primary_features'],'sources':sources,'metadata_sha256':sha(ROOT/'metadata/patients.csv'),'limitations':['Original Duke labels have known definition sensitivity; no relabeling is performed.','Five I-SPY1 DEV missing source outcomes excluded explicitly, labels retained in file but ineligible.','Raw source distributions are audit-only, never used to select predictors or fit transforms.','Exact original IDs for image joins; prior public source mappings use documented notebook numeric suffix correspondence, never row order.','One study context per patient in supplied PNG export; raw longitudinal visits are not available.']}
 (OUT/'audit_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'audit_columns.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'audit_source_columns.json').write_text(json.dumps(raw_inventory,ensure_ascii=False,indent=2),encoding='utf-8')
 pd.DataFrame([{k:v for k,v in r.items() if k!='distribution'} for r in raw_inventory]).to_csv(OUT/'audit_source_columns.csv',index=False)
 (OUT/'audit.md').write_text('# Auditoría clínica multimodal\n\nLa comparación principal usa edad + HR bajo contrato basal posterior a biopsia. Se excluyen cinco etiquetas I-SPY1 sin respuesta fuente; no se corrigen ni sobrescriben etiquetas originales. HER2, menopausia, volumen, raza, subtipos derivados y todas las variables no verificadas quedan fuera.\n\nInventario completo en audit_source_columns.json y audit_columns.json; integridad en audit_summary.json. Las fuentes TCIA se descargan en memoria y se verifican por SHA256. Los inventarios no entrenan transformadores. El dataset clínico local contiene identificadores solo para vinculación exacta con imágenes.\n\nLa ausencia de timestamps individuales y la heterogeneidad de definiciones pCR de Duke limitan la interpretación; la ausencia de solapamiento por ID no demuestra ausencia de duplicados semánticos.\n',encoding='utf-8')
 local_files=[]
 for path in ROOT.rglob('*'):
  if not path.is_file() or path.suffix.lower() not in {'.csv','.xlsx','.xls','.parquet','.tsv'}:continue
  relative=path.relative_to(ROOT).as_posix()
  if any(x in relative for x in ['.venv/','render-runtime/']):continue
  role='derived_predictions_or_experiment' if any(x in relative for x in ['epoch_','predictions','experiments/','training/runs/']) else 'metadata_or_audit_table'
  local_files.append({'path':relative,'bytes':path.stat().st_size,'role':role})
 (OUT/'audit_file_inventory.json').write_text(json.dumps({'files':local_files,'note':'Inventory records all table paths; experiment predictions are not clinical source predictors.'},ensure_ascii=False,indent=2),encoding='utf-8')
 prior=json.loads((ROOT/'reports/pcr_audit_20261028/clinical_lineage/verified_mapping/audit.json').read_text('utf-8'))
 (OUT/'audit_variable_evidence.json').write_text(json.dumps({'prior_mapping_sha256':sha(ROOT/'reports/pcr_audit_20261028/clinical_lineage/verified_mapping/audit.json'),'cohorts':prior['cohorts'],'prediction_gate':json.loads((ROOT/'reports/pcr_audit_20261028/clinical_lineage/age_hr_gate.json').read_text('utf-8'))},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'patients':len(frame),'eligible_dev':report['primary_eligible_dev'],'source_missing_outcomes':len(missing),'source_column_records':len(raw_inventory),'features':schema['primary_features']}))
if __name__=='__main__':main()
