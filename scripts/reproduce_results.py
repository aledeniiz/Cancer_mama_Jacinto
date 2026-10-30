"""Replay released scientific results without MRI images, fitting or training.

Default: verify scientific data hashes, patient coverage/order and public source
membership, recompute four OOF metrics + historical CNN TEST metrics, then
reload clinical/CNN classifier/fusion heads against original CNN embeddings.
This does not recompute MRI encoders or independently validate the final head.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,log_loss,average_precision_score,brier_score_loss
from breastdce_research.evaluation import binary_metrics
from breastdce_research.clinical import ClinicalFoldPredictor,encode_clinical
from breastdce_research.inference import load_checkpoint,load_cnn
from breastdce_research.multimodal import FusionNet

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'evaluation'
MODELS=ROOT/'models/trained'
SCORE_DTYPES={'image':'float64','tabular':'float64','fusion':'float32','linear_fusion':'float64','historical_test':'float64'}


def require(condition,message):
    if not condition:raise ValueError(message)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def original_metrics(labels,probabilities,model):
    """Restore original score dtype before reproducing historical log_loss.

    Fusion originally used a float32 numpy sigmoid column; its shortest decimal
    CSV serialization must be parsed back to float32. Other score columns were
    float64. Computing float64 log_loss on fusion CSV decimals changes BCE by
    ~1e-9 even though labels, ranking and trained predictions are unchanged.
    """
    probability=np.asarray(probabilities,dtype=SCORE_DTYPES[model])
    binary_metrics(labels,probability)  # Validate binary targets/probability domain.
    return {'n':len(probability),'auc':float(roc_auc_score(labels,probability)),
        'bce':float(log_loss(labels,probability,labels=[0,1])),
        'average_precision':float(average_precision_score(labels,probability)),
        'brier':float(brier_score_loss(labels,probability))}


def verify_manifest(data=DATA):
    manifest_path=data/'MANIFEST.json';seal=data/'MANIFEST.sha256'
    require(manifest_path.is_file() and seal.is_file(),'Scientific release manifest is unavailable')
    require(sha256(manifest_path)==seal.read_text(encoding='utf8').split()[0],'Scientific manifest seal differs')
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    for item in manifest['files']:
        path=(data/item['path']).resolve()
        require(path.is_relative_to(data.resolve()),'Manifest path leaves scientific release')
        require(path.is_file() and path.stat().st_size==item['size'] and sha256(path)==item['sha256'],
            'Scientific release file hash/size mismatch: '+item['path'])
    return {'status':'passed','files_checked':len(manifest['files']),'sha256':sha256(manifest_path)}


def verify_model_manifest(models=MODELS):
    path=models/'manifest_sha256.json'
    require(path.is_file(),'Released-model manifest unavailable')
    manifest=json.loads(path.read_text(encoding='utf8'))
    for item in manifest['files']:
        resource=(models/item['path']).resolve()
        require(resource.is_relative_to(models.resolve()),'Model manifest path leaves model release')
        require(resource.is_file() and resource.stat().st_size==item['size'] and sha256(resource)==item['sha256'],
            'Released model hash/size mismatch: '+item['path'])
    return {'status':'passed','files_checked':len(manifest['files']),'sha256':sha256(path)}


def load_frames(data=DATA):
    options={'dtype':{'patient_id':'string','pid':'string','sample_id':'string'}}
    return {
        'patients':pd.read_csv(data/'metadata/patients.csv',**options),
        'samples':pd.read_csv(data/'metadata/samples.csv',**options),
        'clinical':pd.read_csv(data/'clinical/clinical_inputs.csv',**options),
        'oof':pd.read_csv(data/'oof/predictions.csv',**options),
        'test':pd.read_csv(data/'test/predictions.csv',**options),
        'order':pd.read_csv(data/'partitions/eligible_oof_1092.csv',**options),
        'development':pd.read_csv(data/'partitions/development_1097.csv',**options),
        'test_partition':pd.read_csv(data/'partitions/test_176.csv',**options),
    }


def verify_coverage(frames,data=DATA):
    p=frames['patients'].rename(columns={'pid':'patient_id'}).set_index('patient_id')
    s=frames['samples'];c=frames['clinical'];q=frames['oof'];t=frames['test'];order=frames['order']
    require(len(p)==1273 and p.index.is_unique and len(s)==12703,'Unexpected public teaching-case coverage')
    require(set(p.split)=={'train','test'},'Unexpected source split, including possible private validation')
    development=set(p.index[p.split=='train']);testing=set(p.index[p.split=='test'])
    require(len(development)==1097 and len(testing)==176 and not development&testing,'Public train/test identity mismatch')
    require(len(c)==1273 and c.patient_id.is_unique and set(c.patient_id)==set(p.index),'Clinical inputs leave public cohort')
    require(np.array_equal(c.label.to_numpy(),p.loc[c.patient_id,'pCR'].to_numpy()),'Clinical outcomes differ from public source')
    require(np.array_equal(c.split.to_numpy(),p.loc[c.patient_id,'split'].to_numpy()),'Clinical splits differ from public source')
    for feature in ['age','HR']:
        actual=pd.to_numeric(c[feature],errors='coerce').replace(-1,np.nan).to_numpy()
        public=pd.to_numeric(p.loc[c.patient_id,feature],errors='coerce').replace(-1,np.nan).to_numpy()
        require(np.isclose(actual,public,equal_nan=True).all(),'Clinical '+feature+' differs from public teaching source')
    require(len(q)==1092 and q.patient_id.is_unique and set(q.patient_id)<=development,'OOF patient coverage mismatch')
    eligible=set(c.loc[c.split.eq('train') & c.eligible_primary.astype(str).str.lower().isin(['true','1']),'patient_id'])
    require(set(q.patient_id)==eligible,'Primary eligibility and OOF identities differ')
    require(order.row_position.tolist()==list(range(1092)) and order.patient_id.tolist()==q.patient_id.tolist(),
        'OOF patient order differs from the original row-position reference')
    require(np.array_equal(order.fold.to_numpy(),q.fold.to_numpy()),'OOF order reference folds differ')
    require(np.array_equal(q.pCR.to_numpy(),p.loc[q.patient_id,'pCR'].to_numpy()),'OOF outcomes differ from public source')
    require(set(q.fold)==set(range(5)),'OOF fold identifiers mismatch')
    foldmap=s[s.split=='train'].groupby('patient_id').fold.first()
    require(np.array_equal(q.fold.to_numpy(),foldmap.loc[q.patient_id].to_numpy()),'OOF folds differ from public source')
    require(len(t)==176 and t.patient_id.is_unique and set(t.patient_id)==testing,'Historical TEST coverage mismatch')
    require(np.array_equal(t.y_true.to_numpy(),p.loc[t.patient_id,'pCR'].to_numpy()),'TEST outcomes differ from public source')
    require(set(frames['development'].patient_id)==development and set(frames['test_partition'].patient_id)==testing,
        'Released development/test partition identity mismatch')
    sample_lookup=s.set_index('sample_id')
    for fold in range(5):
        partition=pd.read_csv(data/f'partitions/fold_{fold}.csv',dtype={'patient_id':'string','sample_id':'string'})
        require(set(partition.sample_id)<=set(sample_lookup.index),'Historical partition has nonpublic sample IDs')
        reference=sample_lookup.loc[partition.sample_id]
        require(np.array_equal(partition.patient_id.to_numpy(),reference.patient_id.to_numpy()),'Historical sample/patient association differs')
        require(np.array_equal(partition.fold.to_numpy(),reference.fold.to_numpy()),'Historical sample fold differs')
        groups={role:set(partition.loc[partition.role==role,'patient_id']) for role in ['fit','inner','refit','outer']}
        require(not groups['refit']&groups['outer'] and groups['refit']|groups['outer']==development,'Fold fitting/outer leakage or incomplete coverage')
        require(not groups['fit']&groups['inner'] and groups['fit']|groups['inner']==groups['refit'],'Historical inner split mismatch')
        require(groups['outer']==set(foldmap.index[foldmap==fold]),'Historical outer partition mismatch')
    for column in ['image','tabular','fusion','linear_fusion']:
        require(np.isfinite(q[column]).all() and q[column].between(0,1).all(),'Invalid OOF probabilities')
    require(np.isfinite(t.probability).all() and t.probability.between(0,1).all(),'Invalid TEST probabilities')
    return {'status':'passed','public_patients':1273,'public_triplets':12703,'development':1097,'eligible_oof':1092,
        'historical_test':176,'oof_order_matches':True,'each_oof_patient_once':True,
        'oof_fold_counts':[int((q.fold==fold).sum()) for fold in range(5)],'public_source_membership':True,
        'private_teacher_validation_included':False}


def replay_stored_metrics(frames,data=DATA):
    q=frames['oof'];t=frames['test']
    original=json.loads((data/'metrics/oof_original.json').read_text(encoding='utf8'))
    historical=json.loads((data/'metrics/test_original.json').read_text(encoding='utf8'))
    result={}
    for name in ['image','tabular','fusion','linear_fusion']:
        actual=original_metrics(q.pCR,q[name],name);expected=original['models'][name]
        delta={metric:actual[metric]-expected[metric] for metric in ['auc','bce']}
        require(abs(delta['auc'])<=1e-14 and abs(delta['bce'])<=1e-14,'Original OOF metric mismatch for '+name)
        result[name]={'metrics':actual,'difference_from_original':delta,'score_dtype':SCORE_DTYPES[name],'replay':'original patient-score metric recomputation'}
    actual=original_metrics(t.y_true,t.probability,'historical_test')
    delta={metric:actual[metric]-historical[metric] for metric in ['auc','bce']}
    require(abs(delta['auc'])<=1e-14 and abs(delta['bce'])<=1e-14,'Historical TEST metric mismatch')
    return {'oof_protocol':'Five patient-disjoint OOF folds, 1092 eligible development patients',
        'oof':result,'historical_cnn_ema_test':{'metrics':actual,'difference_from_original':delta,
        'replay':'original 176-patient score metric recomputation; MRI encoder not rerun',
        'protocol_separate_from_oof':True,'test_previously_used_with_other_model':historical['test_previously_used_with_other_model']},
        'final_multimodal_checkpoint_has_independent_validation':False}


def replay_models(frames,data=DATA,models=MODELS):
    manifest=verify_model_manifest(models)
    torch.set_num_threads(4)
    q=frames['oof'];clinical=frames['clinical'].set_index('patient_id').loc[q.patient_id]
    seen=np.zeros(len(q),dtype=int);replayed={name:np.full(len(q),np.nan) for name in ['image','tabular','fusion']}
    fold_checks=[]
    ledger=json.loads((data/'protocols/cnn_oof_ledger.json').read_text(encoding='utf8'))
    checkpoint_hash={item['outer_fold']:item['checkpoint_sha256'] for item in ledger['folds']}
    dev=set(frames['development'].patient_id)
    with torch.inference_mode():
        for fold in range(5):
            selected=q.fold.to_numpy()==fold;ids=q.loc[selected,'patient_id'].tolist();seen[selected]+=1
            z=np.load(data/f'embeddings/fold_{fold}.npz',allow_pickle=False)
            require(str(z['extractor_version'])=='loaded_state_v2' and float(z['parity_error'])<1e-7,'Invalid or random-initialization embedding cache')
            require(set(z['ids'])==dev and z['embedding'].shape==(1097,64),'Embedding patient coverage mismatch')
            require(str(z['checkpoint_sha'])==checkpoint_hash[fold],'Embedding/fold checkpoint lineage mismatch')
            lookup={str(identifier):index for index,identifier in enumerate(z['ids'])}
            embeddings=np.asarray([z['embedding'][lookup[identifier]] for identifier in ids],dtype=np.float32)
            original_scores=np.asarray([z['score'][lookup[identifier]] for identifier in ids])
            require(np.max(np.abs(original_scores-q.loc[selected,'image'].to_numpy()))<=1e-14,'Original CNN scores and OOF cache differ')
            model,batchnorm=load_cnn(models/f'cnn_oof/fold_{fold}')
            cnn_metadata=load_checkpoint(models/f'cnn_oof/fold_{fold}')
            require(cnn_metadata.get('fold')==fold and cnn_metadata.get('source_checkpoint_sha256')==str(z['checkpoint_sha']),'Released CNN/embedding lineage differs')
            replayed['image'][selected]=model._classify(torch.from_numpy(embeddings)).sigmoid().numpy().reshape(-1)
            pure_clinical=ClinicalFoldPredictor.load(models/f'clinical_oof/fold_{fold}/model.json')
            require(pure_clinical.fold==fold,'Clinical fold differs')
            replayed['tabular'][selected]=pure_clinical.predict_proba(clinical.loc[selected,['age','HR']])[:,1]
            bundle=load_checkpoint(models/f'fusion_oof/fold_{fold}')
            require(bundle.get('fold')==fold and bundle.get('image_checkpoint_sha256')==str(z['checkpoint_sha']),'Fusion head/embedding lineage differs')
            encoded=encode_clinical(clinical.loc[selected,['age','HR']],bundle['clinical_transform'])
            normalized=(embeddings-np.asarray(bundle['embedding_mean'],dtype=np.float32))/np.asarray(bundle['embedding_std'],dtype=np.float32)
            net=FusionNet();net.load_state_dict(bundle['head_state'],strict=True);net.eval()
            replayed['fusion'][selected]=net(torch.from_numpy(normalized),torch.from_numpy(encoded)).sigmoid().numpy()
            fold_checks.append({'fold':fold,'n':len(ids),'cnn_embedding_checkpoint_match':True,
                'clinical_max_probability_error':float(np.max(np.abs(replayed['tabular'][selected]-q.loc[selected,'tabular'].to_numpy()))),
                'cnn_classifier_max_probability_error':float(np.max(np.abs(replayed['image'][selected]-q.loc[selected,'image'].to_numpy()))),
                'fusion_max_probability_error':float(np.max(np.abs(replayed['fusion'][selected]-q.loc[selected,'fusion'].to_numpy(dtype='float32'))))})
        require((seen==1).all(),'Model replay coverage/order mismatch')
        metrics={}
        for name,probability in replayed.items():
            require(np.isfinite(probability).all(),'Nonfinite model replay probabilities')
            error=float(np.max(np.abs(probability-q[name].to_numpy(dtype=SCORE_DTYPES[name]))))
            tolerance=1e-14 if name=='tabular' else 1e-6
            require(error<=tolerance,'Model probability replay exceeds documented tolerance: '+name)
            actual=original_metrics(q.pCR,probability,name);reference=original_metrics(q.pCR,q[name],name)
            differences={key:actual[key]-reference[key] for key in ['auc','bce']}
            require(abs(differences['auc'])<=1e-6 and abs(differences['bce'])<=1e-6,'Model replay metric mismatch: '+name)
            metrics[name]={'metrics':actual,'max_probability_error':error,'probability_tolerance':tolerance,'metric_difference':differences}
        z=np.load(data/'embeddings/cnn_ema_development.npz',allow_pickle=False)
        require(set(z['ids'])==dev and str(z['extractor_version'])=='loaded_state_v2','Historical EMA feature coverage mismatch')
        ema,_=load_cnn(models/'cnn_ema');meta=load_checkpoint(models/'cnn_ema')
        require(meta['source_checkpoint_sha256']==str(z['checkpoint_sha']),'EMA CNN/feature checkpoint mismatch')
        ema_score=ema._classify(torch.tensor(z['embedding'],dtype=torch.float32)).sigmoid().numpy().reshape(-1)
        ema_error=float(np.max(np.abs(ema_score-z['score'])))
        require(ema_error<=1e-6,'EMA classifier replay exceeds tolerance')
        final=load_checkpoint(models/'multimodal_final')
        require(final['image_checkpoint_sha256']==str(z['checkpoint_sha']),'Final fusion/EMA feature lineage mismatch')
    return {'status':'passed','model_manifest':manifest,'folds':fold_checks,'oof_model_replay':metrics,
        'cnn_ema_development_classifier_max_probability_error':ema_error,
        'MRI_encoder_recomputed':False,'linear_fusion':'Only stored-score metrics replayed: original fitted weights were not persisted; no substitute was trained.',
        'final_multimodal_head':'Present; EMA lineage checked; no independent performance assigned.'}


def reproduce(data=DATA,models=MODELS,metrics_only=False):
    manifest=verify_manifest(data);frames=load_frames(data)
    coverage=verify_coverage(frames,data);metrics=replay_stored_metrics(frames,data)
    replay=None if metrics_only else replay_models(frames,data,models)
    return {'status':'passed','manifest':manifest,'coverage':coverage,'stored_score_metrics':metrics,
        'model_replay':replay,'training_executed':False,'MRI_images_required':False,
        'scope':'Scientific score/embedding/head replay, not MRI image-encoder reproduction or new independent validation'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-only',action='store_true',help='Explicitly skip checkpoint/head reload; replay original scores only')
    parser.add_argument('--output',type=Path,help='Optional NEW JSON report; default prints aggregate results only')
    args=parser.parse_args()
    try:
        report=reproduce(metrics_only=args.metrics_only)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with args.output.open('x',encoding='utf8') as handle:json.dump(report,handle,indent=2,allow_nan=False)
        print(json.dumps(report,indent=2,allow_nan=False))
    except (ValueError,KeyError,FileNotFoundError,FileExistsError) as exc:parser.exit(2,f'Error: {exc}\n')


if __name__=='__main__':main()
