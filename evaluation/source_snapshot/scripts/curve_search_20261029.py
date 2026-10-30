"""Bounded real checkpoint averaging search, same 878/219 split; no TEST."""
import os,sys,json,shutil,subprocess,ctypes,uuid,time,math
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'.venv/Scripts/python.exe').is_file())
sys.path.insert(0,str(ROOT/'scripts'))
import refit_1097_20261027 as base
B=ROOT/'artifacts/curve_search_20261029/prepared'

def prepare():
    if B.exists():raise FileExistsError(B)
    origin=ROOT/'artifacts/compacta_holdout_20261029/prepared'
    op=base.read(origin/'protocol.json');result=base.read(origin/'result.json');run=Path(result['run_dir'])
    B.mkdir(parents=True)
    shutil.copytree(origin/'source_snapshot/src',B/'source_snapshot/src',ignore=shutil.ignore_patterns('__pycache__'))
    (B/'source_snapshot/scripts').mkdir()
    for name in [Path(__file__).name,'refit_1097_20261027.py','bn_recalibration_utils_20261028.py']:
        shutil.copy2(ROOT/'scripts'/name,B/'source_snapshot/scripts'/name)
    shutil.copytree(origin/'metadata',B/'metadata')
    shutil.copy2(run/'normalization.json',B/'normalization.json')
    policy={'arms':['raw_bn','weight_mean','weight_ema08','prediction_mean','prediction_ema08'],
      'epochs':15,'average_prefix':'all checkpoints1..e, no future checkpoint, no label dependent weights',
      'ema_decay':.8,'bn':'recompute on all878 TRAIN patients/8759 slices, labels absent, dropout disabled, batch64, seeded order42',
      'primary_epoch':15,'primary_gates':{'validation_auc_min':.54,'validation_bce_below_train_prior':True,'mean_absolute_gap_epochs2_15_max':.015,'train_and_val_net_drop_epochs2_15':True,'train_and_val_negative_ols_epochs2_15':True,'val_mean_absolute_step_below_raw':True},
      'rank':'among gate pass arms: mean absolute gap plus validation mean absolute step, epochs2..15',
      'fallback':'terminal AUC>=.54, rank same score; explicitly mark joint objective not achieved',
      'max_gpu_seconds':7200,'test_used':False,'new_gradient_training':False,'architecture_unchanged':True,
      'limitation':'Repeatedly reused development holdout; conditional exploration, not external confirmation'}
    p={'policy':policy,'config':op['config'],'image_root':op['image_root'],'cache_dir':op['cache_dir'],'raw_run':str(run),
       'raw_files_sha256':{str(f):base.sha(f) for f in run.iterdir() if f.suffix in ['.pt','.csv'] or f.name=='history.json'},
       'files_sha256':{f.relative_to(B).as_posix():base.sha(f) for f in B.rglob('*') if f.is_file()},
       'cache_manifest_sha256':base.sha(Path(op['cache_dir'])/'manifest.json')}
    base.write(B/'protocol.json',p);(B/'protocol.json.sha256').write_text(base.sha(B/'protocol.json'))
    print('Prepared five fixed averaging variants, 878/219 split, 15 checkpoints each.',flush=True)

def verify(bundle):
    p=base.read(bundle/'protocol.json')
    assert base.sha(bundle/'protocol.json')==(bundle/'protocol.json.sha256').read_text().strip()
    for name,digest in p['files_sha256'].items():assert base.sha(bundle/name)==digest,name
    for name,digest in p['raw_files_sha256'].items():assert base.sha(Path(name))==digest,name
    assert base.sha(Path(p['cache_dir'])/'manifest.json')==p['cache_manifest_sha256']
    return p

def prediction_metrics(frame,probs):
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score
    probs=np.clip(probs,1e-12,1-1e-12);y=frame.y_true.to_numpy()
    out=frame.copy();out['score']=probs;out['logit']=np.log(probs)-np.log1p(-probs)
    out['bce']=-(y*np.log(probs)+(1-y)*np.log1p(-probs))
    return out,{'n':len(y),'n_positive':int(y.sum()),'bce':float(out.bce.mean()),'roc_auc':float(roc_auc_score(y,probs)),'average_precision':float(average_precision_score(y,probs))}

def run_worker():
    p=verify(B);sys.path.insert(0,str(B/'source_snapshot/src'));sys.path.insert(0,str(B/'source_snapshot/scripts'))
    import torch,pandas as pd,numpy as np
    from breastdcedl.data.validation import load_development
    from breastdcedl.data.image_cache import ImageCache
    from breastdcedl.university.data import PatientBagDataset
    from breastdcedl.university.config import PatientConfig
    from breastdcedl.university.training import build_patient_model,evaluate
    from bn_recalibration_utils_20261028 import LabelFreeTrainingSlices,recalibrate,verify_state_change
    torch.set_num_threads(8);torch.manual_seed(42)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    raw=Path(p['raw_run']);rows,_=load_development(B/'metadata');fit=rows[rows.fold!=0];val=rows[rows.fold==0]
    assert fit.patient_id.nunique()==878 and val.patient_id.nunique()==219 and not set(fit.patient_id)&set(val.patient_id)
    norm=base.read(B/'normalization.json');assert set(norm['fit_patient_ids'])==set(fit.patient_id) and set(norm['fit_sample_ids'])==set(fit.sample_id)
    cache=ImageCache(Path(p['cache_dir']),rows)
    datasets={s:PatientBagDataset(r,p['image_root'],norm,train=False,cache=cache) for s,r in [('train',fit),('validation',val)]}
    bn_data=LabelFreeTrainingSlices(fit,set(fit.patient_id),set(val.patient_id),cache,norm)
    assert len(bn_data)==8759
    histories={a:[] for a in p['policy']['arms']}
    for arm in histories:(B/arm).mkdir(exist_ok=False)
    accum={};ema={};ids={}
    # Probability ensembles are distinct multi-checkpoint predictors, not averaged loss values.
    for e in range(1,16):
        entries={a:{'epoch':e} for a in ['prediction_mean','prediction_ema08']}
        for split in ['train','validation']:
            df=pd.read_csv(raw/f'epoch_{e:03d}_{split}_patients.csv').sort_values('patient_id').reset_index(drop=True)
            key=list(zip(df.patient_id,df.y_true,df.n_slices))
            if split in ids:assert ids[split]==key
            ids[split]=key;probs=df.score.to_numpy()
            accum[split]=probs.copy() if e==1 else accum[split]+probs
            ema[split]=probs.copy() if e==1 else .8*ema[split]+.2*probs
            for arm,pred in [('prediction_mean',accum[split]/e),('prediction_ema08',ema[split])]:
                f,m=prediction_metrics(df,pred);f.to_csv(B/arm/f'epoch_{e:03d}_{split}_patients.csv',index=False);entries[arm][split]=m
        for arm,entry in entries.items():histories[arm].append(entry);base.write(B/arm/'history.json',histories[arm])
    started=time.monotonic();averages={}
    for e in range(1,16):
        source=raw/f'epoch_{e:03d}.pt';payload=torch.load(source,map_location='cpu',weights_only=False)
        assert payload['epoch']==e and payload['config']==p['config']
        state=payload['model_state_dict'];model=build_patient_model(PatientConfig(**p['config']))
        assert model.n_parameters==24713
        names=set(dict(model.named_parameters()))
        for arm in ['raw_bn','weight_mean','weight_ema08']:
            assert time.monotonic()-started<7000
            if arm!='raw_bn':
                if e==1:averages[arm]={n:state[n].clone() for n in names}
                else:
                    for n in names:
                        if arm=='weight_mean':averages[arm][n].add_((state[n]-averages[arm][n])/e)
                        else:averages[arm][n].mul_(.8).add_(state[n],alpha=.2)
            current={n:v.clone() for n,v in state.items()}
            if arm!='raw_bn':current.update({n:v.clone() for n,v in averages[arm].items()})
            model.load_state_dict(current,strict=True);model=model.cuda().eval()
            gen=torch.Generator().manual_seed(42)
            loader=torch.utils.data.DataLoader(bn_data,batch_size=64,shuffle=True,num_workers=0,generator=gen)
            before={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
            counts=recalibrate(model.backbone,loader,torch.device('cuda'))
            changed,after=verify_state_change(before,model)
            assert counts['n_slices']==8759
            entry={'epoch':e,'source_checkpoint_sha256':base.sha(source),'bn':counts,'bn_changed_keys':changed}
            for split,data in datasets.items():
                df,metrics=evaluate(model,data,torch.device('cuda'),16,0,42,base.sha(B/'protocol.json'),e,split)
                assert len(df)==(878 if split=='train' else 219)
                df.to_csv(B/arm/f'epoch_{e:03d}_{split}_patients.csv',index=False);entry[split]=metrics
            cp=B/arm/f'epoch_{e:03d}.pt'
            torch.save({'model_state_dict':after,'config':p['config'],'normalization':norm,'epoch':e,'arm':arm,'protocol_sha256':base.sha(B/'protocol.json'),'source_epochs':list(range(1,e+1)) if arm!='raw_bn' else [e],'metrics':entry},cp)
            cp.with_suffix('.pt.sha256').write_text(base.sha(cp));histories[arm].append(entry);base.write(B/arm/'history.json',histories[arm])
            print(json.dumps({'epoch':e,'arm':arm,'train_bce':entry['train']['bce'],'val_bce':entry['validation']['bce'],'val_auc':entry['validation']['roc_auc']}),flush=True)
            model=model.cpu()
        if e==1:
            for arm in ['weight_mean','weight_ema08']:
                a=torch.load(B/'raw_bn/epoch_001.pt',map_location='cpu',weights_only=False)['model_state_dict']
                b=torch.load(B/arm/'epoch_001.pt',map_location='cpu',weights_only=False)['model_state_dict']
                assert all(torch.equal(a[k],b[k]) for k in a)
    base.write(B/'result.json',{'status':'completed','arms':list(histories),'epochs_per_arm':15,'seconds':time.monotonic()-started,'test_used':False,'new_gradient_training':False})

def launch():
    prepare();verify(B)
    from ejecutar_contexto_dilatado_gpu_20261025 import GUID,PowerStatus
    lib=ctypes.WinDLL('powrprof');kernel=ctypes.WinDLL('kernel32')
    get=lib.PowerGetEffectiveOverlayScheme;get.argtypes=[ctypes.POINTER(GUID)]
    setmode=lib.PowerSetActiveOverlayScheme;setmode.argtypes=[ctypes.POINTER(GUID)]
    original=GUID();assert get(ctypes.byref(original))==0
    status=PowerStatus();assert kernel.GetSystemPowerStatus(ctypes.byref(status)) and status.ACLineStatus==1
    maximum=GUID.from_buffer_copy(uuid.UUID('ded574b5-45a0-4f42-8737-46345c09c238').bytes_le)
    power={'original_overlay':str(uuid.UUID(bytes_le=bytes(original)))}
    base.write(B/'supervision.json',{'status':'running','supervisor_pid':os.getpid()})
    try:
        assert setmode(ctypes.byref(maximum))==0
        kernel.SetThreadExecutionState(0x80000003)
        with (B/'training.log').open('w') as log:
            child=subprocess.Popen([sys.executable,'-u',str(B/'source_snapshot/scripts'/Path(__file__).name),'worker'],stdout=log,stderr=subprocess.STDOUT)
            base.write(B/'supervision.json',{'status':'running','worker_pid':child.pid,'supervisor_pid':os.getpid()})
            try: code=child.wait(timeout=7200)
            except subprocess.TimeoutExpired:
                child.kill();child.wait();raise
        assert code==0,f'Training failed: {code}; see training.log'
        base.write(B/'supervision.json',{'status':'completed','returncode':code})
    except BaseException as exc:
        base.write(B/'supervision.json',{'status':'failed','error':str(exc)});raise
    finally:
        power['restore_result']=int(setmode(ctypes.byref(original)))
        restored=GUID();power['restore_query_result']=int(get(ctypes.byref(restored)))
        power['restored_overlay']=str(uuid.UUID(bytes_le=bytes(restored)))
        base.write(B/'power_mode.json',power);kernel.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker': run_worker()
    else: launch()
