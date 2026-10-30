"""Synthetic software fixtures ONLY: never evidence of scientific performance."""
from io import BytesIO
from pathlib import Path
import json
import zipfile
import os
import subprocess
import sys
import runpy
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from breastdce_research.models import CompactPatientCNN
from breastdce_research.multimodal import FusionNet,fit_fusion_head
from breastdce_research.baselines import fit_linear_fusion
from breastdce_research.clinical import ClinicalFoldPredictor,fit_clinical,encode_clinical
from breastdce_research.averaging import ema_parameters,recalibrate_batchnorm
from breastdce_research.evaluation import patient_split,binary_metrics,partition_lineage,verify_partition_lineage,verify_epoch_lineage
from breastdce_research.inference import load_cnn,load_checkpoint,save_checkpoint_new
from breastdce_research.imaging import preprocess,standardize

torch.set_num_threads(1)


def test_original_architecture_and_complete_bag_invariance():
    model=CompactPatientCNN().eval()
    assert model.n_parameters==24713
    images=torch.zeros(2,3,256,256)
    images[1]=1
    with torch.inference_mode():
        h=model._encode(images)
        assert h.shape==(2,64)
        a=model.forward_patient(images,1);b=model.forward_patient(images,2)
        assert torch.allclose(a,b,atol=1e-6)
        assert torch.allclose(a,model._classify(h.mean(0,keepdim=True)),atol=1e-6)
        padded=torch.full((1,3,3,256,256),float('nan'));padded[0,:2]=images
        assert torch.allclose(a,model(padded,torch.tensor([[True,True,False]])),atol=1e-6)
    with pytest.raises(ValueError):model.forward_patient(images,0)
    with pytest.raises(ValueError):model(padded,torch.tensor([[False,False,False]]))


def synthetic_specification():
    # Artificial unit-test parameters, never the private recovered clinical models.
    return {'format':'clinical_logistic_fold_json_v1','classes':[0,1],'fold':0,
        'raw_feature_order':['age','HR'],'transformed_feature_order':['age_scaled','HR_scaled','age_missing','HR_missing'],
        'clinical_transform':{'features':['age','HR'],'median':[40.,0.],'mean':[40.,.5],'std':[10.,.5]},
        'coefficients':[0.,0.,0.,0.],'intercept':0.,'threshold_reference':.5}


def test_clinical_serialization_order_missing_and_input_guards(tmp_path):
    model_file=tmp_path/'synthetic-unit-model.json';model_file.write_text(json.dumps(synthetic_specification()))
    model=ClinicalFoldPredictor.load(model_file)
    records=pd.DataFrame({'age':[40.,None],'HR':[1.,None]})
    assert model.transform(records).dtype==np.float32
    assert np.array_equal(model.predict_proba(records),model.predict_proba(records[['HR','age']]))
    assert np.array_equal(model.transform({'age':-1,'HR':-1}),model.transform({'age':None,'HR':None}))
    assert model.predict_proba([]).shape==(0,2)
    for record in [{'age':12,'HR':1},{'age':50,'HR':3},{'age':float('inf'),'HR':1},{'HR':1}]:
        with pytest.raises(ValueError):model.predict_proba(record)


def test_train_only_transform_is_not_refitted_during_encoding():
    training=pd.DataFrame({'age':[30.,50.],'HR':[0.,1.]})
    state=fit_clinical(training);sealed=json.dumps(state,sort_keys=True)
    encoded=encode_clinical([{'age':100.,'HR':1.}],state)
    assert encoded[0,0]==6.0
    assert json.dumps(state,sort_keys=True)==sealed


def fixture_metadata():
    return pd.DataFrame({'patient_id':['fixture-a','fixture-b','fixture-c','fixture-d'],
        'label':[0,1,0,1],'fold':[0,0,1,1],
        'pre_path':['a-pre','b-pre','c-pre','d-pre'],'early_path':['a-early','b-early','c-early','d-early'],
        'late_path':['a-late','b-late','c-late','d-late']})


def test_patient_partition_rejects_leakage_and_conflicts():
    frame=fixture_metadata();train,heldout=patient_split(frame,0)
    assert not set(train.patient_id)&set(heldout.patient_id)
    duplicate=frame.iloc[0].copy();duplicate['fold']=1;duplicate['pre_path']='other'
    with pytest.raises(ValueError):patient_split(pd.concat([frame,pd.DataFrame([duplicate])]),0)
    frame.loc[1,['pre_path','early_path','late_path']]=frame.loc[0,['pre_path','early_path','late_path']].to_numpy()
    with pytest.raises(ValueError):patient_split(frame,0)


def test_shared_single_phase_is_rejected_across_patients_and_folds(tmp_path):
    frame=fixture_metadata()
    # Only PRE is shared; EARLY/LATE remain different, so full-triplet checks
    # alone would miss this cross-patient/cross-fold leakage.
    frame.loc[2,'pre_path']='folder/../a-pre'
    with pytest.raises(ValueError,match='Shared phase'):
        patient_split(frame,0)
    with pytest.raises(ValueError,match='Shared phase'):
        patient_split(frame,0,image_root=tmp_path)


def test_checkpoint_partition_lineage_rejects_wrong_fold_metadata_and_absence():
    frame=fixture_metadata();lineage=partition_lineage(frame,0)
    assert verify_partition_lineage(lineage,frame.iloc[::-1],0)==lineage
    with pytest.raises(ValueError,match='mismatch'):verify_partition_lineage(lineage,frame,1)
    changed=frame.copy();changed.loc[0,'pre_path']='changed-fixture-path'
    with pytest.raises(ValueError,match='mismatch'):verify_partition_lineage(lineage,changed,0)
    with pytest.raises(ValueError,match='missing'):verify_partition_lineage(None,frame,0)
    normalization={'means':[0.,0.,0.],'std':[1.,1.,1.]}
    checkpoints=[{'epoch':epoch,'partition_lineage':lineage,'normalization':normalization} for epoch in [1,2]]
    assert verify_epoch_lineage(checkpoints,frame,0)==lineage
    with pytest.raises(ValueError,match='mismatch'):
        verify_epoch_lineage([checkpoints[0],{**checkpoints[1],'partition_lineage':partition_lineage(frame,1)}],frame,0)
    with pytest.raises(ValueError,match='Normalization'):
        verify_epoch_lineage([checkpoints[0],{**checkpoints[1],'normalization':{'means':[1.,1.,1.],'std':[1.,1.,1.]}}],frame,0)
    with pytest.raises(ValueError,match='chronological'):
        verify_epoch_lineage([checkpoints[0],{**checkpoints[1],'epoch':3}],frame,0)


def test_evaluate_cli_rejects_wrong_fold_before_reading_images(tmp_path):
    root=Path(__file__).resolve().parents[1]
    frame=fixture_metadata();metadata=tmp_path/'synthetic-fixture-metadata.csv';frame.to_csv(metadata,index=False)
    checkpoint=tmp_path/'synthetic-fixture-checkpoint'
    save_checkpoint_new(checkpoint,{'model_state_dict':CompactPatientCNN().state_dict(),
        'normalization':{'means':[0.,0.,0.],'std':[1.,1.,1.]},
        'config':{'search_variant':'compact'},'partition_lineage':partition_lineage(frame,0)})
    output=tmp_path/'must-not-be-created.json';env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    result=subprocess.run([sys.executable,'-B',str(root/'scripts/evaluate.py'),'--checkpoint',str(checkpoint),
        '--metadata',str(metadata),'--image-root',str(tmp_path/'nonexistent-images'),'--fold','1','--output',str(output)],
        env=env,cwd=root,capture_output=True,text=True)
    assert result.returncode==2 and 'lineage mismatch' in result.stderr and not output.exists()


def test_restricted_checkpoint_and_missing_weight_failure(tmp_path):
    with pytest.raises(FileNotFoundError,match='no trained weights'):load_cnn(tmp_path/'missing.pt')
    bad=tmp_path/'not-a-checkpoint.pt';bad.write_bytes(b'not pickle or tensors')
    with pytest.raises(ValueError,match='JSON/NPZ'):load_checkpoint(bad)
    model=CompactPatientCNN();path=tmp_path/'synthetic-unit-state'
    save_checkpoint_new(path,{'model_state_dict':model.state_dict(),'normalization':{'means':[0.,0.,0.],'std':[1.,1.,1.]},'config':{'search_variant':'compact'}})
    loaded,state=load_cnn(path)
    assert loaded.n_parameters==24713 and not loaded.training
    assert all(torch.equal(model.state_dict()[k],loaded.state_dict()[k]) for k in model.state_dict())
    with pytest.raises(FileExistsError):save_checkpoint_new(path,{})


def test_npz_object_payload_is_rejected_without_pickle(tmp_path):
    path=tmp_path/'synthetic-malformed-bundle';path.mkdir()
    (path/'metadata.json').write_text(json.dumps({'format':'breastdce-json-npz-v1'}))
    # Plain object array fixture, no executable pickle payload.
    np.savez(path/'weights.npz',**{'model_state_dict::bad':np.array([{}],dtype=object)})
    with pytest.raises(ValueError,match='non-pickle'):load_checkpoint(path)


def test_npz_header_shape_is_bounded_before_allocation(tmp_path):
    path=tmp_path/'synthetic-header-bundle';path.mkdir()
    (path/'metadata.json').write_text(json.dumps({'format':'breastdce-json-npz-v1'}))
    buffer=BytesIO()
    np.lib.format.write_array_header_1_0(buffer,{'descr':'<f8','fortran_order':False,'shape':(1_000_001,)})
    with zipfile.ZipFile(path/'weights.npz','w') as archive:
        archive.writestr('model_state_dict::example.npy',buffer.getvalue())
    with pytest.raises(ValueError,match='non-pickle'):load_checkpoint(path)


def test_epoch_ema_and_bn_change_only_buffers():
    model=CompactPatientCNN();first={k:v.clone() for k,v in model.state_dict().items()}
    second={k:v.clone() for k,v in first.items()}
    name=next(iter(dict(model.named_parameters())))
    second[name]=first[name]+1
    averaged=ema_parameters(model,[first,second],.8)
    assert torch.allclose(averaged[name],first[name]+.2,atol=1e-6)
    before={k:v.clone() for k,v in model.named_parameters()}
    result=recalibrate_batchnorm(model,[torch.zeros(1,3,256,256)],{'fixture-train'},{'fixture-heldout'})
    assert result['n_slices']==1 and not model.training
    assert all(torch.equal(v,before[k]) for k,v in model.named_parameters())
    with pytest.raises(ValueError):recalibrate_batchnorm(model,[],{'fixture-a'},{'fixture-a'})


def test_phase_order_and_scaling_are_not_silently_changed():
    def png(value):
        buffer=BytesIO();Image.new('L',(256,256),value).save(buffer,format='PNG');return buffer.getvalue()
    images=preprocess({'LATE':png(255),'PRE':png(0),'EARLY':png(127)})
    assert images.shape==(3,256,256)
    assert float(images[0,0,0])==0 and float(images[2,0,0])==1
    assert torch.equal(images,standardize(images,{'means':[0.,0.,0.],'std':[1.,1.,1.]}))
    with pytest.raises(ValueError):preprocess({'PRE':png(0)})


def test_fusion_dimensions_and_metrics_guards():
    net=FusionNet().eval();assert sum(p.numel() for p in net.parameters())==1225
    assert net(torch.zeros(2,64),torch.zeros(2,4)).shape==(2,)
    with pytest.raises(ValueError):net(torch.zeros(2,128),torch.zeros(2,4))
    # Tiny synthetic metric sanity check; not any scientific cohort/result.
    assert binary_metrics([0,1],[.1,.9])['auc']==1
    with pytest.raises(ValueError):binary_metrics([0,1],[float('nan'),.5])


def test_invalid_fit_requests_rejected_without_training():
    for epochs in [0,-1]:
        with pytest.raises(ValueError,match='epochs'):
            fit_fusion_head(np.zeros((2,64)),np.zeros((2,4)),[0,1],epochs=epochs)
    with pytest.raises(ValueError,match='Nonempty'):
        fit_fusion_head(np.zeros((0,64)),np.zeros((0,4)),[])
    with pytest.raises(ValueError,match='binary pCR'):
        fit_linear_fusion(np.zeros((2,64)),np.zeros((2,4)),[2,3])
    with pytest.raises(ValueError,match='binary pCR'):
        fit_linear_fusion(np.zeros((3,64)),np.zeros((3,4)),[0,1,2])


def test_static_server_does_not_allow_nested_readme_bypass(tmp_path):
    root=Path(__file__).resolve().parents[1]
    handler_class=runpy.run_path(str(root/'scripts/serve_web.py'))['PublicHandler']
    handler=object.__new__(handler_class);handler.directory=str(tmp_path)
    (tmp_path/'private').mkdir();(tmp_path/'private/README.md').write_text('synthetic unit fixture')
    (tmp_path/'README.md').write_text('synthetic public fixture')
    assert Path(handler.translate_path('/README.md'))==tmp_path/'README.md'
    assert Path(handler.translate_path('/private/README.md'))==tmp_path/'__not_public__'
    assert Path(handler.translate_path('/private/LICENSE'))==tmp_path/'__not_public__'
    assert Path(handler.translate_path('/.git/config'))==tmp_path/'__not_public__'


def test_prepare_adapter_preserves_fixture_outcomes_and_folds(tmp_path):
    root=Path(__file__).resolve().parents[1];images=tmp_path/'synthetic-images';images.mkdir()
    frame=fixture_metadata().rename(columns={'label':'pCR','pre_path':'path_pre','early_path':'path_early','late_path':'path_late'})
    frame['split']='train'
    # Metadata adapter checks availability only. These are explicit non-medical
    # unit-test file placeholders, not a scientific MRI example or result.
    for column in ['path_pre','path_early','path_late']:
        for name in frame[column]:(images/name).write_bytes(b'synthetic-file-placeholder')
    original=tmp_path/'synthetic-original.csv';frame.to_csv(original,index=False)
    output=tmp_path/'synthetic-canonical.csv';env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    result=subprocess.run([sys.executable,'-B',str(root/'scripts/prepare_data.py'),'--metadata',str(original),
        '--image-root',str(images),'--output',str(output)],env=env,cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    adapted=pd.read_csv(output)
    assert adapted.patient_id.tolist()==frame.patient_id.tolist()
    assert adapted.label.tolist()==frame.pCR.tolist() and adapted.fold.tolist()==frame.fold.tolist()
    assert len(adapted)==len(frame)
