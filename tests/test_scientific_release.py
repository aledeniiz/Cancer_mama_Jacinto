"""Actual released-score/head reproducibility checks; no generated cohort/fit."""
from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]
REPLAY=runpy.run_path(str(ROOT/'scripts/reproduce_results.py'))


def test_scientific_files_and_public_patient_coverage():
    assert REPLAY['verify_manifest']()['status']=='passed'
    report=REPLAY['verify_coverage'](REPLAY['load_frames']())
    assert report['eligible_oof']==1092 and report['historical_test']==176
    assert report['public_source_membership'] and not report['private_teacher_validation_included']


def test_original_scientific_auc_and_bce_recomputed_from_real_scores():
    report=REPLAY['replay_stored_metrics'](REPLAY['load_frames']())
    expected={'image':.5197346132128741,'tabular':.6450048398806163,
              'fusion':.6325522303783173,'linear_fusion':.6354722916834719}
    for name,auc in expected.items():
        actual=report['oof'][name]
        assert abs(actual['metrics']['auc']-auc)<1e-14
        assert abs(actual['difference_from_original']['bce'])<1e-14
    assert abs(report['historical_cnn_ema_test']['metrics']['auc']-.5523853351741065)<1e-14
    assert not report['final_multimodal_checkpoint_has_independent_validation']


def test_replay_rejects_patient_order_and_outcome_mismatch():
    frames=REPLAY['load_frames']()
    changed=dict(frames);changed['oof']=frames['oof'].iloc[::-1].copy()
    with pytest.raises(ValueError,match='order'):REPLAY['verify_coverage'](changed)
    changed=dict(frames);changed['oof']=frames['oof'].copy()
    changed['oof'].loc[0,'pCR']=1-changed['oof'].loc[0,'pCR']
    with pytest.raises(ValueError,match='outcomes'):REPLAY['verify_coverage'](changed)


def test_released_clinical_and_neural_heads_replay_original_embeddings():
    report=REPLAY['replay_models'](REPLAY['load_frames']())
    assert report['status']=='passed'
    assert len(report['folds'])==5
    assert report['oof_model_replay']['tabular']['max_probability_error']<1e-14
    assert report['oof_model_replay']['image']['max_probability_error']<1e-6
    assert report['oof_model_replay']['fusion']['max_probability_error']<1e-6
    assert not report['MRI_encoder_recomputed']
