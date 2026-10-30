"""Patient-disjoint contracts and continuous-score metrics; no private fixtures."""
import numpy as np
import hashlib
import json
import os
import posixpath
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score, brier_score_loss


def validate_patient_table(frame,image_root=None):
    required={"patient_id","label","fold","pre_path","early_path","late_path"}
    if not required <= set(frame):
        raise ValueError("Local slice CSV requires patient_id,label,fold,pre_path,early_path,late_path")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Metadata must be nonempty and complete")
    if not frame.label.isin([0,1]).all() or not frame.fold.isin(range(5)).all():
        raise ValueError("Labels must be binary and fold identifiers 0..4")
    grouped=frame.groupby("patient_id")[["label","fold"]].nunique()
    if (grouped > 1).any().any():
        raise ValueError("A patient cannot cross folds or have conflicting labels")
    # Repeated slices or a shared exact triplet would leak across patients/folds.
    if frame.duplicated(["pre_path","early_path","late_path"]).any():
        raise ValueError("Duplicate image triplet")
    # One physical phase file cannot belong to two supposedly independent
    # patients, even when the other phases in their triplets are different.
    owners={};root=Path(image_root).resolve() if image_root is not None else None
    for row in frame.itertuples(index=False):
        for column in ('pre_path','early_path','late_path'):
            relative=posixpath.normpath(str(getattr(row,column)).replace('\\','/'))
            if root is None:
                key=os.path.normcase(relative)
            else:
                resolved=(root/relative).resolve()
                if not resolved.is_relative_to(root):raise ValueError('Image path outside authorized root')
                key=os.path.normcase(str(resolved))
            owner=owners.setdefault(key,row.patient_id)
            if owner!=row.patient_id:
                raise ValueError('Shared phase image path across different patients')
    return frame


def patient_split(frame, fold,image_root=None):
    validate_patient_table(frame,image_root=image_root)
    if type(fold) is not int or fold not in range(5):
        raise ValueError("fold must be 0..4")
    train,validation=frame[frame.fold!=fold].copy(),frame[frame.fold==fold].copy()
    if train.empty or validation.empty or set(train.patient_id)&set(validation.patient_id):
        raise ValueError("Nonempty patient-disjoint partitions required")
    return train,validation


def partition_lineage(frame,fold):
    """Deterministic private metadata/split fingerprints, not patient records.

    This binds a generated checkpoint to its intended partition. It does not
    prove data rights, image content identity or an independently audited fit.
    """
    train,validation=patient_split(frame,fold)
    columns=['patient_id','label','fold','pre_path','early_path','late_path']
    records=sorted((str(row.patient_id),int(row.label),int(row.fold),
        str(row.pre_path),str(row.early_path),str(row.late_path))
        for row in frame[columns].itertuples(index=False))
    def fingerprint(value):
        return hashlib.sha256(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode('utf8')).hexdigest()
    return {'schema':'patient-partition-lineage-v1','heldout_fold':fold,
        'metadata_sha256':fingerprint({'columns':columns,'records':records}),
        'training_patient_sha256':fingerprint(sorted(str(x) for x in train.patient_id.unique())),
        'validation_patient_sha256':fingerprint(sorted(str(x) for x in validation.patient_id.unique())),
        'training_patient_count':int(train.patient_id.nunique()),
        'validation_patient_count':int(validation.patient_id.nunique()),
        'training_slice_count':len(train),'validation_slice_count':len(validation)}


def verify_partition_lineage(recorded,frame,fold):
    if not isinstance(recorded,dict) or recorded.get('schema')!='patient-partition-lineage-v1':
        raise ValueError('Checkpoint partition lineage is missing; held-out evaluation is not permitted')
    expected=partition_lineage(frame,fold)
    if recorded!=expected:
        raise ValueError('Checkpoint fold/metadata partition lineage mismatch; refusing a held-out claim')
    return expected


def verify_epoch_lineage(checkpoints,frame,fold):
    """Require the same recorded partition and normalizer in every EMA input."""
    if not checkpoints:raise ValueError('At least one chronological checkpoint required')
    lineage=verify_partition_lineage(checkpoints[0].get('partition_lineage'),frame,fold)
    normalization=checkpoints[0].get('normalization')
    if not isinstance(normalization,dict):raise ValueError('Training normalization missing')
    for checkpoint in checkpoints:
        verify_partition_lineage(checkpoint.get('partition_lineage'),frame,fold)
        if checkpoint.get('normalization')!=normalization:
            raise ValueError('Normalization differs across checkpoints')
    epochs=[checkpoint.get('epoch') for checkpoint in checkpoints]
    if any(type(epoch) is not int for epoch in epochs) or epochs!=list(range(1,len(epochs)+1)):
        raise ValueError('Expected chronological epochs 1..e without gaps or future epochs')
    return lineage


def binary_metrics(labels, probabilities):
    y=np.asarray(labels);p=np.asarray(probabilities,dtype=float)
    if y.ndim!=1 or p.shape!=y.shape or len(y)==0 or not np.isin(y,[0,1]).all():
        raise ValueError("Expected aligned nonempty binary labels and probabilities")
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any() or len(np.unique(y))!=2:
        raise ValueError("Finite probabilities and both outcome classes required")
    return {"n":len(y),"auc":float(roc_auc_score(y,p)),"bce":float(log_loss(y,p,labels=[0,1])),
        "average_precision":float(average_precision_score(y,p)),"brier":float(brier_score_loss(y,p))}
