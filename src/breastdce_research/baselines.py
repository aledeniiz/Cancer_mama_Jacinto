"""Original clinical/linear fusion estimator settings; authorized TRAIN API only.

These helpers are source for reproduction, not invoked by the test suite or
installation. No learned parameters from the medical study are distributed.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from .clinical import fit_clinical,encode_clinical


def fit_clinical_baseline(training_frame,training_labels,fold):
    """Age/HR plus missingness; fit preprocessing on this fold's TRAIN rows."""
    labels=np.asarray(training_labels)
    if labels.shape!=(len(training_frame),) or not np.isin(labels,[0,1]).all() or len(np.unique(labels))!=2:
        raise ValueError('Aligned binary training outcomes with both classes required')
    if type(fold) is not int or fold not in range(5):raise ValueError('Explicit original fold 0..4 required')
    transform=fit_clinical(training_frame)
    encoded=encode_clinical(training_frame,transform)
    estimator=LogisticRegression(C=1,solver='liblinear',max_iter=2000,random_state=42).fit(encoded,labels)
    return {'format':'clinical_logistic_fold_json_v1','fold':fold,'classes':[0,1],
        'raw_feature_order':['age','HR'],'transformed_feature_order':['age_scaled','HR_scaled','age_missing','HR_missing'],
        'clinical_transform':transform,'coefficients':estimator.coef_[0].tolist(),
        'intercept':float(estimator.intercept_[0]),'threshold_reference':.5,
        'scope':'NEW local training output; not the private recovered original OOF model'}


def fit_linear_fusion(training_embeddings,encoded_training_clinical,training_labels):
    """Fit only after train-fitted 64-D embedding and 4-D clinical transforms.

    This is a distinct exploratory logistic fusion baseline, not the 1,225-
    parameter neural fusion head and not the all-development multimodal model.
    """
    image=np.asarray(training_embeddings);clinical=np.asarray(encoded_training_clinical)
    if image.ndim!=2 or len(image)==0 or image.shape[1]!=64 or clinical.shape!=(len(image),4):
        raise ValueError('Aligned training image64 and clinical4 matrices required')
    labels=np.asarray(training_labels)
    if labels.shape!=(len(image),) or not np.isin(labels,[0,1]).all() or len(np.unique(labels))!=2:
        raise ValueError('Aligned binary pCR outcomes with both classes required')
    features=np.concatenate((image,clinical),1)
    if not np.isfinite(features).all():raise ValueError('Finite normalized TRAIN features required')
    return LogisticRegression(C=1,solver='liblinear',max_iter=2000,random_state=42).fit(features,labels)
