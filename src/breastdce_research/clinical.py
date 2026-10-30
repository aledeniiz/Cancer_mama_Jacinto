"""Standalone inference for one ORIGINAL clinical OOF fold, without training/pickle."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit

class ClinicalFoldPredictor:
    def __init__(self, specification):
        self.specification = specification
        if specification.get('format') != 'clinical_logistic_fold_json_v1':
            raise ValueError('Unsupported clinical model format')
        if specification.get('classes') != [0, 1]:
            raise ValueError('Unexpected class order')
        if specification.get('raw_feature_order') != ['age', 'HR']:
            raise ValueError('Unexpected raw feature order')
        if specification.get('transformed_feature_order') != ['age_scaled', 'HR_scaled', 'age_missing', 'HR_missing']:
            raise ValueError('Unexpected encoded feature order')
        self.fold = int(specification['fold'])
        if self.fold not in range(5):
            raise ValueError('Select exactly one OOF fold from 0 to 4')
        self.state = specification['clinical_transform']
        if self.state['features'] != ['age', 'HR']:
            raise ValueError('Unexpected transform feature order')
        self.coefficients = np.asarray(specification['coefficients'], dtype='float64')
        self.intercept = float(specification['intercept'])
        if self.coefficients.shape != (4,) or not np.isfinite(self.coefficients).all() or not np.isfinite(self.intercept):
            raise ValueError('Invalid model parameters')
        for key in ['median', 'mean', 'std']:
            values = np.asarray(self.state[key], dtype='float64')
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError('Invalid transform parameters')
        if np.any(np.asarray(self.state['std']) <= 0):
            raise ValueError('Invalid scaler standard deviation')

    @classmethod
    def load(cls, model_json):
        with Path(model_json).open(encoding='utf-8') as handle:
            return cls(json.load(handle))

    def transform(self, records):
        frame = pd.DataFrame([records] if isinstance(records, dict) else records)
        if len(frame) == 0:
            return np.empty((0, 4), dtype='float32')
        missing_columns = set(self.state['features']) - set(frame.columns)
        if missing_columns:
            raise ValueError('Missing required columns; use explicit null for allowed missing values')
        x = frame.reindex(columns=self.state['features']).apply(pd.to_numeric, errors='coerce').to_numpy(float)
        x[x == -1] = np.nan
        if np.isinf(x).any():
            raise ValueError('Finite clinical inputs required')
        for j, name in enumerate(self.state['features']):
            values = x[:, j][~np.isnan(x[:, j])]
            if name == 'HR' and not np.isin(values, [0, 1]).all():
                raise ValueError('HR must be 0, 1 or missing')
            if name == 'age' and ((values < 18) | (values > 110)).any():
                raise ValueError('Age outside supported adult range')
        missing = np.isnan(x)
        x = np.where(missing, self.state['median'], x)
        transformed = np.concatenate(((x-self.state['mean'])/self.state['std'], missing.astype(float)), 1).astype('float32')
        if not np.isfinite(transformed).all():
            raise ValueError('Unexpected non-finite transformed input')
        return transformed

    def predict_proba(self, records):
        transformed = self.transform(records)
        positive = expit(transformed @ self.coefficients + self.intercept)
        return np.column_stack((1-positive, positive))

    def predict(self, records, threshold=None):
        if threshold is None:
            threshold = self.specification['threshold_reference']
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Threshold must be between zero and one')
        return (self.predict_proba(records)[:, 1] >= threshold).astype('int64')


def fit_clinical(training_frame, features=("age", "HR")):
    """Original transform mathematics. Caller supplies TRAIN rows only."""
    if list(features)!=['age','HR'] or len(training_frame)==0:
        raise ValueError('Nonempty TRAIN rows with ordered age/HR features required')
    x=training_frame[list(features)].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    x[x==-1]=np.nan
    if np.isinf(x).any():
        raise ValueError("Finite clinical inputs required")
    med=np.array([np.nanmedian(c) if np.isfinite(c).any() else 0.0 for c in x.T])
    filled=np.where(np.isnan(x),med,x);mean=filled.mean(0);std=filled.std(0);std[std<1e-8]=1
    return {"features":list(features),"median":med.tolist(),"mean":mean.tolist(),"std":std.tolist()}


def encode_clinical(records,state):
    # Reuse the original fold predictor's exact validated transformation without
    # retaining any original trained coefficients or fitted statistics.
    specification={"format":"clinical_logistic_fold_json_v1","classes":[0,1],"fold":0,
      "raw_feature_order":["age","HR"],"transformed_feature_order":["age_scaled","HR_scaled","age_missing","HR_missing"],
      "clinical_transform":state,"coefficients":[0.0]*4,"intercept":0.0,"threshold_reference":0.5}
    return ClinicalFoldPredictor(specification).transform(records)
