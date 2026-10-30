"""Local multimodal bridge. The HTTP handler supplies its existing lock and PNG decoder.

Public endpoints contain aggregate study evidence and feature definitions only.
Clinical inputs and image tensors are never persisted by this module.
"""
import hashlib
import importlib.util
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts/multimodal_20261029'
REPORTS = ROOT / 'reports/multimodal_20261029'
ORIGINAL_SHA = 'd9fcf647298a88b3106eb5861a866dbab937f22d191d5e5e3804e97fe23677d5'
_CACHE = None
_CACHE_STAMP = None
_CACHE_LOCK = threading.Lock()


def _json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _schema():
    path = ARTIFACTS / 'schema.json'
    if not path.exists():
        return None
    raw = _json(path)
    # Allow only explicitly selected fields to reach the form.
    selected = set(raw.get('primary_features', []))
    return {key: raw.get(key) for key in ('version', 'supported_cohorts', 'timing_caveat', 'missing_rule')} | {
        'features': [field for field in raw.get('features', []) if field.get('name') in selected]
    }


def _load():
    global _CACHE, _CACHE_STAMP
    path = ARTIFACTS / 'final.pt'
    if not path.exists():
        return None
    stat = path.stat()
    stamp = (stat.st_mtime_ns, stat.st_size)
    with _CACHE_LOCK:
        if stamp == _CACHE_STAMP:
            return _CACHE
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        image_sha = bundle.get('image_checkpoint_sha256', bundle.get('checkpoint_sha256'))
        if image_sha != ORIGINAL_SHA:
            raise ValueError('El modelo de imagen del paquete multimodal no coincide con la CNN original.')
        schema = _schema()
        fields = [f['name'] for f in (schema or {}).get('features', [])]
        if not fields or bundle['clinical_transform']['features'] != fields:
            raise ValueError('El esquema clínico no coincide con el transformador guardado.')
        if len(bundle['embedding_mean']) != 64 or len(bundle['embedding_std']) != 64:
            raise ValueError('Dimensión visual incompatible.')
        if not np.isfinite(np.asarray(bundle['embedding_std'], float)).all() or (np.asarray(bundle['embedding_std']) <= 0).any():
            raise ValueError('Escalado visual no válido.')
        source = ROOT / 'src/breastdcedl/multimodal.py'
        spec = importlib.util.spec_from_file_location('dce_local_multimodal', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Strict model-state check on load, before accepting clinical data.
        net = module.FusionNet(len(fields) * 2)
        net.load_state_dict(bundle['head_state'], strict=True)
        net.eval()
        _CACHE = {'bundle': bundle, 'module': module, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        _CACHE_STAMP = stamp
        return _CACHE


def status():
    """Safe to call lazily while a separate study process is producing artifacts."""
    schema = _schema()
    export = REPORTS / 'web_results.json'
    evidence = _json(export) if export.exists() else {}
    public_keys = ('models', 'paired_summary', 'conclusion', 'study_status')
    out = {key: evidence[key] for key in public_keys if key in evidence}
    out['schema'] = schema
    try:
        loaded = _load()
        out['ready'] = loaded is not None
        out['message'] = 'Modelo multimodal disponible.' if loaded else 'El estudio está en preparación. Todavía no hay un modelo multimodal ejecutable.'
        if loaded:
            out['model_sha256'] = loaded['sha256']
    except (ValueError, KeyError, RuntimeError, OSError):
        out['ready'] = False
        out['message'] = 'El paquete multimodal está incompleto o no supera la comprobación de compatibilidad. No se ejecuta inferencia.'
    return out


def validate_clinical(record, schema=None):
    schema = schema or _schema()
    if not schema or not isinstance(record, dict):
        raise ValueError('Se requieren datos clínicos compatibles con el esquema.')
    fields = schema['features']
    names = {field['name'] for field in fields}
    if set(record) - names:
        raise ValueError('El registro contiene variables no admitidas.')
    cleaned = {}
    for field in fields:
        name = field['name']
        value = record.get(name)
        if value is None:
            if not field.get('missing_allowed', False):
                raise ValueError('Falta una variable clínica obligatoria.')
            cleaned[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Las variables clínicas deben ser numéricas finitas o ausentes.')
        if 'min' in field and value < field['min'] or 'max' in field and value > field['max']:
            raise ValueError('Variable clínica fuera de intervalo.')
        if field['type'] == 'category' and value not in field.get('allowed', [c['value'] for c in field.get('categories', [])]):
            raise ValueError('Categoría clínica no admitida.')
        cleaned[name] = float(value)
    return cleaned


def predict(data, image_model, normalization, decode_request, standardize):
    """Caller must retain the existing inference lock, token/origin and body limits."""
    started = time.perf_counter()
    if not isinstance(data, dict) or set(data) - {'slices', 'clinical', 'threshold'}:
        raise ValueError('Solicitud multimodal no válida.')
    loaded = _load()
    if loaded is None:
        raise FileNotFoundError('Modelo multimodal aún no disponible.')
    record = validate_clinical(data.get('clinical'))
    image_data = {'slices': data.get('slices'), 'threshold': data.get('threshold', .5)}
    x, threshold = decode_request(image_data)
    x = standardize(x, normalization)
    with torch.inference_mode():
        # Chunking does not alter patient aggregation. BN and dropout remain in eval.
        embeddings = [image_model._encode(chunk) for chunk in x.split(16)]
        embedding = torch.cat(embeddings, dim=0).mean(dim=0).cpu().numpy()
        if embedding.shape != (64,) or not np.isfinite(embedding).all():
            raise ValueError('Representación visual no válida.')
        original_score = float(image_model.forward_patient(x, chunk_size=16).sigmoid().item())
    score = loaded['module'].predict_embedding(embedding, record, loaded['bundle'])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Puntuación multimodal no válida.')
    return {
        'score': score, 'original_score': original_score, 'threshold': threshold,
        'classification': 'pCR' if score >= threshold else 'no-pCR', 'n_slices': len(x),
        'model': 'CNN propia congelada + rama clínica y fusión',
        'checkpoint_sha256': loaded['sha256'], 'image_checkpoint_sha256': ORIGINAL_SHA,
        'clinical_features': list(record), 'missing_features': [k for k, v in record.items() if v is None],
        'calibration': 'No demostrada para decisiones clínicas',
        'privacy': 'Entradas procesadas únicamente en memoria local',
        'timings_ms': {'total': (time.perf_counter() - started) * 1000},
    }
