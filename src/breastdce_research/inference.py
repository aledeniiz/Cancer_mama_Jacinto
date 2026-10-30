"""Explicit local JSON/NPZ resources only; never deserialize pickle checkpoints."""
from pathlib import Path
import json
import re
import zipfile
import math
import numpy as np
import torch
from .models import CompactPatientCNN


def load_checkpoint(path):
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError("Checkpoint unavailable. Select an existing bundle from models/trained or supply an authorized local checkpoint.")
    if not path.is_dir():
        raise ValueError("Public loaders accept a JSON/NPZ bundle directory only; pickle-based .pt/.pth files require trusted private conversion outside this public loader.")
    try:
        meta=path/'metadata.json';arrays=path/'weights.npz'
        if meta.stat().st_size>1024*1024 or arrays.stat().st_size>16*1024*1024:
            raise ValueError("Excessive bundle size")
        payload=json.loads(meta.read_text(encoding='utf8'))
        if payload.pop('format',None)!='breastdce-json-npz-v1':
            raise ValueError("Unsupported bundle format")
        with zipfile.ZipFile(arrays) as archive:
            info=archive.infolist()
            if len(info)>1024 or len({i.filename for i in info})!=len(info) or sum(i.file_size for i in info)>16*1024*1024:
                raise ValueError("Excessive tensor archive")
            for member in info:
                if not member.filename.endswith('.npy') or '/' in member.filename or '\\' in member.filename:
                    raise ValueError('Unexpected archive member')
                # Validate declared shape BEFORE np.load can allocate storage.
                with archive.open(member) as stream:
                    version=np.lib.format.read_magic(stream)
                    reader={(1,0):np.lib.format.read_array_header_1_0,
                            (2,0):np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:raise ValueError('Unsupported NPY header version')
                    shape,_,dtype=reader(stream,max_header_size=16384)
                    count=math.prod(shape)
                    if dtype.hasobject or len(shape)>8 or any(n<0 for n in shape) or count>1_000_000:
                        raise ValueError('Invalid declared tensor shape/dtype')
                    if member.file_size-stream.tell()!=count*dtype.itemsize:
                        raise ValueError('Tensor payload length does not match declared shape')
        groups={}
        with np.load(arrays,allow_pickle=False) as data:
            for key in data.files:
                if not re.fullmatch(r'(model_state_dict|head_state)::[A-Za-z0-9_.]+',key):
                    raise ValueError("Unexpected tensor group/key")
                group,name=key.split('::',1);value=data[key]
                if value.dtype.name not in ('float16','float32','float64','int8','int16','int32','int64','uint8','bool') or value.size>1_000_000:
                    raise ValueError("Unsupported tensor dtype/shape")
                if not np.isfinite(value).all():
                    raise ValueError("Nonfinite tensor state")
                groups.setdefault(group,{})[name]=torch.from_numpy(value.copy())
        if not groups or any(group in payload for group in groups):
            raise ValueError("Missing or ambiguous tensor state")
        payload.update(groups)
    except Exception as exc:
        raise ValueError("Invalid non-pickle JSON/NPZ checkpoint bundle") from exc
    if not isinstance(payload,dict):
        raise ValueError("Expected a plain checkpoint dictionary")
    return payload


def save_checkpoint_new(path,payload):
    """Create a NEW local bundle, preserving tensor values without pickle."""
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    metadata={};arrays={}
    for key,value in payload.items():
        if key in ('model_state_dict','head_state'):
            arrays.update({f'{key}::{name}':tensor.detach().cpu().numpy() for name,tensor in value.items()})
        else:metadata[key]=value
    metadata['format']='breastdce-json-npz-v1'
    write_new_json(path/'metadata.json',metadata)
    with (path/'weights.npz').open('xb') as handle:np.savez_compressed(handle,**arrays)


def load_cnn(path):
    payload=load_checkpoint(path)
    if 'model_state_dict' not in payload or 'normalization' not in payload:
        raise ValueError("CNN checkpoint requires model_state_dict and training normalization")
    config=payload.get('config',{})
    if config.get('search_variant','compact')!='compact':
        raise ValueError("Only original compact architecture is supported")
    model=CompactPatientCNN(dropout=config.get('dropout',.3))
    try:model.load_state_dict(payload['model_state_dict'],strict=True)
    except RuntimeError as exc:raise ValueError('Checkpoint does not match the exact compact CNN') from exc
    # Inference may use a legacy converted model, but evaluation separately
    # rejects missing/mismatched lineage before making any held-out assertion.
    model.partition_lineage=payload.get('partition_lineage')
    return model.eval(),payload['normalization']


def write_new_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
