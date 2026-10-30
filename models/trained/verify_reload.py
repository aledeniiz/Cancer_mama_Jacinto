"""Cold-load every released model in an independent process; never train."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'src'))


def tensor_fingerprint(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        values=tensor.detach().cpu().numpy()
        h.update(name.encode());h.update(str(values.dtype).encode())
        h.update(json.dumps(list(values.shape)).encode());h.update(values.tobytes())
    return h.hexdigest()


def clinical_fingerprint(model):
    values={key:model[key] for key in ['coefficients','intercept','clinical_transform']}
    return hashlib.sha256(json.dumps(values,sort_keys=True,allow_nan=False).encode()).hexdigest()


def worker(identifier):
    import numpy as np
    import torch
    import breastdce_research
    from breastdce_research.inference import load_cnn,load_checkpoint
    from breastdce_research.imaging import standardize
    from breastdce_research.clinical import ClinicalFoldPredictor
    from breastdce_research.multimodal import FusionNet,predict_embedding
    assert Path(breastdce_research.__file__).resolve().is_relative_to(REPO)
    torch.set_num_threads(2)
    registry=json.loads((ROOT/'registry.json').read_text(encoding='utf8'))
    record=next((x for x in registry['models'] if x['id']==identifier),None)
    if record is None:raise ValueError('Unknown released model')
    directory=(ROOT/identifier).resolve()
    assert directory.is_relative_to(ROOT)
    if record['kind']=='cnn':
        model,norm=load_cnn(directory)
        assert model.n_parameters==24713
        assert tensor_fingerprint(model.state_dict())==record['tensor_fingerprint']
        torch.manual_seed(2718)
        with torch.inference_mode():
            score=float(model.forward_patient(standardize(torch.rand(1,3,256,256),norm)).sigmoid().item())
        tensors=len(model.state_dict())
    elif record['kind']=='neural_head':
        bundle=load_checkpoint(directory)
        model=FusionNet();model.load_state_dict(bundle['head_state'],strict=True)
        assert sum(p.numel() for p in model.parameters())==1225
        assert tensor_fingerprint(bundle['head_state'])==record['tensor_fingerprint']
        assert bundle['image_checkpoint_sha256']==record['image_checkpoint_sha256']
        # A fitted mean is used as a software fixture, not a patient embedding.
        score=predict_embedding(np.asarray(bundle['embedding_mean'],dtype='float32'),
            {'age':48,'HR':1},bundle)
        tensors=len(bundle['head_state'])
    else:
        model_path=directory/'model.json';specification=json.loads(model_path.read_text(encoding='utf8'))
        assert clinical_fingerprint(specification)==record['parameter_fingerprint']
        model=ClinicalFoldPredictor.load(model_path)
        assert model.fold==record['fold']
        score=float(model.predict_proba({'age':48,'HR':1})[0,1]);tensors=0
    assert np.isfinite(score) and 0<=score<=1
    return {'id':identifier,'kind':record['kind'],'fold':record['fold'],
        'status':'passed','state_tensors':tensors,'finite_software_smoke':True,'pid':os.getpid()}


def run():
    manifest=json.loads((ROOT/'manifest_sha256.json').read_text(encoding='utf8'))
    for row in manifest['files']:
        path=(ROOT/row['path']).resolve()
        assert path.is_relative_to(ROOT) and path.is_file()
        assert path.stat().st_size==row['size']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    registry=json.loads((ROOT/'registry.json').read_text(encoding='utf8'))
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    def cold(record):
        response=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',record['id']],
            cwd=REPO,env=env,capture_output=True,text=True,encoding='utf8',check=True)
        return json.loads(response.stdout)
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(cold,registry['models']))
    pids=[x.pop('pid') for x in results]
    assert len(results)==17 and len(set(pids))==17 and os.getpid() not in pids
    return {'status':'passed','cold_independent_processes':len(results),
        'all_processes_distinct':True,'manifest_files_verified':len(manifest['files']),
        'state_tensors_verified':sum(x['state_tensors'] for x in results),
        'clinical_fold_predictors_verified':sum(x['kind']=='clinical_logistic_fold' for x in results),
        'software_fixtures_only':True,'new_training':False,'results':results}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker')
    args=parser.parse_args();print(json.dumps(worker(args.worker) if args.worker else run()))
