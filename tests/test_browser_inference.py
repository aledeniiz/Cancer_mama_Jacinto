"""Compare browser preprocessing with original Python on artificial PNG fixtures."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tempfile
import numpy as np
from PIL import Image
import pytest
import torch
from breastdce_research.imaging import load_phase,standardize
from breastdce_research.clinical import encode_clinical

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which('node') is None,reason='Node required for JavaScript preprocessing parity')
def test_browser_pixels_normalization_clinical_and_invalid_pngs():
    config=json.loads((ROOT/'web/inference-assets/model-config.json').read_text(encoding='utf8'))
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp);pixels=np.arange(65536,dtype=np.uint16).reshape(256,256).astype(np.uint8)
        valid=out/'gray.png';Image.fromarray(pixels).save(valid)
        rgb=out/'rgb.png';Image.new('RGB',(256,256)).save(rgb)
        wrong=out/'wrong.png';Image.new('L',(128,256)).save(wrong)
        corrupt=out/'crc.png';b=bytearray(valid.read_bytes());b[-5]^=1;corrupt.write_bytes(b)
        tiny=out/'tiny.png';tiny.write_bytes(b'not a PNG')
        response=subprocess.run(['node',str(ROOT/'tests/browser_png.cjs'),str(ROOT/'web/browser-inference.js'),
            str(ROOT/'web/inference-assets/model-config.json'),*[str(p) for p in [valid,rgb,wrong,corrupt,tiny]]],
            capture_output=True,text=True,encoding='utf8',check=True)
        report=json.loads(response.stdout)
        assert report['pixel_sha']==hashlib.sha256(pixels.tobytes()).hexdigest()
        tensor=standardize(torch.stack([load_phase(valid)]*3),config['normalization'])[0]
        assert report['normalized_sha']==hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        assert np.array_equal(np.asarray(report['clinical'],dtype=np.float32),encode_clinical({'age':48,'HR':1},config['clinical_transform'])[0])
        assert np.array_equal(np.asarray(report['missing'],dtype=np.float32),encode_clinical({'age':None,'HR':None},config['clinical_transform'])[0])
        assert report['rejected']==[True]*4
