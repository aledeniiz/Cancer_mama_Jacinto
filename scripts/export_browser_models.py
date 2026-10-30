"""Export preserved CNN EMA and final fusion for browser inference, without fitting.

Requires optional export tools: onnx==1.23.2, onnxruntime==1.31.1.
Every output file must be new. MRI fixtures, when supplied, remain local.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from torch import nn

from breastdce_research.inference import load_cnn, load_checkpoint
from breastdce_research.clinical import encode_clinical
from breastdce_research.imaging import preprocess, standardize
from breastdce_research.multimodal import FusionNet

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ['model-cnn-ema.onnx', 'model-classifier.onnx', 'model-fusion.onnx',
               'model-config.json', 'model-verification.json', 'model-README.md', 'model-manifest.json']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


class Encoder(nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.features = cnn.backbone.features
        self.pool = cnn.backbone.pool

    def forward(self, images):
        return self.pool(self.features(images)).flatten(1)


class Head(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.clinical = head.clinical
        self.fusion = head.fusion

    def forward(self, image, clinical):
        return self.fusion(torch.cat((image, self.clinical(clinical)), 1)).flatten()


def export(output, image_root=None, evidence=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in MODEL_FILES):
        raise FileExistsError('Export refuses to overwrite any existing model output')
    torch.set_num_threads(4)
    cnn, normalization = load_cnn(ROOT/'models/trained/cnn_ema')
    bundle = load_checkpoint(ROOT/'models/trained/multimodal_final')
    original_head = FusionNet()
    original_head.load_state_dict(bundle['head_state'], strict=True)
    original_head.eval()
    encoder, classifier, head = Encoder(cnn).eval(), cnn.backbone.classifier.eval(), Head(original_head).eval()
    options = dict(opset_version=17, dynamo=False, do_constant_folding=False)
    with torch.inference_mode():
        torch.onnx.export(encoder, torch.zeros(1,3,256,256), output/'model-cnn-ema.onnx',
            input_names=['images'], output_names=['embedding'],
            dynamic_axes={'images':{0:'slices'}, 'embedding':{0:'slices'}}, **options)
        torch.onnx.export(classifier, torch.zeros(1,64), output/'model-classifier.onnx',
            input_names=['embedding'], output_names=['logit'], **options)
        torch.onnx.export(head, (torch.zeros(1,64),torch.zeros(1,4)), output/'model-fusion.onnx',
            input_names=['image','clinical'], output_names=['logit'], **options)
    for name in MODEL_FILES[:3]:
        onnx.checker.check_model(str(output/name))
    cnn_meta = json.loads((ROOT/'models/trained/cnn_ema/metadata.json').read_text(encoding='utf8'))
    config = {
        'format':'breastdce-browser-onnx-v1', 'opset':17,
        'normalization':normalization,
        'cnn_classifier':{'weight':classifier.weight.detach().numpy().reshape(-1).tolist(),
                          'bias':float(classifier.bias.detach()[0])},
        'clinical_transform':bundle['clinical_transform'],
        'embedding_mean':bundle['embedding_mean'], 'embedding_std':bundle['embedding_std'],
        'models':{
            'encoder':{'file':'model-cnn-ema.onnx','input':'images','output':'embedding'},
            'classifier':{'file':'model-classifier.onnx','input':'embedding','output':'logit'},
            'fusion':{'file':'model-fusion.onnx','inputs':['image','clinical'],'output':'logit'}},
        'source':{
            'cnn_checkpoint_sha256':cnn_meta['source_checkpoint_sha256'],
            'fusion_image_checkpoint_sha256':bundle['image_checkpoint_sha256'],
            'cnn_export_weights_sha256':sha(ROOT/'models/trained/cnn_ema/weights.npz'),
            'fusion_export_weights_sha256':sha(ROOT/'models/trained/multimodal_final/weights.npz')},
        'input_contract':{'phases':['PRE','EARLY','LATE'],'width':256,'height':256,
                          'png_color_type':0,'bit_depth':8,'min_slices':1,'max_slices':64,
                          'normalization_arithmetic':'float32 divide by 255, subtract float32 mean, divide float32 std',
                          'clinical_order':['age_scaled','HR_scaled','age_missing','HR_missing']},
        'scope':{'cnn':'Historical EMA 0.8 epoch 15; historical TEST AUC 0.5524, n=176',
                 'fusion':'Final development-fitted CNN EMA + clinical head; no independently measured AUC',
                 'new_training':False, 'quantized':False}}
    assert config['source']['cnn_checkpoint_sha256']==config['source']['fusion_image_checkpoint_sha256']
    write_json(output/'model-config.json',config)
    session_options=ort.SessionOptions();session_options.intra_op_num_threads=4
    sessions={key:ort.InferenceSession(str(output/value['file']),session_options,providers=['CPUExecutionProvider'])
              for key,value in config['models'].items()}
    checks=[]

    def compare(name, raw, clinical):
        normalized=standardize(raw,normalization)
        with torch.inference_mode():
            pt_embedding=cnn._encode(normalized)
            pt_patient=pt_embedding.mean(0,keepdim=True)
            pt_logit=cnn._classify(pt_patient)
        onnx_embedding=sessions['encoder'].run(None,{'images':normalized.numpy()})[0]
        # Browser extracts one slice at a time and then computes the patient mean.
        sequential=np.concatenate([sessions['encoder'].run(None,{'images':normalized.numpy()[i:i+1]})[0]
                                   for i in range(len(raw))])
        patient=np.mean(sequential,axis=0,keepdims=True,dtype=np.float32)
        onnx_logit=sessions['classifier'].run(None,{'embedding':patient})[0]
        pt_p=float(pt_logit.sigmoid().item());onnx_p=float(torch.tensor(onnx_logit).sigmoid().item())
        pt_normalized=((pt_patient.numpy()-np.asarray(bundle['embedding_mean']))/np.asarray(bundle['embedding_std'])).astype('float32')
        ort_normalized=((patient-np.asarray(bundle['embedding_mean']))/np.asarray(bundle['embedding_std'])).astype('float32')
        encoded=encode_clinical(clinical,bundle['clinical_transform'])
        with torch.inference_mode():pt_f=float(original_head(torch.from_numpy(pt_normalized),torch.from_numpy(encoded)).sigmoid().item())
        fusion_logit=sessions['fusion'].run(None,{'image':ort_normalized,'clinical':encoded})[0]
        ort_f=float(torch.tensor(fusion_logit).sigmoid().item())
        row={'fixture':name,'slices':len(raw),'embedding_max_absolute_error':float(np.max(np.abs(pt_embedding.numpy()-onnx_embedding))),
             'sequential_embedding_max_absolute_error':float(np.max(np.abs(pt_embedding.numpy()-sequential))),
             'cnn_probability_absolute_error':abs(pt_p-onnx_p),'fusion_probability_absolute_error':abs(pt_f-ort_f),
             'reference_cnn_probability':pt_p,'onnx_cnn_probability':onnx_p,
             'reference_fusion_probability':pt_f,'onnx_fusion_probability':ort_f}
        assert max(row['cnn_probability_absolute_error'],row['fusion_probability_absolute_error'])<=1e-5,row
        assert row['embedding_max_absolute_error']<=1e-5,row
        checks.append(row)
        return row

    torch.manual_seed(42)
    compare('synthetic_three_cuts',torch.rand(3,3,256,256),{'age':48,'HR':1})
    compare('synthetic_missing_clinical',torch.rand(2,3,256,256),{'age':None,'HR':None})
    real_performed=False
    if image_root:
        table=pd.read_csv(ROOT/'evaluation/metadata/samples.csv')
        first=table[table.split=='test'].iloc[0].patient_id
        rows=table[table.patient_id==first].sort_values('sample_id')
        record=pd.read_csv(ROOT/'evaluation/clinical/clinical_inputs.csv').set_index('patient_id').loc[first]
        paths=[{phase:Path(image_root)/getattr(row,column) for phase,column in
                [('PRE','path_pre'),('EARLY','path_early'),('LATE','path_late')]} for row in rows.itertuples(index=False)]
        raw=torch.stack([preprocess(item) for item in paths])
        clinical={feature:None if pd.isna(record[feature]) or record[feature]==-1 else float(record[feature]) for feature in ['age','HR']}
        result=compare('public_case_real_mri_patient',raw,clinical)
        real_performed=True
        if evidence:
            evidence=Path(evidence);evidence.mkdir(parents=True,exist_ok=True)
            write_json(evidence/'browser_fixture.json',{'phases':{phase:[str(item[phase].resolve()) for item in paths]
                       for phase in ['PRE','EARLY','LATE']},'clinical':clinical,'expected':result})
            # Fixture stays outside the release. Hashes permit repeat checks without publishing MRI files.
            write_json(evidence/'real_fixture_hashes.json',{'files':[{'phase':phase,'sha256':sha(path)} for item in paths for phase,path in item.items()]})
            fixtures=evidence.parent/'fixtures'
            anonymous={phase:[] for phase in ['PRE','EARLY','LATE']}
            for phase in anonymous:
                directory=fixtures/phase;directory.mkdir(parents=True,exist_ok=True)
                for index,item in enumerate(paths):
                    destination=directory/f'slice_{index+1:03d}.png'
                    with destination.open('xb') as handle:handle.write(item[phase].read_bytes())
                    anonymous[phase].append(str(destination.resolve()))
            references=[]
            for count in sorted({1,len(raw)}):
                for label,values in [('age48_hr1',{'age':48,'HR':1}),('missing',{'age':None,'HR':None})]:
                    reference=compare(f'real_{count}_slices_{label}',raw[:count],values)
                    references.append({'slices':count,'clinical':values,'expected':reference})
            write_json(evidence.parent/'expected.json',{'phases':anonymous,'cases':references,
                       'scope':'Local public-case browser QA only; no files here may be committed to GitHub.'})
    report={'status':'passed','new_training':False,'quantization':False,'real_mri_patient_checked':real_performed,
            'probability_tolerance':1e-5,'embedding_tolerance':1e-5,
            'fixtures':[{key:value for key,value in check.items() if not key.startswith(('reference_','onnx_'))} for check in checks],
            'runtime':{'python':platform.python_version(),'torch':torch.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__},
            'limits':'Conversion and inference parity only; no new scientific validation. Browser WASM/PNG decoder must also be checked independently.'}
    write_json(output/'model-verification.json',report)
    with (output/'model-README.md').open('x',encoding='utf8',newline='\n') as handle:
        handle.write('''# Modelos ONNX para inferencia local en el navegador

Conversión de la CNN EMA y cabeza multimodal final conservadas en `models/trained/`.
Sin reentrenamiento, cuantización, sustitución de modelos ni cambios de resultados.
La CNN recibe PRE/EARLY/LATE estandarizados y produce 64 rasgos por corte. Se
promedian todos los cortes del paciente y se aplica el clasificador original.
La fusión recibe ese promedio normalizado y las cuatro variables clínicas
codificadas con el transformador original; usa su propia cabeza conservada.

`model-config.json` incluye contratos, transformadores y hashes de origen.
`model-verification.json` compara PyTorch y ONNX Runtime CPU, incluyendo un paciente
del caso público cuando se proporcionan MRI locales. Ninguna MRI se distribuye.
Las tolerancias no acreditan una validación externa ni utilidad clínica. La cabeza
multimodal final no hereda la AUC OOF de las cabezas por fold.

Exportación opcional: `python scripts/export_browser_models.py --output NUEVO_DIR`.
Para repetir la prueba desde imágenes añadir `--image-root RAIZ_PNG` y un
`--evidence` externo nuevo. Los archivos de salida existentes nunca se sobrescriben.
Dependencias de exportación: onnx 1.23.2, onnxruntime 1.31.1 y el runtime del proyecto.
''')
    files=[]
    for name in MODEL_FILES[:-1]:
        path=output/name;files.append({'path':name,'size':path.stat().st_size,'sha256':sha(path)})
    write_json(output/'model-manifest.json',{'format':1,'files':files,'note':'Excludes itself; MRI fixtures and local paths are never published.'})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--image-root',type=Path)
    parser.add_argument('--evidence',type=Path)
    args=parser.parse_args()
    try:
        result=export(args.output,args.image_root,args.evidence)
        print(json.dumps(result,indent=2))
    except (ValueError,FileExistsError,FileNotFoundError) as error:
        parser.exit(2,f'Error: {error}\n')
