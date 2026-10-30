"""Local fusion inference requires authorized CNN/head weights and image64 input."""
import argparse,json
from pathlib import Path
from breastdce_research.inference import load_checkpoint,write_new_json
from breastdce_research.multimodal import predict_embedding


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:
        bundle=load_checkpoint(args.head)
        if not {'head_state','clinical_transform','embedding_mean','embedding_std'}<=set(bundle):
            raise ValueError('Fusion head and original fitted transforms required')
        record=json.loads(args.input.read_text(encoding='utf8'))
        probability=predict_embedding(record['embedding64'],record['clinical'],bundle)
        write_new_json(args.output,{'probability':probability,'scope':'User-supplied head. Final head has no independent validation claim.'})
    except (ValueError,KeyError,FileNotFoundError,FileExistsError) as exc:p.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
