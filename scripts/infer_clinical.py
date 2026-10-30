"""Select exactly one private OOF fold model; no global model is invented."""
import argparse,json
from pathlib import Path
from breastdce_research.clinical import ClinicalFoldPredictor
from breastdce_research.inference import write_new_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:
        if not args.model.is_file():raise FileNotFoundError('Private authorized fold model JSON required; no clinical parameters are distributed')
        model=ClinicalFoldPredictor.load(args.model)
        records=json.loads(args.input.read_text(encoding='utf8'))
        write_new_json(args.output,{'fold':model.fold,'scope':'single OOF fold predictor, not final global model',
            'probabilities':model.predict_proba(records)[:,1].tolist()})
    except (ValueError,FileNotFoundError,FileExistsError) as exc:p.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
