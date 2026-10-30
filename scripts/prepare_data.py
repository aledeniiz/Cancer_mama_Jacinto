"""Adapt authorized PREPARED development PNG metadata; no MRI reconstruction."""
import argparse
from pathlib import Path
import pandas as pd
from breastdce_research.evaluation import validate_patient_table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,required=True,help='Authorized prepared original development CSV')
    parser.add_argument('--image-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='NEW private CSV outside this public repository')
    args=parser.parse_args()
    try:
        repository=Path(__file__).resolve().parents[1]
        if args.output.resolve().is_relative_to(repository):
            raise ValueError('Private metadata output must stay outside the public repository')
        if not args.metadata.is_file() or not args.image_root.is_dir():
            raise FileNotFoundError('Authorized prepared metadata and local image root are required')
        original=pd.read_csv(args.metadata,dtype={'patient_id':'string'})
        mapping={'pCR':'label','path_pre':'pre_path','path_early':'early_path','path_late':'late_path'}
        required={'patient_id','pCR','fold','path_pre','path_early','path_late'}
        if not required<=set(original):
            raise ValueError('Prepared metadata requires patient_id,pCR,fold,path_pre,path_early,path_late')
        if any(name in original for name in mapping.values()):
            raise ValueError('Ambiguous mixed original/canonical columns')
        if 'split' in original and not original['split'].eq('train').all():
            raise ValueError('Development adapter refuses historical TEST or other splits; use a separately verified protocol')
        canonical=original[['patient_id','pCR','fold','path_pre','path_early','path_late']].rename(columns=mapping)
        validate_patient_table(canonical,image_root=args.image_root)
        root=args.image_root.resolve()
        for column in ('pre_path','early_path','late_path'):
            if any(not (root/str(value).replace('\\','/')).is_file() for value in canonical[column]):
                raise FileNotFoundError('A phase file referenced by the prepared metadata is unavailable')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf8',newline='') as handle:canonical.to_csv(handle,index=False)
        print('Private prepared metadata adapted without changing outcomes, folds or rows. No images copied.')
    except (ValueError,FileNotFoundError,FileExistsError) as exc:parser.exit(2,f'Error: {exc}\n')


if __name__=='__main__':main()
