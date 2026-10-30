"""Train only with authorized local data; never runs implicitly."""
import argparse
from pathlib import Path
import pandas as pd
from breastdce_research.training import train_cnn
from breastdce_research.inference import write_new_json,save_checkpoint_new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--image-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--confirm-authorized-data',action='store_true',required=True)
    args=parser.parse_args()
    try:
        if not args.metadata.is_file():raise FileNotFoundError('Authorized local metadata CSV required')
        args.output.mkdir(parents=True,exist_ok=False)
        frame=pd.read_csv(args.metadata,dtype={'patient_id':'string'})
        history=[]
        for checkpoint in train_cnn(frame,args.image_root,args.fold,args.epochs):
            path=args.output/f"epoch_{checkpoint['epoch']:03d}"
            save_checkpoint_new(path,checkpoint)
            history.append({'epoch':checkpoint['epoch'],**checkpoint['metrics']})
        write_new_json(args.output/'history.json',history)
    except (ValueError,FileNotFoundError,FileExistsError) as exc:parser.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
