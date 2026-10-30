"""Evaluate one local checkpoint on an explicitly selected patient fold."""
import argparse
from pathlib import Path
import pandas as pd
from breastdce_research.inference import load_cnn,write_new_json
from breastdce_research.evaluation import patient_split,verify_partition_lineage
from breastdce_research.data import read_bags
from breastdce_research.training import evaluate_bags


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--image-root',type=Path,required=True);p.add_argument('--fold',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:
        model,norm=load_cnn(args.checkpoint)
        frame=pd.read_csv(args.metadata,dtype={'patient_id':'string'})
        lineage=verify_partition_lineage(model.partition_lineage,frame,args.fold)
        _,validation=patient_split(frame,args.fold,image_root=args.image_root)
        metrics=evaluate_bags(model,read_bags(validation,args.image_root),norm)
        write_new_json(args.output,{'protocol':'Partition matches recorded checkpoint training lineage; NOT automatically a full five-fold OOF evaluation or an independent audit of training','fold':args.fold,'partition_lineage':lineage,'metrics':metrics})
    except (ValueError,FileNotFoundError,FileExistsError) as exc:p.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
