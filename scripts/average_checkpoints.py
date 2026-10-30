"""Post-hoc epoch EMA plus TRAIN-only label-free BN recalibration."""
import argparse
from pathlib import Path
import pandas as pd
import torch
from breastdce_research.inference import load_checkpoint,save_checkpoint_new
from breastdce_research.models import CompactPatientCNN
from breastdce_research.averaging import ema_parameters,recalibrate_batchnorm
from breastdce_research.evaluation import patient_split,verify_epoch_lineage
from breastdce_research.imaging import preprocess,standardize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',nargs='+',type=Path,required=True,help='Chronological raw epoch checkpoints; never future epochs')
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--image-root',type=Path,required=True)
    p.add_argument('--fold',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:
        frame=pd.read_csv(args.metadata,dtype={'patient_id':'string'})
        train,heldout=patient_split(frame,args.fold,image_root=args.image_root)
        checkpoints=[load_checkpoint(path) for path in args.checkpoints]
        lineage=verify_epoch_lineage(checkpoints,frame,args.fold)
        epochs=[cp['epoch'] for cp in checkpoints]
        normalization=checkpoints[0]['normalization']
        model=CompactPatientCNN();model.load_state_dict(ema_parameters(model,[cp['model_state_dict'] for cp in checkpoints]))
        root=args.image_root.resolve();order=torch.randperm(len(train),generator=torch.Generator().manual_seed(42)).tolist()
        def batches():
            for start in range(0,len(order),64):
                images=[]
                for row in train.iloc[order[start:start+64]].itertuples():
                    paths={phase:(root/str(getattr(row,col)).replace('\\','/')).resolve() for phase,col in [('PRE','pre_path'),('EARLY','early_path'),('LATE','late_path')]}
                    if any(not path.is_relative_to(root) for path in paths.values()):raise ValueError('Image path outside authorized root')
                    images.append(standardize(preprocess(paths),normalization))
                yield torch.stack(images)
        bn=recalibrate_batchnorm(model,batches(),set(train.patient_id),set(heldout.patient_id))
        save_checkpoint_new(args.output,{'model_state_dict':model.state_dict(),'normalization':normalization,
            'config':{'search_variant':'compact','dropout':.3},'partition_lineage':lineage,'source_epochs':epochs,'ema_decay':.8,'bn':bn})
    except (ValueError,KeyError,FileNotFoundError,FileExistsError) as exc:p.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
