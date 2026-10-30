"""Score one authorized patient bag without requiring its treatment outcome."""
import argparse
from pathlib import Path
import torch
from breastdce_research.inference import load_cnn,write_new_json
from breastdce_research.imaging import preprocess,standardize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    for phase in ('pre','early','late'):p.add_argument('--'+phase,nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:
        model,normalization=load_cnn(args.checkpoint)
        if not len(args.pre)==len(args.early)==len(args.late) or not 1<=len(args.pre)<=512:
            raise ValueError('Provide matching ordered phase lists for 1..512 cuts')
        slices=torch.stack([preprocess({'PRE':pre,'EARLY':early,'LATE':late})
            for pre,early,late in zip(args.pre,args.early,args.late)])
        with torch.inference_mode():probability=float(model.forward_patient(standardize(slices,normalization)).sigmoid().item())
        write_new_json(args.output,{'probability':probability,'n_slices':len(slices),'scope':'Research only; user-supplied authorized checkpoint'})
    except (ValueError,FileNotFoundError,FileExistsError) as exc:p.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
