"""Extract private local patient embeddings; nothing is uploaded or published."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from breastdce_research.inference import load_cnn
from breastdce_research.data import read_bags
from breastdce_research.imaging import standardize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','metadata','image-root','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    try:
        model,norm=load_cnn(args.checkpoint);identifiers=[];embeddings=[]
        with torch.inference_mode():
            for identifier,images,_ in read_bags(pd.read_csv(args.metadata,dtype={'patient_id':'string'}),args.image_root):
                x=standardize(images,norm)
                embedding=torch.cat([model._encode(chunk) for chunk in x.split(16)]).float().mean(0)
                identifiers.append(identifier);embeddings.append(embedding.numpy())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('xb') as handle:np.savez(handle,ids=np.array(identifiers),embedding=np.stack(embeddings))
        print('Private local embeddings written; keep outputs outside the public repository.')
    except (ValueError,FileNotFoundError,FileExistsError) as exc:p.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
