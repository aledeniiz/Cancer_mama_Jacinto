"""Original fixed-width fusion head: frozen image64 plus clinical4 -> logit."""
import numpy as np
import torch
from torch import nn
from .clinical import encode_clinical


class FusionNet(nn.Module):
    def __init__(self, clinical_dim=4):
        super().__init__()
        if clinical_dim != 4:
            raise ValueError("Original age/HR transform has four encoded features")
        self.clinical=nn.Sequential(nn.Linear(clinical_dim,8),nn.ReLU(),nn.Dropout(.2))
        self.fusion=nn.Sequential(nn.Linear(72,16),nn.ReLU(),nn.Dropout(.3),nn.Linear(16,1))

    def forward(self,image,clinical):
        if image.ndim!=2 or image.shape[1]!=64 or clinical.shape!=(len(image),4):
            raise ValueError("Expected aligned image64 and clinical4 matrices")
        if not torch.isfinite(image).all() or not torch.isfinite(clinical).all():
            raise ValueError("Nonfinite fusion input")
        return self.fusion(torch.cat((image,self.clinical(clinical)),1)).flatten()


def predict_embedding(embedding64,clinical_record,bundle):
    net=FusionNet();net.load_state_dict(bundle['head_state'],strict=True);net.eval()
    embedding=np.asarray(embedding64,dtype='float32').reshape(1,64)
    mean=np.asarray(bundle['embedding_mean']);std=np.asarray(bundle['embedding_std'])
    if mean.shape!=(64,) or std.shape!=(64,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
        raise ValueError("Invalid training-fitted embedding normalization")
    embedding=(embedding-mean)/std
    with torch.inference_mode():
        return float(net(torch.tensor(embedding,dtype=torch.float32),
            torch.tensor(encode_clinical(clinical_record,bundle['clinical_transform']))).sigmoid().item())


def fit_fusion_head(training_embeddings,training_clinical,training_labels,epochs=80,seed=42):
    """Original fixed 80-epoch recipe; never invoke on outer/test for selection.

    This function is provided for authorized reproduction, not executed by the
    repository tests and not a claim that released aggregate results were rerun.
    Inputs must already use TRAIN-fitted image/clinical transforms.
    """
    if type(epochs) is not int or epochs<1:
        raise ValueError('Positive integer fusion epochs required; refusing an untrained random head')
    x=torch.as_tensor(training_embeddings,dtype=torch.float32)
    c=torch.as_tensor(training_clinical,dtype=torch.float32)
    y=torch.as_tensor(training_labels,dtype=torch.float32)
    if x.ndim!=2 or len(x)==0 or x.shape[1]!=64 or c.shape!=(len(x),4):
        raise ValueError('Nonempty aligned image64 and clinical4 TRAIN matrices required')
    if not torch.isfinite(x).all() or not torch.isfinite(c).all():
        raise ValueError('Finite TRAIN features required')
    if y.shape!=(len(x),) or not torch.isfinite(y).all() or not torch.isin(y,torch.tensor([0.,1.])).all():
        raise ValueError("One binary outcome per training patient required")
    torch.manual_seed(seed);net=FusionNet()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    for _ in range(epochs):
        net.train()
        for ix in torch.randperm(len(y)).split(64):
            optimizer.zero_grad()
            loss=nn.functional.binary_cross_entropy_with_logits(net(x[ix],c[ix]),y[ix])
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step()
    return net.eval()
