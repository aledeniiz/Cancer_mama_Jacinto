"""Portable authorized-data training entrypoint; not a historical rerun claim.

The historical experiment also used frozen data/cache/source manifests. Those
private manifests are not distributed, so exact metric reproduction requires
the original authorized inputs and partition definitions.
"""
import math
import torch
from torch import nn
from .data import read_bags
from .evaluation import patient_split,binary_metrics,partition_lineage
from .imaging import fit_image_standardizer,standardize
from .models import CompactPatientCNN


def moderate_geometry(x,generator):
    # Same per-bag, phase-coherent affine recipe as the original patient pipeline.
    draws=torch.rand(5,generator=generator)
    angle=float((draws[0]*14-7)*math.pi/180);scale=float(.95+draws[1]*.1)
    tx,ty=float((draws[2]*2-1)*.06),float((draws[3]*2-1)*.06)
    theta=x.new_tensor([[math.cos(angle)/scale,-math.sin(angle)/scale,tx],
                        [math.sin(angle)/scale,math.cos(angle)/scale,ty]])
    grid=nn.functional.affine_grid(theta[None].expand(len(x),-1,-1),x.shape,align_corners=False)
    return nn.functional.grid_sample(x,grid,mode='bilinear',padding_mode='border',align_corners=False)


def evaluate_bags(model,bags,normalization):
    model.eval();labels=[];probabilities=[]
    with torch.inference_mode():
        for _,images,label in bags:
            labels.append(label)
            probabilities.append(float(model.forward_patient(standardize(images,normalization)).sigmoid().item()))
    return binary_metrics(labels,probabilities)


def train_cnn(frame,image_root,fold=0,epochs=15,seed=42):
    """Explicit opt-in API; no training is invoked during install or tests.

    CPU-reference adaptation: patient-balanced BCE, same compact architecture,
    train-only normalization, 10 sampled cuts, 8 patients/update, moderate bag
    augmentation, AdamW/cosine. Device/RNG/loader changes preclude bitwise claims.
    This produces raw epoch states; apply post-hoc EMA and TRAIN BN separately.
    """
    if type(epochs) is not int or epochs<1:
        raise ValueError("Positive integer epochs required")
    torch.manual_seed(seed);generator=torch.Generator().manual_seed(seed)
    train,validation=patient_split(frame,fold,image_root=image_root)
    lineage=partition_lineage(frame,fold)
    normalization=fit_image_standardizer(bag for _,bag,_ in read_bags(train,image_root))
    model=CompactPatientCNN();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    # Return epoch states to the caller; the public CLI creates new local files.
    for epoch in range(1,epochs+1):
        model.train();people=list(read_bags(train,image_root))
        order=torch.randperm(len(people),generator=generator);optimizer.zero_grad();window=0
        for position,index in enumerate(order.tolist()):
            _,images,label=people[index]
            subset=torch.randperm(len(images),generator=generator)[:10]
            images=moderate_geometry(standardize(images[subset],normalization),generator)
            logit=model.forward_patient(images)
            loss=nn.functional.binary_cross_entropy_with_logits(logit,torch.tensor([[label]]))
            loss.backward();window+=1
            if window==8 or position==len(order)-1:
                for p in model.parameters():
                    if p.grad is not None:p.grad.div_(window)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step();optimizer.zero_grad();window=0
        scheduler.step()
        yield {'epoch':epoch,'model_state_dict':{n:v.detach().clone() for n,v in model.state_dict().items()},
            'normalization':normalization,'config':{'search_variant':'compact','dropout':.3},
            'partition_lineage':lineage,
            'metrics':{'train':evaluate_bags(model,read_bags(train,image_root),normalization),
                       'validation':evaluate_bags(model,read_bags(validation,image_root),normalization)},
            'scope':'new authorized-data raw-epoch run; not a released historical checkpoint'}
