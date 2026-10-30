"""Post-hoc epoch-parameter EMA and label-free TRAIN BatchNorm recalibration.

The historical EMA decay was .8 over chronological epoch checkpoints. It was
not an attention layer, a separately trained CNN or a test-tuned predictor.
"""
import torch
from torch import nn


def ema_parameters(model,epoch_states,decay=.8):
    if not 0<=decay<1:
        raise ValueError("EMA decay must be in [0,1)")
    names=set(dict(model.named_parameters()));average=None;latest=None
    for state in epoch_states:
        if set(state)!=set(model.state_dict()):
            raise ValueError("Checkpoint state structure mismatch")
        latest={name:value.detach().clone() for name,value in state.items()}
        if average is None:
            average={name:latest[name].clone() for name in names}
        else:
            for name in names:
                average[name].mul_(decay).add_(latest[name],alpha=1-decay)
    if average is None:
        raise ValueError("At least one chronological checkpoint required")
    latest.update(average)
    return latest


def recalibrate_batchnorm(model,training_batches,training_ids,heldout_ids):
    """Caller supplies only training images, normalized by TRAIN statistics.

    Cut-weighted cumulative batch moments preserve the historical operation;
    they are not exact pooled population variance. Dropout remains disabled.
    """
    if not set(training_ids) or set(training_ids)&set(heldout_ids):
        raise ValueError("Disjoint nonempty training population required")
    layers=[m for m in model.modules() if isinstance(m,nn.BatchNorm2d)]
    if not layers or any(not m.track_running_stats for m in layers):
        raise ValueError("Tracked BatchNorm2d required")
    momentum={m:m.momentum for m in layers};model.eval()
    parameters={n:v.detach().clone() for n,v in model.named_parameters()}
    for layer in layers:layer.reset_running_stats();layer.train()
    count=batches=0
    try:
        with torch.no_grad():
            for x in training_batches:
                if x.ndim!=4 or len(x)==0 or not torch.isfinite(x).all():
                    raise ValueError("Expected finite nonempty label-free slices")
                for layer in layers:layer.momentum=len(x)/(count+len(x))
                model._encode(x)
                count+=len(x);batches+=1
        if count==0:
            raise ValueError("Empty calibration loader")
    finally:
        for layer in layers:layer.momentum=momentum[layer]
        model.eval()
    if any(not torch.equal(p,parameters[n]) for n,p in model.named_parameters()):
        raise RuntimeError("Recalibration changed learned parameters")
    return {'n_slices':count,'n_batches':batches,'weighting':'cut-weighted cumulative batch moments'}
