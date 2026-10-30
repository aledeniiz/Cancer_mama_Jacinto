"""Label-free TRAIN slice view and cut-weighted cumulative BN recalibration."""
import torch
from torch import nn
from torch.utils.data import Dataset


class LabelFreeTrainingSlices(Dataset):
    def __init__(self, rows, train_ids, validation_ids, cache, normalization):
        actual=set(rows.patient_id)
        if actual!=set(train_ids) or actual&set(validation_ids):
            raise ValueError('Calibration requires exactly all fitting TRAIN patients')
        if ('fold' in rows and rows.fold.eq(0).any()) or ('split' in rows and not rows.split.eq('train').all()):
            raise ValueError('Calibration forbids validation or TEST samples')
        if rows.sample_id.duplicated().any():
            raise ValueError('Duplicate calibration cut')
        self.rows=rows[['patient_id','sample_id','slice_index']].sort_values(['patient_id','slice_index']).to_dict('records')
        self.cache,self.normalization=cache,normalization

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        from breastdcedl.university.data import standardize
        return standardize(self.cache.get(self.rows[index]['sample_id']),self.normalization)


def recalibrate(model, loader, device):
    layers=[m for m in model.modules() if isinstance(m,nn.BatchNorm2d)]
    if not layers or any(not m.track_running_stats for m in layers):
        raise ValueError('Tracked BatchNorm2d required')
    momentum={layer:layer.momentum for layer in layers}
    model.eval()
    for layer in layers:
        layer.reset_running_stats();layer.train()
    count=batches=0
    try:
        with torch.no_grad():
            for x in loader:
                if not isinstance(x,torch.Tensor) or x.ndim!=4 or not torch.isfinite(x).all():
                    raise ValueError('Expected finite label-free slice tensor')
                n=len(x)
                if n<1:
                    raise ValueError('Empty calibration batch')
                for layer in layers:
                    layer.momentum=n/(count+n)
                model.slice_logits(x.to(device)) if hasattr(model,'slice_logits') else model(x.to(device))
                count+=n;batches+=1
        if not count:
            raise ValueError('Empty calibration loader')
        if any(not torch.isfinite(layer.running_mean).all() or not torch.isfinite(layer.running_var).all()
               or (layer.running_var<0).any() or int(layer.num_batches_tracked)!=batches for layer in layers):
            raise ValueError('Invalid recalibrated BN buffers')
    finally:
        for layer in layers:
            layer.momentum=momentum[layer]
        model.eval()
    return {'n_slices':count,'n_batches':batches,'weighting':'cut-weighted cumulative batch moments; not exact global population variance'}


def verify_state_change(original_state, recalibrated):
    allowed={f'{name}.{suffix}' for name,layer in recalibrated.named_modules() if isinstance(layer,nn.BatchNorm2d)
             for suffix in ('running_mean','running_var','num_batches_tracked')}
    after={name:value.detach().cpu() for name,value in recalibrated.state_dict().items()}
    if after.keys()!=original_state.keys():
        raise ValueError('State structure changed')
    changed=[name for name,value in original_state.items() if not torch.equal(value.cpu(),after[name])]
    if not set(changed)<=allowed:
        raise ValueError('Learned parameter or non-BN buffer changed')
    if any(parameter.grad is not None for parameter in recalibrated.parameters()):
        raise ValueError('Recalibration created gradients')
    return changed,after
