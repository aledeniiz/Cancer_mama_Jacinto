"""Exact compact architecture adapted from the frozen SearchPatientCNN source.

Only the compact branch is retained. Backbone state keys remain compatible;
the wider prospective/attention variants are deliberately not substituted.
"""
import torch
from torch import nn


class _CompactBackbone(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        blocks, previous = [], 3
        for channels in (8, 16, 32, 64):
            blocks.append(nn.Sequential(nn.Conv2d(previous, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2)))
            previous = channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, 1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)


class CompactPatientCNN(nn.Module):
    """24,713 learned parameters; one logit per patient, 64-D mean embedding."""
    embed_dim = 64
    model_version = "scratch-cnn-search-20261026-v1-compact"

    def __init__(self, dropout=0.3):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.backbone = _CompactBackbone(dropout)

    def _encode(self, slices):
        if (slices.ndim != 4 or tuple(slices.shape[1:]) != (3, 256, 256)
                or len(slices) == 0 or not slices.is_floating_point() or not torch.isfinite(slices).all()):
            raise ValueError("Expected finite nonempty PRE/EARLY/LATE slices [N,3,256,256]")
        return self.backbone.pool(self.backbone.features(slices)).flatten(1)

    def _classify(self, pooled):
        result = torch.cat([self.backbone.classifier(self.backbone.dropout(h)) for h in pooled.split(1)]).float()
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite patient logit")
        return result

    def patient_embeddings(self, x, mask):
        if x.ndim != 5 or tuple(x.shape[2:]) != (3,256,256) or not x.is_floating_point():
            raise ValueError("Expected floating point patient bags [B,K,3,256,256]")
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(x.shape[:2]) or mask.device != x.device:
            raise ValueError("Expected aligned boolean patient mask")
        counts = mask.sum(1)
        if len(x) == 0 or (counts == 0).any():
            raise ValueError("Every patient needs a real slice")
        embeddings = self._encode(x[mask])
        return torch.cat([h.float().mean(0,keepdim=True) for h in embeddings.split(counts.tolist())])

    def forward(self, x, mask):
        return self._classify(self.patient_embeddings(x, mask))

    def forward_patient(self, slices, chunk_size=16):
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("Positive integer inference chunk size required")
        embeddings = torch.cat([self._encode(chunk) for chunk in slices.split(chunk_size)])
        return self._classify(embeddings.float().mean(0,keepdim=True))

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())
