"""Strict exported PNG preprocessing; PRE/EARLY/LATE, not raw DICOM intensities."""
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

PHASES = ("PRE", "EARLY", "LATE")
MAX_IMAGE_BYTES = 2 * 1024 * 1024


def load_phase(source):
    if isinstance(source, bytes):
        payload = source
    else:
        source = Path(source)
        if source.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("PNG exceeds 2 MiB limit")
        payload = source.read_bytes()
    if not 0 < len(payload) <= MAX_IMAGE_BYTES:
        raise ValueError("Empty or excessive PNG")
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format != "PNG" or image.size != (256,256) or image.mode != "L" or getattr(image,"n_frames",1) != 1:
                raise ValueError("Expected a single 256x256 8-bit grayscale PNG")
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            pixels = np.array(image,dtype=np.float32) / np.float32(255.0)
    except (UnidentifiedImageError,OSError,SyntaxError,Image.DecompressionBombError) as exc:
        raise ValueError("Corrupt or unreadable PNG") from exc
    return torch.from_numpy(pixels)


def preprocess(phases):
    if set(phases) != set(PHASES):
        raise ValueError("Exactly PRE, EARLY and LATE are required")
    return torch.stack([load_phase(phases[phase]) for phase in PHASES])


def standardize(x, state):
    if state.get("representation", "original") != "original":
        raise ValueError("This release preserves original PRE/EARLY/LATE representation only")
    means = torch.as_tensor(state["means"],dtype=x.dtype,device=x.device).view(3,1,1)
    std = torch.as_tensor(state["std"],dtype=x.dtype,device=x.device).view(3,1,1)
    if not torch.isfinite(means).all() or not torch.isfinite(std).all() or (std<=0).any():
        raise ValueError("Invalid persisted normalization")
    return (x-means)/std


def fit_image_standardizer(training_bags):
    """Call with training images only; no labels, validation or test images."""
    sums, squares, count = torch.zeros(3,dtype=torch.float64),torch.zeros(3,dtype=torch.float64),0
    for bag in training_bags:
        if bag.ndim != 4 or tuple(bag.shape[1:]) != (3,256,256) or not torch.isfinite(bag).all():
            raise ValueError("Expected finite raw patient bags")
        values=bag.double().permute(1,0,2,3).reshape(3,-1)
        sums += values.sum(1); squares += values.square().sum(1); count += values.shape[1]
    if count == 0:
        raise ValueError("No training images")
    means=sums/count;std=(squares/count-means.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    return {"means":means.tolist(),"std":std.tolist(),"representation":"original"}
