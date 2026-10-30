"""Local authorized data only. The repository contains no medical records."""
from pathlib import Path
import torch
from .imaging import preprocess
from .evaluation import validate_patient_table


def read_bags(frame, image_root):
    validate_patient_table(frame,image_root=image_root)
    root=Path(image_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Authorized local image root is required")
    for identifier,group in frame.groupby("patient_id",sort=True):
        slices=[]
        for row in group.itertuples():
            sources={phase:(root/str(getattr(row,column)).replace('\\','/')).resolve() for phase,column in
                (("PRE","pre_path"),("EARLY","early_path"),("LATE","late_path"))}
            if any(not p.is_relative_to(root) for p in sources.values()):
                raise ValueError("Image paths must stay inside the authorized image root")
            slices.append(preprocess(sources))
        yield str(identifier),torch.stack(slices),float(group.label.iloc[0])
