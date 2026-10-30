# Local authorized input contracts

Reviewed public-case records and labels are in `evaluation/`, and model bundles
in `models/trained/`. Unit-test arrays remain synthetic software fixtures and
are never used to claim scientific performance.

The local slice metadata CSV must have columns:

```text
patient_id,label,fold,pre_path,early_path,late_path
```

One row represents one corresponding three-phase cut; repeated patient IDs form
a bag. Labels are binary, each patient belongs to one fold (0–4), and phase paths
are relative to the separately supplied authorized image root. Images must be
single-frame 256×256 grayscale 8-bit PNG files. Raw DICOM/physical intensities
cannot be substituted without a separately validated export pipeline.

Clinical local JSON accepts `age` and `HR` (binary 0/1); explicit null is allowed.
Both keys are required. Outputs remain private and must not be committed.

Fusion local JSON has `embedding64` (64 values extracted using the matching
frozen CNN) and `clinical` (age/HR object). Its fitted normalization and neural
head must come from the same authorized fold or final bundle. Do not mix folds.

## Preparing an authorized metadata export

`scripts/prepare_data.py --metadata prepared_development.csv --image-root authorized_images --output private_outputs/metadata.csv`

Run with real paths outside this repository. The adapter maps the original
prepared columns `patient_id,pCR,fold,path_pre,path_early,path_late` to the public
input contract. It changes no labels, folds, patient rows or images, checks
cross-patient phase-file ownership and file availability, and refuses to write
private metadata into this repository. If a `split` column is present it must
contain development `train` rows only; TEST must retain its separate protocol.
Study eligibility exclusions are not invented or recomputed: supply the correct
authorized prepared cohort. No data download, DICOM reconstruction, registration
or proprietary MRI export is implemented or claimed. The documented starting
point is the original grayscale PNG export.
