# Model resources are intentionally withheld

This directory contains no trained weights, clinical coefficients, fitted
normalization values or medical data. Distribution rights/privacy clearance for
the original learned artifacts have not been established. Small size is not
permission to publish them.

The source implements the original **24,713-parameter compact CNN**, post-hoc
epoch EMA (decay 0.8) plus train-only BatchNorm recalibration, the age/HR clinical
pipeline and the 1,225-parameter neural fusion head. All are explicit code, not
pretrained substitutes.

For authorized local use, keep resources outside the repository:

- CNN: a non-pickle JSON/NPZ bundle with `model_state_dict`,
  `normalization` (`means`, `std`, `representation`) and compact `config`.
  Held-out evaluation additionally requires `partition_lineage`, binding the
  checkpoint to the exact metadata and training/validation patient partition.
  The public training entrypoint writes these fingerprints automatically.
- Clinical: select one of five private fold-specific `model.json` files. The
  five OOF evaluators are not a single global clinical model. The original study
  did not save/train a separate all-development clinical estimator.
- Fusion: `head_state`, training-fitted `clinical_transform`, `embedding_mean`,
  `embedding_std`, and the matching frozen CNN used for 64-D extraction. The
  final all-development head is distinct from fold heads and has no independent
  performance estimate. Do not transfer its OOF score to that checkpoint.

Public loaders never call `torch.load`. A bundle directory contains
`metadata.json` (format `breastdce-json-npz-v1`) and `weights.npz`; tensors use
keys `model_state_dict::parameter.name` or `head_state::parameter.name` and
`numpy.load(allow_pickle=False)`. Archive size is bounded. Historical `.pt`
checkpoints require trusted private conversion outside this loader. No unsafe
fallback or arbitrary pickle loading is provided. New local runs write this
same non-pickle format without changing tensor values.

`evaluate.py` rejects an absent lineage, a different fold or altered metadata
before loading images/scoring. EMA additionally requires identical partition
lineage and normalization across chronological epochs. A private legacy
conversion must carry provenance verified against the original fitting inputs;
do not invent lineage merely to pass these checks. Missing lineage permits only
inference with no held-out performance assertion. Fingerprints detect accidental
mismatch; they do not authenticate a maliciously forged training history or
prove image content equivalence.

The historical experiment used PyTorch 2.5.1; this public package specifies
2.10.0, without altering architecture or published results. The original
environment is evidence, not a recommendation for loading foreign models.
Upstream advisories explain why `weights_only=True` alone was insufficient:
[2025 advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)
and [2026 advisory, patched in 2.10.0](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
Public loaders avoid pickle entirely, including on patched runtimes.

Without these private resources the public web and code tests work, but
historical patient inference and exact scientific metric reproduction do not.
