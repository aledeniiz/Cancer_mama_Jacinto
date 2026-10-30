# Preserved trained models: image, clinical and multimodal

This directory contains lossless inference exports of the existing trained
models. No architecture, result, parameter or transform was refitted, optimized
or replaced for release. Original checkpoints and backups remain untouched.
Only exact inference tensors, fitted transforms, configuration, versions and
aggregate provenance are carried here; optimizer states, random-generator state,
patient rows, fitting identifiers and workstation paths are excluded.

## Model registry and interfaces

| Directory | Artifacts | Scope |
|---|---:|---|
| `cnn_ema/` | 1 CNN | Historical compact EMA 0.8, epoch 15; train-only BatchNorm recalibration |
| `cnn_oof/fold_0..4/` | 5 CNNs | Ordinary fold-local historical outer-fold image encoders; not the globally selected EMA |
| `fusion_oof/fold_0..4/` | 5 neural heads | Age/HR clinical transforms plus matching frozen image features for each outer fold |
| `multimodal_final/` | 1 neural head | Development-fitted head using the historical EMA; no independent performance estimate |
| `clinical_oof/fold_0..4/model.json` | 5 clinical predictors | Preserved/recovered original clinical OOF evaluators; not a global final model |

`registry.json` binds every model to its source artifact SHA-256, fold and scope.
CNN/head directories contain `metadata.json` and `weights.npz`, compatible with
`breastdce_research.inference.load_cnn` / `load_checkpoint`. NPZ tensors are loaded
with pickle disabled and bounded header/array checks. Original pickle `.pt`
files are not redistributed. CNNs retain exactly 24,713 trainable parameters;
each neural clinical/fusion head has 1,225 trainable parameters.

Clinical model JSON is compatible with
`breastdce_research.clinical.ClinicalFoldPredictor.load`. Its coefficient order,
intercept, median imputation, scaling and missingness encoding are unchanged.
The intercepts were algebraically recovered in the earlier preservation phase
because the original fitting script never saved them. The preserved predictors
match original OOF probabilities to machine precision; identity with the bits
of a never-serialized original intercept is not asserted. No separate global
clinical-only estimator was fitted by that study, and none is fabricated here.

The original linear-fusion fitted estimator was **not saved**. Its aggregate
metrics and stored scores can be replayed, but exact arbitrary-input inference
cannot be recovered from this directory. No substitute linear model was trained.

## Input transforms and model pairing

CNN inputs are matching PRE/EARLY/LATE exported PNG channels, grayscale 8-bit,
256×256. Intensities are divided by 255 and standardized with each CNN's exact
saved training-fitted channel means/std. Four convolution/BatchNorm/ReLU/pooling
blocks create 64-dimensional slice embeddings, averaged at patient level before
dropout/linear classification. Raw DICOM or arbitrary screenshots are not direct
substitutes for that export pipeline.

Neural heads retain their own clinical median/mean/std and 64-dimensional image
embedding mean/std. HR is numeric 0/1 with missingness indicators, not one-hot.
Order after clinical transformation is age-scaled, HR-scaled, age-missing,
HR-missing. Each head's `image_checkpoint_sha256` identifies its matching CNN.
Do not substitute the final EMA for an OOF image encoder or combine transforms
from different folds. Fitted normalizers are parameters required for inference;
their patient/sample fitting lists are not needed and are not copied.

## Scientific interpretation

The matched 1,092-patient OOF comparison uses the ordinary five fold CNNs and
fold-specific heads/clinical predictors. Historical EMA TEST AUC 0.5524 belongs
to a different 176-patient protocol. OOF AUC 0.6326 must not be assigned to the
final development-fitted multimodal head. Clinical-only has the highest observed
OOF AUC 0.6450; no fusion advantage over clinical-only was demonstrated.

Development data were reused exploratorily. Five unknown outcomes were excluded
from clinical/fusion evaluation but had affected historical image encoders.
Intervals are conditional and do not include all model-selection/retraining
uncertainty. The historical TEST had already been used with other models.
Neither external validation nor clinical utility is established. Academic and
educational research only; these models do not support treatment decisions.

## Verification and remaining reproduction boundary

`tensor_verification.json` records exact equality of all **192 state tensors**
across six CNNs and six heads, including names, shapes, dtypes and values. Fitted
normalizers, embedding transforms and clinical coefficients/intercepts were
copied without numerical change. Source environment versions are retained;
export verification uses patched PyTorch 2.10.0 CPU rather than the historical
2.5.1+cu121 environment. This is preservation, not retraining.

Run from an installed checkout or with the documented source dependencies:

```sh
python -B models/trained/verify_reload.py
```

The script checks artifact hashes and cold-loads all 17 models in separate
processes with small artificial software fixtures. These fixtures never claim
scientific performance. `reload_verification.json` records the performed result.
The file manifest excludes itself and that final post-manifest verification
report to avoid circular hashing.

Original historical partition hashes are preserved with each fold. They are not
invented current-format `partition_lineage` values: the generic image-evaluation
CLI therefore rejects these historical bundles for a fresh held-out claim until
their lineage has been explicitly matched to the supplied authorized metadata.
Stored embedding/score replay and renewed MRI image encoding are different
checks. Exact MRI encoding additionally requires the corresponding images,
which are deliberately excluded from the source release.
