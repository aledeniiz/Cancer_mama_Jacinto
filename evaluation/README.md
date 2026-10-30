# Released scientific evidence — no MRI images

This directory supports an actual replay of the original scores, labels,
patient partitions and frozen-feature model heads. It contains no synthetic
cohort and no private teacher-validation data. The user explicitly authorized
release of the project except medical images; before inclusion every released
patient/sample identity, label, age and HR value was checked against the
downloaded public teaching-case exports. See `PUBLIC_DATA_VERIFICATION.json`.

## One command

After installing the repository's documented dependencies:

```bash
python scripts/reproduce_results.py
```

The command makes no network requests, fits no model, changes no files by
default and requires no MRI images. It checks SHA-256 integrity, identities,
coverage, original order, fold independence, labels and clinical inputs; then
recomputes ROC-AUC/BCE of all four OOF score columns and the separate historical
CNN EMA TEST predictions. It reloads the five clinical models, CNN classifiers
and neural fusion heads against their original 64-D CNN embeddings. The EMA
classifier is checked on development embeddings as a parity test, without
assigning development or final-head performance as independent validation.

`--metrics-only` explicitly skips checkpoint reload. `--output` writes an
optional **new** aggregate JSON report and refuses to overwrite an existing one.

| Original protocol | Patients | Model | ROC-AUC |
|---|---:|---|---:|
| Five-fold OOF | 1,092 | CNN fold models | 0.5197 |
| Five-fold OOF | 1,092 | Age + HR | 0.6450 |
| Five-fold OOF | 1,092 | Neural fusion | 0.6326 |
| Five-fold OOF | 1,092 | Linear fusion | 0.6355 |
| Historical separate TEST | 176 | CNN EMA | 0.5524 |

The OOF CNN models are ordinary fold-local compact CNNs; they are not the
globally selected historical EMA checkpoint. These protocols must not be mixed.
The fusion did not demonstrate an advantage over clinical variables alone.
The all-development multimodal final head has no independent performance
estimate. The original linear-fusion estimator was not persisted, so its stored
score metrics can be recomputed but its model cannot be reloaded faithfully;
no replacement estimator has been trained.

## Contents and populations

- `metadata/`: exact public teaching-case `patients.csv` (1,273 patients) and
  `samples.csv` (12,703 three-phase cut records). Image paths remain references;
  the 38,109 PNG files themselves are excluded.
- `clinical/`: original age/HR table and eligibility notes, values verified
  against public metadata. The sentinel −1 and NaN represent original missing
  information; no clinical values or outcomes were invented.
- `oof/`: original four-model predictions for 1,092 eligible development
  patients in their original order. Three age values and two HR values are
  missing as expected. Five development patients with uncertain original
  I-SPY1 outcome provenance were excluded from the new clinical/fusion fits.
- `test/`: original 176-patient CNN EMA scores, outcomes and slice counts.
- `partitions/`: 1,097 development and 176 TEST identities, 1,092 eligible OOF
  row positions, and the exact historical sample-level fit/inner/refit/outer
  roles for five CNN folds. Historical CNN fitting still included the five
  excluded noisy-label cases; that residual limitation is retained.
- `embeddings/`: original real 64-D features and scores for each frozen CNN
  fold and the historical EMA. Only corrected `loaded_state_v2` extracts are
  included. The invalid random-initialization extraction is excluded.
- `metrics/`: original aggregate results, uncertainty and limitations.
- `protocols/`: original configurations, environment and completed ledger.
  Absolute machine paths are sanitized; source/export hashes are recorded.
- `source_snapshot/`: six exact historical scripts plus the original fusion
  module and API, for provenance. These are not the installed portable code.

## What this reproduces — and what it does not

Stored-score metric replay is checked to 1e-14. Recovered clinical fold
probabilities agree to machine precision. CPU/library differences in float32
CNN/fusion classifier arithmetic are allowed up to 1e-6 in probability, with
metric differences explicitly reported. Embedding checkpoint identities are
checked against the original ledger and released model metadata.

The original neural-fusion score column was float32, while CNN, clinical,
linear-fusion and historical TEST scores were float64. The replay restores these
original types before calling the original sklearn metric functions. Reading
shortest-decimal float32 CSV values as float64 changes fusion BCE by about
1e-9; preserving the original dtype recovers its exact stored BCE. This is a
serialization/numerical policy, not a changed score or modified result; see
`protocols/metric_replay.json`.

The MRI encoder cannot be rerun without images. Replaying derived features is
not image preprocessing/extraction reproduction. This is not a new model fit,
new experiment, independent validation or evidence of clinical utility. The
historical TEST had been used previously with another model; that limitation
is preserved in its original metric record.

Historical snapshot scripts rely on their original sealed campaign layout and
may contain legacy pickle loading. They must not be used to load foreign
checkpoints. The portable installed package uses non-pickle JSON/NPZ exports.
Sanitizing paths does not turn snapshots into a self-contained historical
training runner or establish bitwise GPU training reproducibility.

`SOURCE_PROVENANCE.json` records original/export hashes and any path adaptation.
`MANIFEST.json` and its SHA-256 seal cover this scientific release. Dataset
rights and source-code licensing are separate: read [LICENSE.md](LICENSE.md).
