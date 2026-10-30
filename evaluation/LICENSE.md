# Scientific data and derived artifacts: separate from code licensing

BreastDCEDL originates from the dataset described in the
[Scientific Data paper](https://doi.org/10.1038/s41597-026-06589-6) and
[Zenodo record 18114231](https://zenodo.org/records/18114231). The upstream
dataset is made available under **Creative Commons Attribution–NonCommercial
4.0 International (CC BY-NC 4.0)**:
[license terms](https://creativecommons.org/licenses/by-nc/4.0/).

Retain attribution, identify adaptations and respect the noncommercial
restriction and any applicable source-dataset terms. Do not interpret this
repository's code license as permission for commercial use of medical data,
derived features or trained models. This file does not waive dataset rights or
provide clinical-use authorization. The MRI images themselves are not included.

The released patient/sample metadata was verified byte-for-byte against the
public teaching-case exports. Clinical age/HR values, labels and patient/sample
membership were independently compared with those exports before inclusion.
`PUBLIC_DATA_VERIFICATION.json` supplies their source SHA-256 fingerprints.
The exact publicly accessible references checked were
[patients.csv](https://storage.googleapis.com/usecasesf-breastdcedl-alumnos-8264/breastdcedl/metadata/patients.csv)
and
[samples.csv](https://storage.googleapis.com/usecasesf-breastdcedl-alumnos-8264/breastdcedl/metadata/samples.csv).
These public, de-identified teaching-case identifiers are retained for faithful
reproduction; no separately private teacher-validation records were included.

Project source code retains its separately declared **CC BY 4.0** license and
upstream teaching-case attribution described in the repository's root license
and attribution documentation. That code license does not relicense datasets,
model weights or scientific data derived from the upstream collection.

Scientific predictions, partitions and derived embeddings are released for
noncommercial academic reproducibility under the same conservative dataset
restriction. This is a redistribution notice, not a claim that all upstream
rights are owned by the project author. See root attribution documentation for
the teaching-case source URL and original author credits.

The original teaching-case publication/access instructions are preserved in
[original-data-access.md](../docs/original-data-access.md).
