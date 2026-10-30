# Data access, units and privacy

This repository distributes no patient images, identifiers, clinical tables,
labels, individual predictions or private split assignments. Obtain data through
the original custodians and confirm the applicable terms before processing it.

The source publication is Fridman et al., **BreastDCEDL: A standardized deep
learning-ready breast DCE-MRI dataset of 2,070 patients**, Scientific Data 13, 264
(2026), [DOI](https://doi.org/10.1038/s41597-026-06589-6).
The version-specific [Zenodo record 18114231](https://zenodo.org/records/18114231)
declares **CC BY-NC 4.0** for the dataset. Its access and noncommercial conditions
are separate from this repository's code license and from each source cohort's
terms. No medical-data download is executed automatically by this repository.

The paper's complete dataset must not be confused with the prepared subset used
in the local experiments:

| Unit | Prepared subset | Development | Historical TEST |
|---|---:|---:|---:|
| Patients | 1,273 | 1,097 | 176 |
| Matched PRE/EARLY/LATE slice triplets | 12,703 | 10,945 | 1,758 |
| Individual phase PNG files | 38,109 | 32,835 | 5,274 |

The matched clinical/fusion OOF comparison uses **1,092 eligible development
patients**, excluding five unknown historical outcomes. The images are not
38,109 independent patients. Multiple slices and phases from the same patient
must remain within one partition. Prepared metadata counts were checked without
copying its records into this repository.

Inputs are pretreatment DCE-MRI phases and baseline age/hormone receptor status
(HR). Numeric HR uses 0/1 encoding; it is not a one-hot feature. Values unavailable
at prediction time and post-treatment outcomes must never be used as predictors.
Individual acquisition timestamps were not completely established by the study;
baseline provenance is an explicit limitation rather than a guaranteed temporal
audit.

Keep authorized data and weights outside the checkout or in ignored local
directories. Provide paths explicitly to the preparation and evaluation scripts.
The example/schema and unit-test fixtures are artificial software examples and
are not evidence of reproducing the reported clinical metrics.

