# Attribution and adaptation notice

This is a curated public software/documentation edition of the local
`Cancer_mama_Jacinto` student research project, maintained in its local Git
metadata under the public handle **aledeniiz**. It is not a mirror of the original
private working directory or its history. Code was selected/adapted for portable
paths, explicit model-artifact inputs, privacy-safe outputs and executable tests.
No scientific result was changed by this preparation.

The source project declares Creative Commons Attribution 4.0 International;
its `LICENSE` is preserved verbatim. This notice does not assert that one person
owns every contribution or that the dataset license is the software license.

The underlying data-preparation project is
[BreastDCEDL](https://github.com/naomifridman/BreastDCEDL). Its upstream software
license identifies **Copyright 2025 Naomi Fridman**, CC BY 4.0. Retain upstream
attribution where its material is used. The compact CNN and fusion investigation
described here are the student's research implementation; do not attribute the
whole local implementation or its findings to the dataset authors.

Dataset/publication citation:

Fridman, N., Solway, B., Fridman, T., Barnea, I. & Goldstein, A.
*BreastDCEDL: A standardized deep learning-ready breast DCE-MRI dataset of 2,070
patients*. Scientific Data **13**, 264 (2026).
[DOI: 10.1038/s41597-026-06589-6](https://doi.org/10.1038/s41597-026-06589-6).

The prepared experimental subset contains 1,273 patients; this is not the complete
source dataset. Dataset access is governed separately by the
[version-specific Zenodo record](https://zenodo.org/records/18114231) and source
cohort terms. Reviewed public-case metadata, predictions, embeddings and preserved model
parameters are included under the separate research/noncommercial conditions
described in docs/redistribution.md. MRI images, private teacher validation,
standalone institutional templates/logos and publisher figures are not redistributed.
The student's final academic presentation is included with institutional branding
preserved; third-party marks/assets are not relicensed by the code LICENSE.
No endorsement by UAX, dataset authors,
source cohorts or clinical institutions is implied.

The browser inference runtime is Microsoft ONNX Runtime Web 1.31.1, redistributed
under its separate MIT license and third-party notices in web/inference-assets/.
The source-project CC license does not replace those component licenses.
