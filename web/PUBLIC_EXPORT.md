# Public copy of the finished research platform

This directory preserves the original platform's Spanish presentation, visual design, procedural animations, CNN laboratory, architecture explorer, EMA explanation, evaluation charts, and multimodal comparison. It is a controlled copy of the finished application, not a new design. The original private application was read only and remains unchanged.

`source-provenance.json` records relative source paths and SHA-256 fingerprints. Public runtime and rebuilding depend exclusively on files distributed here. No source directory, private server, or original workstation path is required.

## What remains functional

- Chapter navigation, scrolling presentation, theme and animation controls.
- Procedural educational DCE-MRI illustrations and convolution laboratory. These graphics are explicitly synthetic, not patient images or real trained activations.
- Manipulable CNN architecture, layer descriptions, EMA contributions and training-only BatchNorm explanation.
- Historical aggregate CNN learning, ROC and precision–recall curves, with plot inspection and zoom.
- The four-model, 1,092-patient OOF comparison and model-specific aggregate curves. The historical 176-patient EMA TEST remains a different protocol.
- Explanatory dialogs and downloading the sanitized aggregate presentation data.

## What is intentionally unavailable

- Individual patient identifiers, labels, scores and slice counts. The public patient array is empty; the patient explorer displays an explicit protected-data state.
- Trained weights and fitted normalizer/imputer parameters. The normalization toggle is disabled, without substituting invented parameters.
- Per-score ROC thresholds. Public ROC points contain false-positive and true-positive rates only.
- All calibration bins, because some original bins contain individual observations. No replacement curve is fabricated.
- Patient-dependent threshold exploration, threshold comparison and score histogram. Their panels state that private predictions are not distributed; they do not show artificial zero counts.
- Image uploads and live inference. Their original layout remains visible with disabled controls and a clear unavailable message. No API request, token, probe image, synthetic prediction or silent fallback is implemented in this public copy.

The five private clinical OOF evaluators remain distinct from the final private multimodal checkpoint. The latter has no independently measured performance. No model parameters or patient-level OOF records are included here.

## Build and inspect

From the repository root:

```console
python web/build_public.py
python web/build_public.py --check
python scripts/serve_web.py
```

Open the `/web/` path on the localhost address printed by the server. The generated HTML embeds only the sanitized local aggregate data and scripts. It does not call the private inference service. The public builder never reads the original project.

The source files are retained so the export is inspectable. The project license applies to copied project code; no UAX logo, institutional deck, patient image, private backend or third-party font is copied. Links to the original scientific context remain ordinary navigation links, not loaded resources.
