# Static copy of the finished research platform

This directory preserves the original platform’s Spanish presentation, visual design, procedural animations, CNN laboratory, architecture explorer, EMA explanation, evaluation charts, and multimodal comparison. It is a controlled copy of the finished application, not a redesign. The original application remains unchanged.

**The repository and the browser page have different roles.** The repository includes trained-model artifacts, model metadata and evaluation files for local analysis. This web page remains static: it does not load those model files or individual evaluation records, accept MRI uploads, execute inference, or call a backend API. MRI images are not distributed.

`source-provenance.json` describes the browser bundle only and records relative source paths and SHA-256 fingerprints. Rebuilding the page uses exclusively the local web sources; it does not require the original workstation or its server.

## What remains functional in the web

- Chapter navigation, scrolling presentation, theme and animation controls.
- Procedural educational DCE-MRI illustrations and convolution laboratory. These graphics are explicitly synthetic, not patient images or trained activations.
- Manipulable CNN architecture, layer descriptions, EMA contributions and training-only BatchNorm explanation.
- Historical aggregate CNN learning, ROC and precision–recall curves, with plot inspection and zoom.
- The multimodal section’s three displayed OOF approaches: clinical-only, neural fusion and linear fusion, on 1,092 patients. The CNN-only OOF result remains preserved in the repository’s complete scientific results but is intentionally excluded from this section’s selector and table. The historical 176-patient EMA TEST remains a different protocol.
- Explanatory dialogs and downloading the aggregate presentation data used by the page.

## Artifacts available outside the browser bundle

Model weights, fitted preprocessing, model metadata and evaluation records are distributed separately under `models/trained/` and `evaluation/`. Consult the repository’s `models/README.md` and execution documentation for the exact artifact inventory and local commands. These files are not silently loaded into the page.

The five clinical OOF evaluators are fold-specific models, not a single clinical model trained on all patients. They remain distinct from the final multimodal checkpoint, which has no independently measured performance. Providing these artifacts does not establish new scientific validation.

Local image inference still requires compatible MRI obtained with legitimate access. The absence of MRI is explicit; the page does not substitute synthetic image results for measured model performance.

## What the static page intentionally does not load or execute

- Individual identifiers, labels, predictions and slice counts. The page’s patient array remains empty and the explorer explains that the repository’s evaluation files can be analysed locally.
- Model weights and fitted normalizers/imputers. These are separate repository artifacts, not browser state. The normalization toggle stays disabled, without invented replacement parameters.
- Per-score ROC thresholds or calibration bins. The page shows aggregate curves and does not reconstruct individual records from them. The original scientific results are not altered.
- Interactive patient-dependent threshold comparisons or score histograms. Their panels state that the page does not load the individual evaluation files; they do not show fabricated zero counts.
- MRI uploads or live inference. The original layout stays visible with disabled input and execution controls. There is no API request, backend connection, predicted score or silent fallback in this static copy.

## Build and inspect

From the repository root:

```console
python web/build_public.py
python web/build_public.py --check
python scripts/serve_web.py
```

Open `/web/` at the localhost address printed by the server. The generated HTML embeds the page’s aggregate data and educational scripts. Its public builder never reads the original project, a checkpoint or a patient table.

## Presentation assets and redistribution

This web directory contains no UAX logo, institutional slide deck, patient image, copied font binary or newly added institutional asset. The embedded presentations and architecture scenes are the original procedural HTML/canvas explanations. Existing scientific-context links are ordinary navigation links, not loaded third-party resources.

The project license applies to copied project code. Availability or licensing of model, dataset-derived and presentation files elsewhere in the repository is documented separately; this web note does not grant rights to third-party MRI or institutional branding.
