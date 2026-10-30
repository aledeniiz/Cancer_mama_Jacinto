# Verification of the public edition

Verification date: 30 October 2026. No new training, hyperparameter optimization
or scientific experiment was executed. Tests use artificial software fixtures;
they do not reproduce the private clinical results.

## Fresh installation and portability

A new Python 3.11 environment was created outside the release directory. CPU
PyTorch 2.10.0 and `.[test,plots]` were installed from package registries. The
runtime versions are in [tested-environment.json](tested-environment.json).
The source project, its virtual environment and its checkpoints were not used
as dependencies of the new installation.

**16 tests passed** in the new environment. A second copy of the public checkout
was placed in a separate directory: **all 16 tests passed there**, and its web
builder check passed. Imports in that test select the relocated source directory.
No original workstation path is required. The safe checkpoint loader rejects
pickle files, malformed array headers and unexpected tensor shapes/types.

The suite includes wrong-fold/metadata-lineage rejection, phase-file ownership,
training-only transforms, architecture and aggregation, missing local resources,
invalid labels/empty fusion inputs, and the local server path allow-list. Invalid
training requests are rejected before fitting; no training loop was run.

Ten script `--help` contracts were exercised successfully, including preparation,
training, evaluation, CNN/clinical/fusion inference, EMA and the server. Figures
were rendered from the public aggregate JSON. These operations are software and
export checks, not an independent validation of the trained models.

## Faithful web export

The finished platform's Spanish design, chapters, procedural graphics and
multimodal section are preserved. Browser inspection covered the normal viewport
and a 390-pixel mobile viewport. No JavaScript errors were reported during those
checks. All **six image-upload controls** were verified disabled in the public
copy. Patient records, learned normalizers, calibration bins and score thresholds
are withheld; no inference API is called and no prediction fallback is invented.

`python web/build_public.py --check` verifies the generated page against its own
sanitized source files. The same check passed after relocation. See
[PUBLIC_EXPORT.md](../web/PUBLIC_EXPORT.md) for exactly which interactions remain
available and which depend on private resources.

The local server returned 200 for the public web and aggregate JSON resources,
and 404 for private/dotfile probe paths. Its tests also exercise nested private
README/LICENSE denial; a 404 for an absent file alone is not a security proof.
No server was deployed publicly and the original private platform was untouched.

## Privacy and file review

Every release file is inspected, regardless of Git ignore status. The local
scanner checks known credential patterns, filesystem paths, unexpected binaries,
large files, forbidden data/model/cache formats, embedded binary images and
exact duplicate files. Matched values are not printed. A separate private scan
compared release text against all **1,273 known patient identifiers** and found
no matches; the identifier list and private scan inputs are not distributed.

The sanitized presentation payload has zero patient rows, no learned normalizer
values and no calibration bins. Aggregate ROC/PR coordinates contain no score
thresholds. No real weights, clinical coefficients, fitted statistics, individual
OOF/TEST rows, institutional deck, source medical image, credential or private
local experiment history is included in the reviewed release tree. Existing
public commits are preserved; old records were not purged from that history.
Artificial test constants are not learned parameters.

These are bounded checks, not a proof against every possible secret or privacy
attack. Rights review and future additions still require human review. The code
license is preserved; dataset terms and excluded-resource decisions are documented
in [redistribution.md](redistribution.md) and [NOTICE.md](../NOTICE.md).

## Repeatable release checks

```sh
python -B -m pytest -q
python -B web/build_public.py --check
python -B scripts/audit_release.py
python -B scripts/render_results.py
```

To create a **new**, nonexisting release manifest after inspecting every file:

```sh
python -B scripts/audit_release.py --write-manifest docs/release_manifest.json
```

The manifest lists repository-relative paths, byte sizes and SHA-256 values.
It excludes its own file to avoid self-reference. Regenerate it under a new name
after later changes, and review the complete directory before any commit.

## Publication state

The initial preparation completed without publishing, initializing Git or copying
private Git history into the clean directory. Subsequent publication targets the
existing repository and preserves its existing history; current hosting setup
is described in [deployment.md](deployment.md). Trained weights and UAX redistribution
are not cleared and those resources remain excluded. Full private experiment
reproduction requires legitimately obtained data, exact private artifacts and
protocol lineage. The public checkout alone cannot reproduce that evidence.

