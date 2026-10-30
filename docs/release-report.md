# Initial public package preparation report

## Included

Curated compact CNN, EMA/BatchNorm utilities, image and clinical preprocessing,
neural/linear fusion implementations, resource-gated preparation/training/
evaluation/inference commands, portable configuration, 16 software tests,
aggregate scientific results/figures, English research documentation and a
controlled copy of the finished Spanish interactive platform. The web is a
privacy-reviewed export of the existing design, not a redesigned replacement.

## Excluded

Medical images, image caches, clinical tables, labels and identifiers,
patient-level predictions and splits, all trained weights and fitted
normalizers, historical Git data, virtual environments, bytecode/caches, private
reports/logs, credentials, backups, publisher figures and UAX presentations/
branding. The original working project and verified backups were preserved.

## Evidence and constraints

Fresh-environment and relocated-checkout tests passed. Aggregate values retain
the original OOF AUCs 0.5197 / 0.6450 / 0.6326 / 0.6355. Historical EMA TEST
AUC 0.5524 stays under its separate 176-patient protocol. No metric or
architecture was changed. The final multimodal head has no independent
performance estimate, and the five clinical OOF models are not a global final
model. Public source contains loading code but no private model parameters.

The public web retains the original appearance and educational interactions.
Private patient exploration and inference controls are explicitly unavailable.
See [verification](verification.md) and the [web export contract](../web/PUBLIC_EXPORT.md).

The existing CC BY 4.0 source license and attribution are retained. Data have
separate CC BY-NC 4.0/custodian terms. No weight or UAX asset redistribution
permission is assumed. Review ownership/attribution and the exact release
manifest before a future publication.

## Proposed commit, not executed

```text
Prepare privacy-reviewed DCE-MRI research source and faithful web export

Include portable compact CNN and multimodal code, tests and aggregate evidence.
Preserve the finished platform while withholding private data and model assets.
Document protocol differences, reproducibility limits and redistribution terms.
```

At the initial preparation checkpoint no repository, remote, commit, push or
deployment had been created. Subsequent publication configuration is documented
in [deployment.md](deployment.md). Directory counts, sizes and hashes are
available in the release manifest and local scan output.
