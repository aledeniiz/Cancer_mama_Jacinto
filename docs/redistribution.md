# Redistribution decisions

| Component | Public package decision |
|---|---|
| Curated source, configurations and tests | Included; local project declares CC BY 4.0; adaptations documented |
| Aggregate metrics, ROC/PR coordinates and BCE histories | Included without thresholds, identifiers or prediction rows |
| Newly drawn architecture diagrams | Included; no patient image or paper figure copied |
| Original datasets, image caches, clinical tables and labels | Excluded; medical-data privacy and separate custodian/license terms |
| Individual OOF/TEST scores and actual split identifiers | Excluded; only protocol definitions are described |
| CNN, EMA, clinical and multimodal trained weights/transforms | Excluded; redistribution/derived-data permissions not established |
| UAX template, logo and final presentation | Excluded; redistribution permission not established |
| Publisher figures/PDF | Excluded; no permission to adapt or republish assumed |
| Third-party fonts, private HTML, screenshots, logs, backups | Excluded; unnecessary or rights/private-content uncertainty |
| Original Git history | Excluded; historical commits may contain private tables |

Source software declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The source dataset [Zenodo record](https://zenodo.org/records/18114231) declares
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). These licenses
are not interchangeable. The publisher's paper also has separate terms; this
package links to it without reproducing its figures.

The local teaching guide permits publishing personal work while retaining
attribution and forbids private validation data/labels. That is not blanket
permission to redistribute datasets, institutional branding or learned weights.
The original project's license is preserved rather than replaced by MIT or a
new claim of unrestricted ownership. See [attributions](../NOTICE.md).

The initial directory preparation copied no Git history. Publication to the
user-selected existing repository preserves that repository's prior history;
removing a file from its current tree does not purge older commits. Review the
exact release manifest and ownership/attribution before each commit. A clean scan
is bounded evidence, not legal clearance or proof that every possible secret
format has been detected.

