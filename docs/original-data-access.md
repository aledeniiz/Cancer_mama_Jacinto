# Original teaching dataset: provenance and access

The original repository began as a teaching case based on **BreastDCEDL**.
That provenance is preserved even though its individual clinical tables,
illustrations and earlier helper scripts are not part of the curated public
edition. No local dataset or original backup is removed by publishing this edition.

The teaching guide and downloader are retained in the repository's earlier commit:

- [Original access guide](https://github.com/aledeniiz/Cancer_mama_Jacinto/blob/22aa191/GUIA.md)
- [Original downloader and source location](https://github.com/aledeniiz/Cancer_mama_Jacinto/blob/22aa191/descargar_datos.py)
- [BreastDCEDL upstream](https://github.com/naomifridman/BreastDCEDL)
- [Version-specific dataset record and conditions](https://zenodo.org/records/18114231)

Use those sources only with the permissions and license applicable to your use.
This project does not automatically download medical data or republish its
individual records. The current [data guide](datasets.md) explains the prepared
subset, image/patient units and authorized local input requirements.

The public release replaces the earlier helper workflow with documented,
resource-gated scripts for preparation, training, evaluation and inference.
Its datasets remain separate local resources. The original Git history is not
rewritten: removing a table from the current tree does not remove its past versions.
