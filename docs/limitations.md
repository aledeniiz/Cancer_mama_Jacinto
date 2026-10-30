# What the evidence supports

The project demonstrates implementation of a compact CNN from scratch,
patient-level image aggregation, clinical preprocessing, two multimodal fusion
strategies, reproducible evaluation and a privacy-safe aggregate web explorer.
It does not demonstrate clinical utility.

- CNN discrimination is close to chance under the matched OOF protocol.
- Clinical-only has the highest observed OOF ROC-AUC. Fusion has not demonstrated
  an improvement over the clinical baseline.
- Validation is exploratory and internal; development data were reused across
  previous experiments. The confidence intervals are conditional on selected
  models, not a correction for selection or training variability.
- Five unknown historical outcomes were excluded from clinical/fusion analyses.
  Historical fold CNN training had included those labels; exclusion does not
  retroactively remove their influence from frozen image embeddings.
- Cohort differences, endpoint harmonization, missing clinical variables and
  incomplete temporal provenance limit generalization and interpretation.
- The historical TEST was used previously with other models. It is separate from
  matched OOF but does not provide pristine external confirmation.
- The final development-fitted multimodal checkpoint has not been independently
  validated. A new prospective or genuinely independent patient cohort is needed.
- Preserved weights and reviewed public-case references permit metric checks and
  replay of saved embedding stages. This does not rerun the historical training
  or independently validate image extraction; pixel inference needs the MRIs.
- The interface is an aggregate-results explorer, not a live medical predictor.

Academic and educational research only. No diagnosis, treatment recommendation,
patient triage or claim of clinical authorization is provided.

