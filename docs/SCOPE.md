# Reproducibility scope

## Fully recomputed from released data

The scripts in `code/scripts/` recompute the following outputs from `data/grid/grid_1km.csv`:

- base counterfactual coupling-excess model tables;
- spatially constrained coupling-excess tables and summaries;
- seed-sensitivity and matched-strata summaries;
- pet-retail classification-error sensitivity;
- signature-label assignment, model generalization metrics, class-wise metrics, feature importance, SHAP summary, SHAP sample and the model-explanation figure.

## Released as derived reference outputs

The repository includes derived, machine-readable reference outputs for multiscale sensitivity, POI-domain quality checks, coordinate handling, manual validation, local-atlas summaries and claim-boundary tables. These outputs support the supplementary material but are not rebuilt from raw source layers in this public package because the raw POI, housing and platform records are not redistributed.

## Not redistributed

Raw third-party POI records, raw housing transaction records, platform records, address-level local examples and any individual-level material are outside the public repository. The manuscript interprets all POI-derived measures as facility-opportunity proxies, not demand, service use, service quality, welfare, affordability or causal effects.

## Verification command

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python code/scripts/check_inputs.py
PYTHONDONTWRITEBYTECODE=1 python code/scripts/run_all.py
```

A successful run produces `reproducibility/output/run_summary.json` with `status` equal to `pass`.
