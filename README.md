# Beyond facility abundance

This repository supports the study "Beyond facility abundance: Companion-animal services as a diagnostic layer for all-age urban service assessment in Beijing". It provides the released data, code, reference outputs and figures needed to reproduce the aggregate analyses reported in the article and supplementary material.

## Repository layout

- `data/grid/`: released analysis-ready 1 km grid table used to rerun the core counterfactual and signature-formation analyses.
- `data/ref/`: reference outputs from the validated analysis run.
- `data/tables/`: machine-readable versions of the final manuscript tables.
- `data/supp/`: machine-readable versions of the supplementary tables.
- `data/support/`: derived summaries that support supplementary checks, validation boundaries, multiscale sensitivity and local-atlas interpretation.
- `data/validation/`: manual-validation summaries and claim-boundary checks.
- `figures/`: reference figure files and the model-explanation figure in PNG, SVG and PDF formats.
- `code/`: scripts for input checks, table reproduction, model reproduction, figure reproduction and output verification.
- `reproducibility/output/`: output folder written by `code/scripts/run_all.py`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 python code/scripts/check_inputs.py
PYTHONDONTWRITEBYTECODE=1 python code/scripts/run_all.py
```

A successful run writes `reproducibility/output/run_summary.json` with `status: pass`. On Windows, place the repository in a short path such as `C:\repo` before extracting or running the scripts.

## Reproducibility boundary

The repository redistributes derived, analysis-ready grid-level data and aggregate validation outputs. It does not redistribute raw third-party POI records, raw housing transaction records, platform records, or individual-level information. The supplied grid table is sufficient to reproduce the main CCEM tables, spatial robustness tables, signature-formation model, SHAP summary, SHAP sample table and model-explanation figure. Multiscale and validation materials are included as derived reference outputs where upstream construction depends on source layers that cannot be redistributed.

POI-derived measures should be interpreted as facility-opportunity proxies rather than demand, use, quality, affordability, welfare or causal effects.

## Expected verification

`code/scripts/run_all.py` performs the following checks:

- recomputes the base and spatial CCEM tables from the released grid table;
- recomputes the signature-formation models, feature importance, SHAP summary, SHAP sample and the model-explanation figure;
- compares recomputed outputs against reference outputs;
- checks manuscript tables, supplementary tables, supporting outputs, manual-validation summaries, reference figures and the repository manifest.
