# Data dictionary

## Analysis-ready grid

`data/grid/grid_1km.csv` is the released 1 km grid-level analytical table. Key fields include grid identifiers and centroids, population exposure, housing-price percentile, service-density percentile, opportunity-percentile measures for companion-animal, daily-life, health-care, elder-care, education, recreation and transit layers, and pet-retail/pet-medical subtype buffer measures.

## CCEM reference outputs

`data/ref/ccem/base_ccem.csv` and `spatial_ccem.csv` contain matched counterfactual coupling-excess outputs. Key fields are:

- `O`: observed high-high count.
- `E`: expected high-high count under within-stratum independence.
- `Delta`: observed minus expected count.
- `rho`: observed-to-expected ratio.
- `q97.5`: upper-tail permutation quantile.
- `M`: observed count minus `q97.5`.
- `R`: permutation draws.
- `seed`: random seed.

## Signature-formation outputs

`data/ref/sig/` contains model metrics, classwise metrics, feature importance and SHAP summaries for the bundled pet-service maturity class.

## Manuscript and supplementary tables

`data/tables/` and `data/supp/` provide machine-readable versions of the final article and supplementary tables. They are included to keep the reported table values inspectable in the repository and to prevent version drift between manuscript text and machine-readable materials.

## Supporting outputs

`data/support/` contains derived supporting summaries for POI-domain quality, coordinate handling, multiscale checks, spatial specificity, diagnostic typology, local atlas summaries, validation support and claim boundaries. These files are aggregate or derived materials; raw third-party records are not included.
