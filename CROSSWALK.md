# Data, code, figure and table crosswalk

## Main article

| Element | Machine-readable support | Reproduction or verification path |
|---|---|---|
| Table 1 | `data/tables/t1_ccem.csv`; `data/ref/ccem/base_ccem.csv` | recomputed by `code/scripts/make_tables.py`; checked in `run_summary.json` |
| Table 2 | `data/tables/t2_spatial.csv`; `data/ref/ccem/spatial_ccem.csv`; `data/support/spatial_ranks.csv` | recomputed by `code/scripts/make_tables.py`; checked in `run_summary.json` |
| Table 3 | `data/tables/t3_model.csv`; `data/ref/sig/model.csv` | recomputed by `code/scripts/make_model.py`; checked in `run_summary.json` |
| Table 4 | `data/tables/t4_classes.csv`; `data/ref/sig/labels.csv` | recomputed by `code/scripts/make_model.py`; checked in `run_summary.json` |
| Table 5 | `data/tables/t5_claims.csv`; `data/support/*_claims.csv` | verified as released manuscript table and supporting boundary summaries |
| Figure 1 | `figures/main/Figure_1.png` | reference conceptual figure |
| Figure 2 | `figures/main/Figure_2.png`; `data/ref/ccem/base_ccem.csv`; `data/ref/ccem/misclass.csv`; `data/ref/ccem/multiscale.csv` | underlying tables checked in `run_summary.json` |
| Figure 3 | `figures/main/Figure_3.png`; `data/ref/ccem/spatial_ccem.csv`; `data/support/spatial_ranks.csv` | underlying spatial tables checked in `run_summary.json` |
| Figure 4 | `figures/main/Figure_4.png`; `data/support/pet_layers.csv`; `data/support/pet_split.csv` | verified as derived reference figure and supporting summaries |
| Figure 5 | `figures/main/Figure_5.png`; `data/support/atlas_summary.csv`; `data/support/atlas_grids.csv`; `data/support/atlas_counts.csv` | verified as derived reference figure and supporting summaries |
| Figure 6 | `figures/main/Figure_6.png`; `figures/model/fig6_model.*`; `data/ref/sig/` | model-explanation panel recomputed by `code/scripts/make_model.py` |

## Supplementary material

| Element | Machine-readable support | Verification path |
|---|---|---|
| Tables S1-S7 | `data/supp/s01_*` through `s07_*`; `data/support/poi_rules.json`; `data/validation/` | checked by `code/scripts/check_inputs.py` and `verify.py` |
| Claim-boundary validation update | `data/supp/s_update.csv`; `data/validation/claims.csv` | checked as manual-validation support |
| Tables S8-S15b | `data/supp/s08_*` through `s15b_*`; `data/ref/ccem/`; `data/support/threshold.csv`; `data/support/multiscale.csv` | core CCEM outputs recomputed; derived multiscale outputs checked |
| Tables S16-S18 | `data/supp/s16_*` through `s18_*`; `data/support/spatial_ranks.csv`; `data/support/pet_layers.csv`; `data/support/pet_split.csv` | checked as derived supporting outputs |
| Tables S19-S24 | `data/supp/s19_*` through `s24_*`; `data/support/sig_profiles.csv`; `data/support/atlas_summary.csv`; `data/ref/sig/` | signature model outputs recomputed; local atlas summaries checked |
| Supplementary claim-boundary summary | `data/supp/s_claims.csv`; `data/support/*_claims.csv` | checked as boundary support |
| Figures S1-S3 | `figures/supp/Figure_S1.png`; `Figure_S2.png`; `Figure_S3.png` | verified as reference figures |

## Public-release boundary

The repository provides released analysis-ready data, aggregate reference outputs, manuscript tables, supplementary tables, validation summaries and reference figures. Raw third-party POI, housing, platform and address-level records are not redistributed. The reproducibility scripts therefore rebuild the core aggregate analyses from released grid data and verify derived materials that support the manuscript and supplementary package.
