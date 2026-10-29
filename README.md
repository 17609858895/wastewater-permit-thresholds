# Wastewater permit-threshold analysis

Core research code for **Effluent phosphorus at US wastewater treatment plants accumulates below permit limits and follows tightened limits**.

The primary reconstruction covers seven parameters in the C2 and C3 concentration fields of US publicly owned treatment works permits. It models rounded and nondetect reports as intervals, coarsens detected values in [0.8L, 2L), fits normal and logistic location-scale components with five permit folds, and combines expected near-limit exceedances with observed extreme exceedances. C2/C3 are administrative concentration fields, not universally monthly/daily averages.

## Setup and quick verification

Python 3.11 is the reference analysis runtime. Install the pinned scientific packages in a new environment:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The tests use synthetic inputs only. They check analytical likelihood gradients, interval precedence, the hybrid-count definition and permit grouping in a short fit. They do not reproduce or independently validate national estimates.

## Data

Download annual NPDES DMR ZIPs and ICIS permit/facility/inspection/single-event-violation CSVs from the [EPA ECHO download service](https://echo.epa.gov/tools/data-downloads) and consult the [DMR documentation](https://echo.epa.gov/tools/data-downloads/icis-npdes-dmr-and-limit-data).

Place files as follows (or set `ECHO_RAW_ROOT` to a directory with these two subdirectories):

```text
raw_data/
  annual_dmr/npdes_dmrs_fy2009.zip ... npdes_dmrs_fy2026.zip
  icis/ICIS_PERMITS.csv
  icis/ICIS_FACILITIES.csv
  icis/NPDES_INSPECTIONS.csv
  icis/NPDES_SE_VIOLATIONS.csv
```

The study's 2026 record was incomplete at retrieval. EPA files are revised over time; a fresh download may change sample counts. Raw data, record-level predictions, fitted model files, manuscript files and artwork are not redistributed here. The manuscript's supplementary tables report the numerical results of the study.

## Primary analysis

Run from the repository root. The extraction stage uses up to six processes, and full fitting requires substantial memory and runtime.

```sh
python scripts/01_extract_potw_dmr.py
python scripts/02_build_panel.py
python scripts/03_discontinuity_descriptives.py
python scripts/04_features.py
python scripts/00_extract_regulatory_tables.py
python scripts/run_primary.py
python scripts/15u_postprocess.py
```

`run_primary.py` runs the production driver for both fields of the seven primary parameters and TRC, which is excluded from the primary aggregate and retained for precision sensitivity. The driver saves mixture components and out-of-fold predictions; it selects mixture weights on their pooled interval log scores without an additional nested holdout. Legacy helper `05_fit_counterfactual.py` supplies bounds and feature definitions; its standalone driver is a different configuration and should not be substituted for `05u_fit_cliff_stack.py`.

## Additional analyses

| Purpose | Entry point | Required prior outputs |
|---|---|---|
| Threshold heterogeneity and precision-grid diagnostics | `03b_discontinuity_heterogeneity.py`, `10b_density_jump.py` | screened sample/features |
| Model-free limit changes | `07_limit_change_events.py` | screened sample |
| Unchanged-limit controls for limit changes | `07c_limit_change_controls.py` | screened sample |
| Semi-synthetic reconstruction and comparators | `08_semisynthetic_benchmark.py`, then `08b_semisynthetic_nearest.py` | BOD5 features |
| Repeated semi-synthetic recovery with the final mixture | `08c_semisynthetic_mixture_repeats.py` (`--job`, or `--queue`, then `--collect`) | BOD5 features |
| Separate normal-component simulation | `sim_crossfit_check.py` | none |
| Reported-label probabilities and permit rankings | `14u_label_bias_targeting.py` | features and production predictions |
| Adjusted regulatory targeting | `17u_adjusted_targeting.py` | production post-processing and regulatory tables |
| Model-based limit-change profiles | `18u_future_limit_test.py` | production predictions and features |
| Precision-spread estimates | `19u_latent_scale_checks.py` | production predictions and features |
| Historical gradients and example sequence | `20u_gradients_example.py` | production predictions and features |
| Phosphorus by permit-limit level | `21u_tp_limit_levels.py` | production predictions |

Run each listed entry point with `python scripts/<entry point>`. For window sensitivity, the production driver accepts `--kappa`, `--lam` and `--placebo`; the exact settings are described in the Supporting Information. Analytical scripts retain the original computation and output structure. Public-release changes concern raw-data paths, explanatory docstrings, the auxiliary-table extraction helper and a sequential entry point.

## Interpretation

The shortfall is a model-conditioned discrepancy, not a verified count of omitted or legally noncompliant discharges. It includes exceedances prevented by operation and is therefore an upper bound on exceedances missing from the record. Bootstrap intervals keep component fits, weights and windows fixed. `08_semisynthetic_benchmark.py` fits the normal component with three folds and one realization at each of four displacements; `08c_semisynthetic_mixture_repeats.py` repeats the recovery with the five-fold national mixture (two replicates at displacement probabilities 0 and 0.5, one at 0.25 and 0.75, and permit-clustered displacement). All simulations use BOD5 C2 series. Regulatory-record comparisons use full-period fitted distributions and are retrospective. `07c_limit_change_controls.py` compares model-free limit-change contrasts with up to 20 unchanged-limit series per event; the model-based profiles of `18u_future_limit_test.py` have no controls. See the manuscript and Supporting Information for definitions and limitations.

Contact: Chong Liu, chong.liu@auckland.ac.nz.
