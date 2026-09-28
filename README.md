[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22969710.svg)](https://doi.org/10.5281/zenodo.22969710)

# NeurMOC

NeurMOC reconstructs changes in the meridional overturning circulation (MOC) across the Atlantic and Southern Oceans from April 2003 to December 2024. It combines satellite-informed estimates of ocean bottom pressure, sea surface height, and zonal wind with a dual-branch neural network (DBNN) trained on climate-model simulations.

This repository provides the reconstruction and code documenting its development and evaluation. The code is not packaged as a standalone workflow: reproducing the analysis requires large external CMIP6, satellite, and in situ datasets that are not included here.

Explore the reconstruction in the [interactive viewer](https://huaiyuwei.github.io/neurmoc/).

## Reconstruction data

The reconstruction is available in [NetCDF](data/NeurMOC_data.nc), with a [MATLAB copy](data/NeurMOC_data.mat). It includes monthly MOC anomalies and uncertainty estimates, trends and trend uncertainty, and significance flags. See the [data notes](data/README.md) for variable definitions and interpretation.

## Code guide

| Task | Main code |
|---|---|
| Prepare climate-model data | `scripts/01_make_basin_masks.py` through `scripts/07_combine_experiments.py`; `neurmoc/cmip_io.py`, `neurmoc/grids.py`, `neurmoc/filtering.py` |
| Define and train the DBNN | `neurmoc/model.py`, `neurmoc/training.py`, `neurmoc/naming.py`; `scripts/08_train_nn.py` |
| Evaluate cross-model performance | `scripts/10_test_out_of_sample.py`; `neurmoc/evaluation.py`, `neurmoc/model_test_cases.py` |
| Prepare satellite inputs and comparison datasets | `scripts/12_prep_satellite_inputs.py`, `scripts/13_prep_insitu_moc_obs.py`; `neurmoc/rapid.py` |
| Reconstruct the observed period | `scripts/14_reconstruct_real_world.py`; `neurmoc/inference.py` |
| Estimate uncertainty | `scripts/15_compute_trend_budget.py`, `scripts/16_grace_noise_montecarlo.py`; `neurmoc/satellite_products.py` |
| Analyze trends and robustness | `scripts/18_combination_trends.py`, `scripts/21_trend_robustness.py`; `neurmoc/moc_utils.py` |
| Analyze input relevance | `scripts/17_RealWorld_LRP.py`, `scripts/19_model_LRP.py`; `neurmoc/lrp.py`, `neurmoc/lrp_targets.py` |


