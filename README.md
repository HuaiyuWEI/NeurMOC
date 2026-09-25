# NeurMOC

This repository contains the code and data accompanying the NeurMOC
manuscript, which reconstructs changes in the Atlantic and Southern Ocean
meridional overturning circulation (MOC) from satellite observations of ocean
bottom pressure, sea surface height, and wind, using a neural network trained
on climate-model simulations.

The code is provided to document the methodology. It depends on large
external datasets (CMIP6 simulations, satellite products, and in situ
observations) that are not included, and it is not packaged to run as a
standalone workflow.

The reconstruction can also be explored in the
[interactive viewer](https://huaiyuwei.github.io/neurmoc/).

## Where each part of the method is implemented

| Manuscript section (Materials and Methods) | Code |
|---|---|
| CMIP6 simulations and data preprocessing | `scripts/01_make_basin_masks.py` to `scripts/07_combine_experiments.py`, `neurmoc/cmip_io.py`, `neurmoc/grids.py`, `neurmoc/filtering.py` |
| DBNN architecture | `neurmoc/model.py` |
| DBNN training and sensitivity tests | `scripts/08_train_nn.py`, `neurmoc/training.py`, `neurmoc/naming.py` |
| Cross-model evaluation | `scripts/10_test_out_of_sample.py`, `neurmoc/evaluation.py`, `neurmoc/model_test_cases.py` |
| Satellite data | `scripts/12_prep_satellite_inputs.py` |
| RAPID, OSNAP, and ECCO comparison records | `scripts/13_prep_insitu_moc_obs.py`, `neurmoc/rapid.py` |
| Satellite-based reconstruction | `scripts/14_reconstruct_real_world.py`, `neurmoc/inference.py` |
| Monthly uncertainty | `scripts/15_compute_trend_budget.py`, `scripts/16_grace_noise_montecarlo.py`, `neurmoc/satellite_products.py` |
| Trend estimates and significance | `neurmoc/moc_utils.py`, `scripts/15_compute_trend_budget.py`, `scripts/18_combination_trends.py`, `scripts/21_trend_robustness.py` |
| Layer-wise relevance propagation | `neurmoc/lrp.py`, `neurmoc/lrp_targets.py`, `scripts/17_RealWorld_LRP.py`, `scripts/19_model_LRP.py` |

## Data

The reconstruction is provided as [NetCDF](data/NeurMOC_data.nc), with a
[MATLAB-format copy](data/NeurMOC_data.mat). See the [data notes](data/README.md)
for the contents. 
