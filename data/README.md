# NeurMOC reconstruction, version 1.0

These files contain the NeurMOC reconstruction for April 2003–December 2024,
based on JPL GRACE/GRACE-FO ocean bottom pressure, DUACS sea surface height,
and CCMP zonal wind.

- `NeurMOC_data.nc` is the primary dataset, with variable descriptions and
  units in its metadata. It contains the monthly latitude–density MOC anomaly
  (`moc`, Sv) and its one-sigma uncertainty (`moc_uncertainty`), the linear
  trend (`trend`, Sv yr⁻¹) and its one-sigma uncertainty
  (`trend_uncertainty_total`), and trend significance under the pointwise
  criterion (`trend_significant`) and with false discovery rate control
  (`trend_significant_fdr`).
- `NeurMOC_data.mat` is a MATLAB-format copy of the same fields; its `README`
  variable lists them.

The reconstructed values are **anomalies**: at each latitude–density point,
the mean over April 2003–December 2024 has been removed. `moc_baseline` is the
ACCESS-ESM1.5 2004–2009 mean state, the reference of the network's anomalies,
used to locate the cell cores and to set the sign convention of the abyssal
cells; it should not be interpreted as an observed mean.
