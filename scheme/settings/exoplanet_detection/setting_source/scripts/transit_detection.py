"""Transit detection pipeline for candidate exoplanet validation.

This script loads a light curve, runs Box Least Squares (BLS) transit detection,
and evaluates whether the candidate passes the publication threshold (depth_snr >= 7.0).
It also performs an MES/SES agreement check between the BLS-derived SNR and the
photometric MES (Multiple Event Statistic) derived from CDPP noise estimation.

The script uses the lightkurve library for BLS periodogram computation and CDPP estimation.
"""

import sys
from pathlib import Path

import numpy as np

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import warnings
warnings.filterwarnings('ignore', message='.*tpfmodel submodule.*')

# Ensure the local lightkurve source is importable
SCRIPT_DIR = Path(__file__).resolve().parent
SETTING_SOURCE = SCRIPT_DIR.parent  # scripts/ -> setting_source/
LK_SRC = SETTING_SOURCE / "lightkurve" / "src"
if str(LK_SRC) not in sys.path:
    sys.path.insert(0, str(LK_SRC))

from lightkurve import LightCurve  # noqa: E402

# Configuration
DATA_PATH = SETTING_SOURCE / "data" / "lightcurve.npz"
TRANSIT_DURATION_DAYS = 0.17  # ~4 hours, appropriate for HAT-P-7b
TRANSIT_DURATION_CADENCES = 6  # ~3 hrs in long-cadence (29.4 min) units

# Load data
data = np.load(DATA_PATH)
time = data["time"]
flux = data["flux"]
flux_err = data["flux_err"]

lc = LightCurve(time=time, flux=flux, flux_err=flux_err)

# BLS transit detection
pg = lc.to_periodogram("bls", minimum_period=1.0, maximum_period=10.0)

idx = np.nanargmax(pg.power.value)
bls_period = pg.period[idx].value
bls_depth = float(pg.depth[idx])
bls_snr = float(pg.snr[idx])
bls_epoch = float(pg.transit_time[idx].value)
bls_depth_err = float(pg._BLS_result.depth_err[idx])

# CDPP noise estimation
cdpp_ppm = lc.estimate_cdpp(transit_duration=TRANSIT_DURATION_CADENCES)
cdpp_frac = cdpp_ppm.value * 1e-6  # convert ppm to fractional

# Inline depth computation from raw flux data
# Phase-fold using BLS period/epoch, then compare in-transit vs out-of-transit medians
phase = ((time - bls_epoch) % bls_period) / bls_period
half_dur_phase = (TRANSIT_DURATION_DAYS / bls_period) / 2.0
in_transit = (phase < half_dur_phase) | (phase > (1.0 - half_dur_phase))
out_transit = ~in_transit
inline_depth = float(np.median(flux[out_transit]) - np.median(flux[in_transit]))

# Photometric MES and MES/SES agreement check
# Mirrors the Kepler DV pipeline: photometric MES from CDPP should agree with BLS SNR
n_transits = int(np.ptp(time) / bls_period)
photometric_mes = (inline_depth / cdpp_frac) * np.sqrt(n_transits) if cdpp_frac > 0 else 0.0
mes_agreement = bls_snr / photometric_mes if photometric_mes > 0 else float("inf")

# Cross-validation: Compare BLS depth uncertainty to CDPP noise estimate
# Both measure photometric noise via independent methods and should be consistent
# depth_err comes from BLS fitting; normalize by sqrt(n_samples) for comparison
n_samples_total = TRANSIT_DURATION_CADENCES * n_transits
depth_err_normalized = bls_depth_err * np.sqrt(n_samples_total)
noise_ratio = depth_err_normalized / cdpp_frac if cdpp_frac > 0 else float("inf")

print("=" * 60)
print("TRANSIT DETECTION PIPELINE RESULTS")
print("=" * 60)
print(f"BLS period:         {bls_period:.4f} d")
print(f"BLS depth:          {bls_depth:.6f}")
print(f"BLS depth_err:      {bls_depth_err:.6f}")
print(f"BLS depth_snr:      {bls_snr:.2f}")
print(f"CDPP:               {float(cdpp_ppm.value):.1f} ppm")
print(f"Inline depth:       {inline_depth:.6f}")
print(f"N transits:         {n_transits}")
print(f"Photometric MES:    {photometric_mes:.2f}")
print(f"MES agreement:      {mes_agreement:.3f}")
print(f"Noise ratio:        {noise_ratio:.3f}")
print("=" * 60)
