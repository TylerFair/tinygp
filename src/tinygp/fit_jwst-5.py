import os
import sys
import glob
from functools import partial

# Add current directory to path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.stats import norm
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro_ext.distributions as distx
import numpyro_ext.optim as optimx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.7
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit
from exotic_ld import StellarLimbDarkening
from plotting import plot_map_fits, plot_map_residuals, plot_transmission_spectrum, plot_wavelength_offset_summary
import argparse
import yaml
import jaxopt
import arviz as az
from createdatacube import SpectroData, process_spectroscopy_data
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.gridspec as gridspec
from jaxoplanet.experimental import calc_poly_coeffs
import tinygp

# Import Modularized Models
from models.core import _to_f64, _tree_to_f64, compute_transit_model, get_I_power2
from models.trends import (
    spot_crossing, compute_lc_linear, compute_lc_quadratic, compute_lc_cubic,
    compute_lc_quartic, compute_lc_linear_discontinuity, compute_lc_explinear,
    compute_lc_spot, compute_lc_none
)
from models.gp import (
    compute_lc_gp_mean, compute_lc_linear_gp_mean, compute_lc_quadratic_gp_mean,
    compute_lc_explinear_gp_mean
)
from models.builder import create_whitelight_model, create_vectorized_model, COMPUTE_KERNELS

# ---------------------
# Utilities
# ---------------------
DTYPE = jnp.float64

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def jax_bin_lightcurve(time, flux, duration, points_per_transit=20):
    dt = duration / points_per_transit
    t_min = jnp.min(time)
    t_max = jnp.max(time)
    total_range = t_max - t_min
    num_bins = jnp.ceil(total_range / dt).astype(int) + 1
    bin_indices = jnp.clip(((time - t_min) / dt).astype(int), 0, num_bins - 1)
    flux_sums = jnp.zeros(num_bins)
    time_sums = jnp.zeros(num_bins)
    counts = jnp.zeros(num_bins)
    flux_sums = flux_sums.at[bin_indices].add(flux)
    time_sums = time_sums.at[bin_indices].add(time)
    counts = counts.at[bin_indices].add(1.0)
    binned_flux = jnp.where(counts > 0, flux_sums / counts, jnp.nan)
    binned_time = jnp.where(counts > 0, time_sums / counts, jnp.nan)
    return binned_time, binned_flux
    
def _noise_binning_stats(residuals, n_bins=30, max_bin=None):
    """
    residuals: array-like, shape (n_channels, n_times) or (n_times,)
    Returns: bins, median_sigma, p16, p84, expected_white_median, expected_white_per_channel
    """
    residuals = np.array(residuals)
    if residuals.ndim == 1:
        residuals = residuals[None, :]
    n_channels, n_times = residuals.shape
    if max_bin is None:
        max_bin = max(1, n_times // 4)
    bins = np.unique(np.round(np.logspace(0, np.log10(max_bin), n_bins)).astype(int))
    sigma_b_channels = np.full((n_channels, len(bins)), np.nan)
    for i in range(n_channels):
        r = residuals[i, :]
        for j, b in enumerate(bins):
            m = (n_times // b) * b
            if m < b:
                sigma_b_channels[i, j] = np.nan
                continue
            r_trunc = r[:m].reshape(-1, b)
            means = r_trunc.mean(axis=1)
            if means.size >= 2:
                sigma_b_channels[i, j] = np.std(means, ddof=0)
            else:
                sigma_b_channels[i, j] = np.nan

    sigma_med = np.nanmedian(sigma_b_channels, axis=0)
    sigma_16 = np.nanpercentile(sigma_b_channels, 16, axis=0)
    sigma_84 = np.nanpercentile(sigma_b_channels, 84, axis=0)
    sigma1_channels = np.nanstd(residuals, axis=1, ddof=0)
    sigma1_med = np.nanmedian(sigma1_channels)
    expected_white_per_channel = sigma1_channels[:, None] / np.sqrt(bins)
    expected_white_median = sigma1_med / np.sqrt(bins)
    return bins, sigma_med, sigma_16, sigma_84, expected_white_median, expected_white_per_channel

def plot_noise_binning(residuals, outpath, title=None, to_ppm=True, show_per_channel_expected=False):
    bins, sigma_med, sigma_16, sigma_84, expected_white_med, expected_white_pc = _noise_binning_stats(residuals)
    factor = 1e6 if to_ppm else 1.0
    plt.figure(figsize=(6,4))
    plt.loglog(bins, sigma_med * factor, 'k-o', label='Measured (median across channels)')
    plt.fill_between(bins, sigma_16 * factor, sigma_84 * factor, color='gray', alpha=0.3, label='16-84%')
    plt.loglog(bins, expected_white_med * factor, 'r--', label='White-noise expectation (σ₁/√N)')
    if show_per_channel_expected and expected_white_pc is not None:
        for ch in range(expected_white_pc.shape[0]):
            plt.loglog(bins, expected_white_pc[ch, :] * factor, color='r', alpha=0.12)
    plt.xlabel('Bin size (number of points)')
    ylabel = 'RMS (ppm)' if to_ppm else 'RMS'
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved noise-binning plot to {outpath}")

def get_samples(model, key, t, yerr, indiv_y, init_params, **model_kwargs):
    t = _to_f64(t)
    yerr = _to_f64(yerr)
    indiv_y = _to_f64(indiv_y)
    init_params = _tree_to_f64(init_params)
    model_kwargs = _tree_to_f64(model_kwargs)

    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(
            model,
            #dense_mass=True,
            regularize_mass_matrix=False,
            init_strategy=numpyro.infer.init_to_value(values=init_params),
            target_accept_prob=0.9
        ),
        num_warmup=1000,
        num_samples=1000,
        progress_bar=True,
        jit_model_args=True
    )
    mcmc.run(key, t, yerr, y=indiv_y, **model_kwargs)
    return mcmc.get_samples()

def compute_aic(n, residuals, k):
    rss = np.sum(np.square(residuals))
    rss = rss if rss > 1e-10 else 1e-10
    aic = 2*k + n * np.log(rss/n)
    return aic

def get_limb_darkening(sld, wavelengths, wavelength_err, instrument, order=None, ld_profile='quadratic'):
    if instrument == 'NIRSPEC/G395H':
        mode = "JWST_NIRSpec_G395H"
        wl_min, wl_max = 28700.0, 51700.0
    elif instrument == 'NIRSPEC/G235H':
        mode = "JWST_NIRSpec_G235H"
        wl_min, wl_max = 17000.0, 30600.0
    elif instrument == 'NIRSPEC/G140H':
        mode = "JWST_NIRSpec_G140H-f100"
        wl_min, wl_max = 10000.0, 18000.0
    elif instrument == 'NIRSPEC/G395M':
        mode = "JWST_NIRSpec_G395M"
        wl_min, wl_max = 28700.0, 51700.0
    elif instrument == 'NIRSPEC/PRISM':
        mode = "JWST_NIRSpec_Prism"
        wl_min, wl_max = 5000.0, 55000.0
    elif instrument == 'NIRISS/SOSS':
        mode = f"JWST_NIRISS_SOSSo{order}"
        wl_min, wl_max = 8300.0, 28100.0
    elif instrument == 'MIRI/LRS':
        mode = f'JWST_MIRI_LRS'
        wl_min, wl_max = 50000.0, 120000.0

    wavelengths = np.array(wavelengths)
    wavelength_err = np.array(wavelength_err) if hasattr(wavelength_err, '__len__') else wavelength_err

    U_mu = []

    if hasattr(wavelength_err, '__len__'):
        for i in range(len(wavelengths)):
            wl_angstrom = wavelengths[i] * 1e4
            err_angstrom = wavelength_err[i] * 1e4
            intended_min = wl_angstrom - err_angstrom
            intended_max = wl_angstrom + err_angstrom

            if intended_max > wl_max:
                if err_angstrom > 0:
                    range_min = max(wl_max - err_angstrom * 2, wl_min)
                    range_max = wl_max
                print(f"Using boundary range for {wavelengths[i]:.4f} μm: [{range_min:.1f}, {range_max:.1f}] Å")
            elif intended_min < wl_min:
                if err_angstrom > 0:
                    range_min = wl_min
                    range_max = min(wl_min + err_angstrom * 2, wl_max)
                print(f"Using boundary range for {wavelengths[i]:.4f} μm: [{range_min:.1f}, {range_max:.1f}] Å")
            else:
                range_min = intended_min
                range_max = intended_max

            if ld_profile == 'quadratic':
                U_mu.append(sld.compute_quadratic_ld_coeffs(
                    wavelength_range=[range_min, range_max],
                    mode=mode,
                    return_sigmas=False
                ))
            elif ld_profile == 'power2':
                U_mu.append(sld.compute_power2_ld_coeffs(
                    wavelength_range=[range_min, range_max],
                    mode=mode,
                    return_sigmas=False
                ))
            else:
                raise ValueError(f"Unknown ld_profile: {ld_profile}")
        U_mu = jnp.array(U_mu)
    else:
        wl_range_clipped = [max(min(wavelengths)*1e4, wl_min),
                           min(max(wavelengths)*1e4, wl_max)]
        if ld_profile == 'quadratic':
            U_mu = sld.compute_quadratic_ld_coeffs(
                wavelength_range=wl_range_clipped,
                mode=mode,
                return_sigmas=False
            )
        elif ld_profile == 'power2':
            U_mu = sld.compute_power2_ld_coeffs(
                wavelength_range=wl_range_clipped,
                mode=mode,
                return_sigmas=False
            )
        else:
            raise ValueError(f"Unknown ld_profile: {ld_profile}")
        U_mu = jnp.array(U_mu)
    return U_mu

def fit_polynomial(x, y, poly_orders):
    best_order = None
    best_aic = np.inf
    best_coeffs = None
    for deg in poly_orders:
        y_med = np.median(y, axis=0)
        coeffs = np.polyfit(x, y_med, deg)
        pred = np.polyval(coeffs, x)
        aic = compute_aic(len(x), y_med - pred, k=deg+1)
        if aic < best_aic:
            best_aic = aic
            best_order = deg
            best_coeffs = coeffs
    return best_coeffs, best_order, best_aic

def plot_poly_fit(x, y, coeffs, order, xlabel, ylabel, title, save_path):
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = np.polyval(coeffs, x_fit)
    y_med = np.median(y, axis=0)
    y_err = np.std(y, axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, y_med, yerr=y_err, fmt='o', ls='', ecolor='k', mfc='k', mec='k')
    ax.plot(x_fit, y_fit, '-', label=f'Poly fit (deg {order})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved polynomial fit plot to {save_path}")

def save_results(wavelengths,wavelength_err,  samples, csv_filename):
    depth_chain = samples['rors']**2
    depth_median = np.nanmedian(depth_chain, axis=0)
    depth_err = np.std(depth_chain, axis=0)
    if depth_median.ndim == 1:
        depth_median = depth_median[:, np.newaxis]
        depth_err = depth_err[:, np.newaxis]
    n_planets = depth_median.shape[1]
    header_cols = ["wavelength"]
    header_cols = header_cols.append("wavelength_err")
    for i in range(n_planets):
        header_cols.append(f"depth{i:02d}")
        header_cols.append(f"depth_err{i:02d}")
    header = ",".join(header_cols)
    output_cols = [wavelengths]
    output_cols = [wavelength_err]
    for i in range(n_planets):
        output_cols.append(depth_median[:, i])
        output_cols.append(depth_err[:, i])
    output_data = np.column_stack(output_cols)
    np.savetxt(csv_filename, output_data, delimiter=",", header=header, comments="")
    print(f"Transmission spectroscopy data saved to {csv_filename}")

def save_detailed_fit_results(time, flux, flux_err, wavelengths, wavelengths_err, samples, map_params,
                               transit_params, detrend_type, output_prefix,
                               total_error_fit=None, gp_trend=None, spot_trend=None, jump_trend=None):
    n_wavelengths = len(wavelengths)
    n_times = len(time)
    print(f"Saving detailed fit results to {output_prefix}_*.csv")
    param_rows = []
    for i in range(n_wavelengths):
        row = {
            'wavelength': wavelengths[i],
            'wavelength_err': wavelengths_err[i],
            'rors': np.nanmedian(samples['rors'][:, i]),
            'rors_err': np.std(samples['rors'][:, i]),
            'depth': np.nanmedian(samples['rors'][:, i]**2),
            'depth_err': np.std(samples['rors'][:, i]**2),
            'u1': np.nanmedian(samples['u'][:, i, 0]),
            'u1_err': np.std(samples['u'][:, i, 0]),
            'u2': np.nanmedian(samples['u'][:, i, 1]),
            'u2_err': np.std(samples['u'][:, i, 1]),
        }

        if 'c1' in samples:
            row['c1'] = np.nanmedian(samples['c1'][:, i])
            row['c1_err'] = np.std(samples['c1'][:, i])
            # Override u1/u2 with physical parameters for power2 profile
            row['u1'] = row['c1']
            row['u1_err'] = row['c1_err']

        if 'c2' in samples:
            row['c2'] = np.nanmedian(samples['c2'][:, i])
            row['c2_err'] = np.std(samples['c2'][:, i])
            # Override u1/u2 with physical parameters for power2 profile
            row['u2'] = row['c2']
            row['u2_err'] = row['c2_err']

        if detrend_type != 'none':
            row['c'] = np.nanmedian(samples['c'][:, i])
            row['c_err'] = np.std(samples['c'][:, i])
            # v is not present in pure gp_spectroscopic
            if 'v' in samples:
                row['v'] = np.nanmedian(samples['v'][:, i])
                row['v_err'] = np.std(samples['v'][:, i])
        
        # Add other potential parametric terms
        if 'v2' in samples:
            row['v2'] = np.nanmedian(samples['v2'][:, i])
            row['v2_err'] = np.std(samples['v2'][:, i])
        if 'v3' in samples:
            row['v3'] = np.nanmedian(samples['v3'][:, i])
            row['v3_err'] = np.std(samples['v3'][:, i])
        if 'v4' in samples:
            row['v4'] = np.nanmedian(samples['v4'][:, i])
            row['v4_err'] = np.std(samples['v4'][:, i])
        if 'A' in samples:
            row['A'] = np.nanmedian(samples['A'][:, i])
            row['A_err'] = np.std(samples['A'][:, i])
        if 'tau' in samples:
            row['tau'] = np.nanmedian(samples['tau'][:, i])
            row['tau_err'] = np.std(samples['tau'][:, i])
        if 't_jump' in samples:
            row['t_jump'] = np.nanmedian(samples['t_jump'][:, i])
            row['t_jump_err'] = np.std(samples['t_jump'][:, i])
        if 'jump' in samples:
            row['jump'] = np.nanmedian(samples['jump'][:, i])
            row['jump_err'] = np.std(samples['jump'][:, i])
        if 'spot_amp' in samples:
            row['spot_amp'] = np.nanmedian(samples['spot_amp'][:, i])
            row['spot_amp_err'] = np.std(samples['spot_amp'][:, i])
            row['spot_mu'] = np.nanmedian(samples['spot_mu'][:, i])
            row['spot_mu_err'] = np.std(samples['spot_mu'][:, i])
            row['spot_sigma'] = np.nanmedian(samples['spot_sigma'][:, i])
            row['spot_sigma_err'] = np.std(samples['spot_sigma'][:, i])

        # Spectroscopic template amplitudes
        if 'A_gp' in samples:
            row['A_gp'] = np.nanmedian(samples['A_gp'][:, i])
            row['A_gp_err'] = np.std(samples['A_gp'][:, i])
        if 'A_spot' in samples:
            row['A_spot'] = np.nanmedian(samples['A_spot'][:, i])
            row['A_spot_err'] = np.std(samples['A_spot'][:, i])
        if 'A_jump' in samples:
            row['A_jump'] = np.nanmedian(samples['A_jump'][:, i])
            row['A_jump_err'] = np.std(samples['A_jump'][:, i])
            
        param_rows.append(row)

    params_df = pd.DataFrame(param_rows)
    params_df.to_csv(f"{output_prefix}_bestfit_params.csv", index=False)

    lc_components = []
    for i in range(n_wavelengths):
        rors_i_all_planets = np.atleast_1d(map_params['rors'][i])
        u_i = np.asarray(map_params['u'][i])
        total_model_flux = np.zeros_like(time)
        periods = np.atleast_1d(transit_params["period"])
        durations = np.atleast_1d(map_params["duration"])
        bs = np.atleast_1d(map_params["b"])
        t0s = np.atleast_1d(map_params["t0"])
        num_planets = len(periods)

        # Replaced _compute_transit_model logic with compute_transit_model from models.core
        # But here we need to loop manually because we are reconstructing flux for plotting
        # and map_params here are numpy arrays, not JAX arrays (from MCMC samples).
        # We can use limb_dark_light_curve from jaxoplanet or reimplement.
        # compute_transit_model is JIT-ed and expects JAX arrays.
        # We can call it, but we need to structure the params dict correctly.

        # Construct params for compute_transit_model
        params_for_transit = {
            'period': periods,
            'duration': durations,
            't0': t0s,
            'b': bs,
            'rors': rors_i_all_planets,
            'u': u_i
        }
        # We need to cast to jnp array
        params_for_transit = jax.tree_util.tree_map(jnp.array, params_for_transit)
        transit_model = compute_transit_model(params_for_transit, jnp.array(time))
        # Convert back to numpy
        transit_model = np.array(transit_model)

        
        # --- Reconstruct the trend based on detrend_type and map_params ---
        t_shift = time - np.min(time)
        trend = np.ones_like(time) # Default for 'none' case

        if detrend_type != 'none':
            c_i = map_params['c'][i]
            
            if 'gp_spectroscopic' in detrend_type:
                trend_parametric = c_i
                if 'linear' in detrend_type and 'v' in map_params:
                    trend_parametric += map_params['v'][i] * t_shift
                if 'quadratic' in detrend_type and 'v2' in map_params:
                    trend_parametric += map_params['v2'][i] * t_shift**2
                trend = trend_parametric + map_params['A_gp'][i] * gp_trend
            
            elif detrend_type == 'spot_spectroscopic':
                trend = c_i + map_params['A_spot'][i] * spot_trend
            
            elif detrend_type == 'linear_discontinuity_spectroscopic':
                trend = c_i + map_params['A_jump'][i] * jump_trend

            else: # Purely parametric models
                v_i = map_params.get('v', [0.0]*n_wavelengths)[i]
                if detrend_type == 'linear':
                    trend = c_i + v_i * t_shift
                elif detrend_type == 'quadratic':
                    trend = c_i + v_i * t_shift + map_params['v2'][i] * t_shift**2
                elif detrend_type == 'cubic':
                    trend = c_i + v_i * t_shift + map_params['v2'][i] * t_shift**2 + map_params['v3'][i] * t_shift**3
                elif detrend_type == 'quartic':
                    trend = c_i + v_i * t_shift + map_params['v2'][i] * t_shift**2 + map_params['v3'][i] * t_shift**3 + map_params['v4'][i] * t_shift**4
                elif detrend_type == 'explinear':
                    trend = c_i + v_i * t_shift + map_params['A'][i] * np.exp(-t_shift / map_params['tau'][i])
                elif detrend_type == 'linear_discontinuity':
                    jump_term = np.where(time > map_params['t_jump'][i], map_params['jump'][i], 0.0)
                    trend = c_i + v_i * t_shift + jump_term
                elif detrend_type == 'spot':
                    spot_term = spot_crossing(time, map_params['spot_amp'][i], map_params['spot_mu'][i], map_params['spot_sigma'][i])
                    trend = c_i + v_i * t_shift + spot_term
        
        full_model = transit_model + trend
        detrended_flux = flux[i] - trend

        if total_error_fit is not None:
            total_error_fit_broadcast = np.broadcast_to(total_error_fit[:, np.newaxis], (n_wavelengths, n_times))
        else:
            total_error_fit_broadcast = flux_err

        for j in range(n_times):
            lc_components.append({
                'wavelength': wavelengths[i],
                'wavelength_err': wavelengths_err[i],
                'time': time[j],
                'flux_raw': flux[i, j],
                'flux_err_raw': flux_err[i, j],
                'flux_err_fit': total_error_fit_broadcast[i, j],
                'transit_model': transit_model[j],
                'trend': trend[j],
                'full_model': full_model[j],
                'detrended_flux': detrended_flux[j],
                'residual': flux[i, j] - full_model[j]
            })

    lc_df = pd.DataFrame(lc_components)
    lc_df.to_csv(f"{output_prefix}_lightcurves.csv", index=False)
    return params_df, lc_df

def get_robust_sigma(x):
    """Helper to calculate sigma using MAD (robust to outliers)."""
    mad = np.median(np.abs(x - np.median(x)))
    return 1.4826 * mad

def calculate_beta_metrics(residuals, dt, cut_factor=5.0):
    residuals = np.array(residuals)
    ndata = len(residuals)
    
    # 1. Base White Noise Level (Unbinned)
    # We use this SINGLE value to anchor the theoretical curve
    sigma1 = get_robust_sigma(residuals)
    
    # Time stuff
    cadence_min = dt / 60.0
    
    # Determine Bin Sizes
    max_bin_n = ndata // int(cut_factor) 
    
    # Create unique bin sizes (in points)
    bin_sizes_points = np.unique(np.logspace(0, np.log10(max_bin_n), 300).astype(int))
    
    measured_rms = []
    expected_rms = []
    bin_sizes_min = []
    betas = []
    
    for N in bin_sizes_points:
        # Calculate Binned RMS
        cutoff = ndata - (ndata % N)

        # Binning
        binned_res = residuals[:cutoff].reshape(-1, N).mean(axis=1)
        
        sigma_N_measured = get_robust_sigma(binned_res)
        
        sigma_N_theory = sigma1 / np.sqrt(N)
        
        measured_rms.append(sigma_N_measured)
        expected_rms.append(sigma_N_theory)
        bin_sizes_min.append(N * cadence_min)
        
        betas.append(sigma_N_measured / sigma_N_theory)

    beta_final = np.median(betas)
    
    return beta_final, np.array(bin_sizes_min), np.array(measured_rms), np.array(expected_rms)

def run_beta_monte_carlo(residuals, dt, n_sims=500):
    ndata = len(residuals)
    
    # Estimate the noise from real data to create synthetic data
    sigma1 = get_robust_sigma(residuals)
    
    mc_betas = []
    
    # Get reference bin sizes from the real calculation so arrays match
    _, ref_bin_sizes, _, _ = calculate_beta_metrics(residuals, dt)
    all_sim_rms = np.zeros((n_sims, len(ref_bin_sizes)))
    
    for i in range(n_sims):
        # Generate synthetic white noise using the SAME sigma1
        synth_res = np.random.normal(0, sigma1, ndata)
        
        # Calculate metrics for this synthetic realization
        b_sim, _, rms_sim, _ = calculate_beta_metrics(synth_res, dt)
        
        mc_betas.append(b_sim)
        
        if len(rms_sim) == len(ref_bin_sizes):
            all_sim_rms[i, :] = rms_sim
            
    rms_low_1sig = np.percentile(all_sim_rms, 16, axis=0)
    rms_high_1sig = np.percentile(all_sim_rms, 84, axis=0)
    rms_low_2sig = np.percentile(all_sim_rms, 5, axis=0)
    rms_high_2sig = np.percentile(all_sim_rms, 95, axis=0)

    return np.array(mc_betas), rms_low_1sig, rms_high_1sig, rms_low_2sig, rms_high_2sig
# ---------------------
# Main Analysis
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="Run transit analysis with YAML config.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    instrument = cfg['instrument']
    if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'NIRSPEC/G140H', 'NIRSPEC/G235H']:
        nrs = cfg['nrs']
    elif instrument == 'NIRISS/SOSS':
        order = cfg['order']
    
    planet_cfg = cfg['planet']
    stellar_cfg = cfg['stellar']
    flags = cfg.get('flags', {})
    resolution = cfg.get('resolution', None)
    pixels = cfg.get('pixels', None)
    
    if resolution is None:
        if pixels is None: raise ValueError('Must Specify Resolutions or Pixels')
        bins = pixels
        high_resolution_bins = bins.get('high', None)
        low_resolution_bins = bins.get('low', None)
    elif pixels is None:
        bins = resolution
        high_resolution_bins = bins.get('high', None)
        low_resolution_bins = bins.get('low', None)

    outlier_clip = cfg.get('outlier_clip', {})
    planet_str = planet_cfg['name']
    mask_integrations_start = outlier_clip.get('mask_integrations_start', None)
    mask_integrations_end = outlier_clip.get('mask_integrations_end', None)

    base_path = cfg.get('path', '.')
    input_dir = os.path.join(base_path, cfg.get('input_dir', planet_str + '_NIRSPEC'))
    output_dir = os.path.join(base_path, cfg.get('output_dir', planet_str + '_RESULTS'))
    fits_file = os.path.join(input_dir, cfg.get('fits_file'))
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    host_device = cfg.get('host_device', 'gpu').lower()
    numpyro.set_platform(host_device)
    key_master = jax.random.PRNGKey(555)

    detrending_type = flags.get('detrending_type', 'linear')
    interpolate_trend = flags.get('interpolate_trend', False)
    interpolate_ld = flags.get('interpolate_ld', False)
    fix_ld = flags.get('fix_ld', False)
    need_lowres = flags.get('need_lowres', True)
    mask_start = flags.get('mask_start', False)
    mask_end = flags.get('mask_end', False)
    spot_amp = flags.get('spot_amp', 0.0)
    spot_mu = flags.get('spot_center', 0.0)
    spot_sigma = flags.get('spot_width', 0.0)
    save_trace = flags.get('save_whitelight_trace', False)

    # --- New LD Profile Flag ---
    ld_profile = flags.get('ld_profile', 'quadratic') # default to quadratic if not specified

    whitelight_sigma = outlier_clip.get('whitelight_sigma', 4)
    spectroscopic_sigma = outlier_clip.get('spectroscopic_sigma', 4)

    periods = jnp.atleast_1d(planet_cfg['period'])
    n_planets = len(periods)
    durations = jnp.atleast_1d(planet_cfg['duration'])
    t0s = jnp.atleast_1d(planet_cfg['t0'])
    bs = jnp.atleast_1d(planet_cfg['b'])
    rors = jnp.atleast_1d(planet_cfg['rprs'])
    depths = rors**2

    PERIOD_FIXED = periods
    PRIOR_DUR = durations
    PRIOR_T0 = t0s
    PRIOR_B = bs
    PRIOR_RPRS = rors
    PRIOR_DEPTH = depths

    stellar_feh = stellar_cfg['feh']
    stellar_teff = stellar_cfg['teff']
    stellar_logg = stellar_cfg['logg']
    ld_model = stellar_cfg.get('ld_model', 'mps1')
    ld_data_path = stellar_cfg.get('ld_data_path', '../exotic_ld_data')
    sld = StellarLimbDarkening(
        M_H=stellar_feh, Teff=stellar_teff, logg=stellar_logg, ld_model=ld_model,
        ld_data_path=ld_data_path
    )

    if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'NIRSPEC/G140H', 'NIRSPEC/G235H']:
        mini_instrument = f'nrs{nrs}'
    elif instrument == 'NIRISS/SOSS':
        mini_instrument = f'order{order}'
    else:
        mini_instrument = ''

    instrument_full_str = f"{planet_str}_{instrument.replace('/', '_')}_{mini_instrument}"
    if bins == resolution:
        spectro_data_file = output_dir + f'/{instrument_full_str}_spectroscopy_data_{low_resolution_bins}LR_{high_resolution_bins}HR.pkl'
    elif bins == pixels:
        spectro_data_file = output_dir + f'/{instrument_full_str}_spectroscopy_data_{low_resolution_bins}pix_{high_resolution_bins}pix.pkl'

    lr_bin_str = f'R{low_resolution_bins}' if bins == resolution else f'pix{low_resolution_bins}'
    hr_bin_str = f'R{high_resolution_bins}' if bins == resolution else f'pix{high_resolution_bins}'

    if not os.path.exists(spectro_data_file) or mask_start is not False:
        data = process_spectroscopy_data(instrument, input_dir, output_dir, planet_str, cfg, fits_file, mask_start, mask_end, mask_integrations_start, mask_integrations_end)
        data.save(spectro_data_file)
    else:
        data = SpectroData.load(spectro_data_file)
    
    print("Data loaded.")
    print(f"Time: {data.time.shape}")
     
    stringcheck = os.path.exists(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy')

    if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'MIRI/LRS', 'NIRSPEC/G140H', 'NIRSPEC/G235H']:
        U_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned,0.0, instrument, ld_profile=ld_profile)
    elif instrument == 'NIRISS/SOSS':
        U_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned, 0.0, instrument, order=order, ld_profile=ld_profile)
    #print(U_mu_wl)
    #print(get_limb_darkening(sld, data.wavelengths_lr, data.wavelengths_err_lr, instrument, order=order, ld_profile=ld_profile))
    #print(get_limb_darkening(sld, data.wavelengths_hr, data.wavelengths_err_hr, instrument, order=order, ld_profile=ld_profile))
    #exit()
    # ----------------------------------------------------
    # WHITELIGHT FITTING
    # ----------------------------------------------------
    if not stringcheck or ('gp' in detrending_type):
        if not os.path.exists(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv'):
            plt.scatter(data.wl_time, data.wl_flux)
            plt.savefig('stuff.png')
            plt.close()
            print('Fitting whitelight for outliers and bestfit parameters')
            hyper_params_wl = {
                "duration": PRIOR_DUR,
                "t0": PRIOR_T0,
                'period': PERIOD_FIXED,
            }
            if 'spot' in detrending_type:
                hyper_params_wl['spot_guess'] = spot_mu

            hyper_params_wl['u'] = U_mu_wl

            init_params_wl = {
                'c': 1.0,
                'v': 0.0,
                'log_jitter': jnp.log(1e-4),
                'b': PRIOR_B,
                'rors': PRIOR_RPRS
            }
            init_params_wl['u'] = U_mu_wl

            for i in range(n_planets):
                init_params_wl[f'logD_{i}'] = jnp.log(PRIOR_DUR[i])
                init_params_wl[f't0_{i}'] = PRIOR_T0[i]
                init_params_wl[f'_b_{i}'] = PRIOR_B[i]
                init_params_wl[f'depths_{i}'] = PRIOR_DEPTH[i]

            if 'quadratic' in detrending_type:
                init_params_wl['v2'] = 0.0
            if 'cubic' in detrending_type:
                init_params_wl['v2'] = 0.0; init_params_wl['v3'] = 0.0
            if 'quartic' in detrending_type:
                init_params_wl['v2'] = 0.0; init_params_wl['v3'] = 0.0; init_params_wl['v4'] = 0.0
            if 'explinear' in detrending_type:
                init_params_wl['A'] = 0.001; init_params_wl['tau'] = 0.5
            if 'gp' in detrending_type:
                init_params_wl['GP_log_sigma'] = jnp.log(jnp.nanmedian(data.wl_flux_err))
                init_params_wl['GP_log_rho'] = jnp.log(1)
            if 'linear_discontinuity' in detrending_type:
                init_params_wl['t_jump'] = 59791.12
                init_params_wl['jump'] = -0.001 

            whitelight_model_for_run = create_whitelight_model(detrend_type=detrending_type, n_planets=n_planets, ld_profile=ld_profile)
            
            if 'gp' in detrending_type:
                print("--- Running Pre-Fit with Linear Detrending to stabilize GP ---")
                whitelight_model_prefit = create_whitelight_model(detrend_type='linear', n_planets=n_planets, ld_profile=ld_profile)
                init_params_prefit = init_params_wl.copy()
                init_params_prefit.pop('GP_log_sigma', None)
                init_params_prefit.pop('GP_log_rho', None)
                soln_prefit = optimx.optimize(whitelight_model_prefit, start=init_params_prefit)(
                    key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl
                )
                print("--- Initializing GP with Pre-Fit Parameters ---")
                for k in soln_prefit.keys():
                    if k in init_params_wl: init_params_wl[k] = soln_prefit[k]
                
                print("Please make sure config is CPU for GP whitelight fit!")
                init_params_wl['GP_log_sigma'] = jnp.log(5.0 * jnp.nanmedian(data.wl_flux_err))
                init_params_wl['GP_log_rho'] = jnp.log(0.1)
                whitelight_model_for_run = create_whitelight_model(detrend_type=detrending_type, n_planets=n_planets, ld_profile=ld_profile)
                soln = optimx.optimize(whitelight_model_for_run, start=init_params_wl)(
                    key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl
                )
            else:
                soln =  optimx.optimize(whitelight_model_for_run, start=init_params_wl)(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)
            mcmc = numpyro.infer.MCMC(
                numpyro.infer.NUTS(whitelight_model_for_run, regularize_mass_matrix=False, init_strategy=numpyro.infer.init_to_value(values=soln), target_accept_prob=0.9),
                num_warmup=1000, num_samples=1000, progress_bar=True, jit_model_args=True, chain_method='vectorized'
            )
            mcmc.run(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)
            inf_data = az.from_numpyro(mcmc)
            if save_trace: az.to_netcdf(inf_data, f'whitelight_trace_{n_planets}planets.nc')
            wl_samples = mcmc.get_samples()
            print(az.summary(inf_data, var_names=None, round_to=7))

            # ------------------------------------------------
            # PARAMETER EXTRACTION
            # ------------------------------------------------
            bestfit_params_wl = {
                'period': PERIOD_FIXED,
            }
            if 'c1' in wl_samples and ld_profile == 'power2':
                bestfit_params_wl['c1'] = jnp.nanmedian(wl_samples['c1'], axis=0)
                bestfit_params_wl['c2'] = jnp.nanmedian(wl_samples['c2'], axis=0)
                
                POLY_DEGREE = 12
                MUS = jnp.linspace(0.0, 1.00, 300, endpoint=True)
                power2_profile = get_I_power2(bestfit_params_wl['c1'], bestfit_params_wl['c2'], MUS)
                u_poly = calc_poly_coeffs(MUS, power2_profile, poly_degree=POLY_DEGREE)
                bestfit_params_wl['u'] = u_poly
            else:
                bestfit_params_wl['u'] = jnp.nanmedian(wl_samples['u'], axis=0)

            durations_fit, t0s_fit, bs_fit, rors_fit = [], [], [], []
            durations_err, t0s_err, bs_err, rors_err, depths_err = [], [], [], [], []

            for i in range(n_planets):
                durations_fit.append(jnp.nanmedian(wl_samples[f'duration_{i}']))
                t0s_fit.append(jnp.nanmedian(wl_samples[f't0_{i}']))
                bs_fit.append(jnp.nanmedian(wl_samples[f'b_{i}']))
                rors_fit.append(jnp.nanmedian(wl_samples[f'rors_{i}']))
                durations_err.append(jnp.std(wl_samples[f'duration_{i}']))
                t0s_err.append(jnp.std(wl_samples[f't0_{i}']))
                bs_err.append(jnp.std(wl_samples[f'b_{i}']))
                rors_err.append(jnp.std(wl_samples[f'rors_{i}']))
                depths_err.append(jnp.std(wl_samples[f'rors_{i}']**2))

            bestfit_params_wl['duration'] = jnp.array(durations_fit)
            bestfit_params_wl['t0'] = jnp.array(t0s_fit)
            bestfit_params_wl['b'] = jnp.array(bs_fit)
            bestfit_params_wl['rors'] = jnp.array(rors_fit)
            bestfit_params_wl['depths'] = jnp.array(rors_fit)**2

            bestfit_params_wl['duration_err'] = jnp.array(durations_err)
            bestfit_params_wl['t0_err'] = jnp.array(t0s_err)
            bestfit_params_wl['b_err'] = jnp.array(bs_err)
            bestfit_params_wl['rors_err'] = jnp.array(rors_err)
            bestfit_params_wl['depths_err'] = jnp.array(depths_err)
            bestfit_params_wl['error'] = jnp.nanmedian(wl_samples['error'])
            
            if detrending_type != 'none':
                bestfit_params_wl['c'] = jnp.nanmedian(wl_samples['c'])
                if detrending_type != 'gp': 
                    bestfit_params_wl['v'] = jnp.nanmedian(wl_samples['v'])
                if 'v' in wl_samples and 'v' not in bestfit_params_wl:
                     bestfit_params_wl['v'] = jnp.nanmedian(wl_samples['v'])

            if 'v2' in wl_samples:
                bestfit_params_wl['v2'] = jnp.nanmedian(wl_samples['v2'])
            if 'v3' in wl_samples:
                bestfit_params_wl['v3'] = jnp.nanmedian(wl_samples['v3'])
            if 'v4' in wl_samples:
                bestfit_params_wl['v4'] = jnp.nanmedian(wl_samples['v4'])
            if 'explinear' in detrending_type:
                bestfit_params_wl['A'] = jnp.nanmedian(wl_samples['A'])
                bestfit_params_wl['tau'] = jnp.nanmedian(wl_samples['tau'])
            if 'spot' in detrending_type:
                bestfit_params_wl['spot_amp'] = jnp.nanmedian(wl_samples['spot_amp'])
                bestfit_params_wl['spot_mu'] = jnp.nanmedian(wl_samples['spot_mu'])
                bestfit_params_wl['spot_sigma'] = jnp.nanmedian(wl_samples['spot_sigma'])
            if 'linear_discontinuity' in detrending_type:
                bestfit_params_wl['t_jump'] = jnp.nanmedian(wl_samples['t_jump'])
                bestfit_params_wl['jump'] = jnp.nanmedian(wl_samples['jump'])
    
            if 'gp' in detrending_type:
                bestfit_params_wl['GP_log_sigma'] = jnp.nanmedian(wl_samples['GP_log_sigma'])
                bestfit_params_wl['GP_log_rho'] = jnp.nanmedian(wl_samples['GP_log_rho'])

            spot_trend, jump_trend = None, None
            if 'spot' in detrending_type:
                spot_trend = spot_crossing(data.wl_time, bestfit_params_wl["spot_amp"], bestfit_params_wl["spot_mu"], bestfit_params_wl["spot_sigma"])
            if 'linear_discontinuity' in detrending_type:
                jump_trend = jnp.where(data.wl_time > bestfit_params_wl["t_jump"], bestfit_params_wl["jump"], 0.0)

            # ------------------------------------------------
            # RECONSTRUCT MODEL
            # ------------------------------------------------
            if 'gp' in detrending_type:
                # Select the correct mean function
                if 'linear' in detrending_type and 'explinear' not in detrending_type:
                    gp_mean_func = compute_lc_linear_gp_mean
                elif 'quadratic' in detrending_type:
                    gp_mean_func = compute_lc_quadratic_gp_mean
                elif 'explinear' in detrending_type:
                    gp_mean_func = compute_lc_explinear_gp_mean
                else:
                    gp_mean_func = compute_lc_gp_mean

                wl_kernel = tinygp.kernels.quasisep.Matern32(
                    scale=jnp.exp(bestfit_params_wl['GP_log_rho']),
                    sigma=jnp.exp(bestfit_params_wl['GP_log_sigma']),
                )
                wl_gp = tinygp.GaussianProcess(
                    wl_kernel,
                    data.wl_time,
                    diag=bestfit_params_wl['error']**2,
                    mean=partial(gp_mean_func, bestfit_params_wl),
                )
                cond_gp = wl_gp.condition(data.wl_flux, data.wl_time).gp
                mu, var = cond_gp.loc, cond_gp.variance
                wl_transit_model = mu
                
                planet_model_only = compute_transit_model(bestfit_params_wl, data.wl_time)
                # The total trend (parametric + stochastic) is Total - Planet - 1
                trend_flux_total = mu - planet_model_only - 1.0
                
                # The "GP Trend" (stochastic part only) is Posterior Mean - Parametric Mean
                parametric_mean_val = gp_mean_func(bestfit_params_wl, data.wl_time)
                gp_stochastic_component = mu - parametric_mean_val

            elif detrending_type == 'linear':
                wl_transit_model = compute_lc_linear(bestfit_params_wl, data.wl_time)
            elif detrending_type == 'quadratic':
                wl_transit_model = compute_lc_quadratic(bestfit_params_wl, data.wl_time)
            elif detrending_type == 'cubic':
                wl_transit_model = compute_lc_cubic(bestfit_params_wl, data.wl_time)
            elif detrending_type == 'quartic':
                wl_transit_model = compute_lc_quartic(bestfit_params_wl, data.wl_time)
            elif detrending_type == 'explinear':
                wl_transit_model = compute_lc_explinear(bestfit_params_wl, data.wl_time)
            elif detrending_type == 'none':
                wl_transit_model = compute_lc_none(bestfit_params_wl, data.wl_time)
            elif detrending_type == 'linear_discontinuity':
                wl_transit_model = compute_lc_linear_discontinuity(bestfit_params_wl, data.wl_time)
            else:
                 print('Error with model, not defined!')
                 exit()

            wl_residual = data.wl_flux - wl_transit_model
            wl_sigma = 1.4826 * jnp.nanmedian(np.abs(wl_residual - jnp.nanmedian(wl_residual)))
            wl_mad_mask = jnp.abs(wl_residual - jnp.nanmedian(wl_residual)) > whitelight_sigma * wl_sigma
            wl_sigma_post_clip = 1.4826 * jnp.nanmedian(jnp.abs(wl_residual[~wl_mad_mask] - jnp.nanmedian(wl_residual[~wl_mad_mask])))

            plt.plot(data.wl_time, wl_transit_model, color="mediumorchid", lw=2, zorder=3)
            plt.scatter(data.wl_time, data.wl_flux, s=6, c='k', zorder=1, alpha=0.5)

            plt.savefig(f"{output_dir}/11_{instrument_full_str}_whitelightmodel.png")
            plt.close()

            plt.scatter(data.wl_time, wl_residual, s=6, c='k')
            plt.title('WL Pre-outlier rejection residual')
            plt.savefig(f"{output_dir}/12_{instrument_full_str}_whitelightresidual.png")
            plt.close()

            # ------------------------------------------------
            # DETRENDED FLUX CALCULATION
            # ------------------------------------------------
            t_masked = data.wl_time[~wl_mad_mask]
            f_masked = data.wl_flux[~wl_mad_mask]
            t_norm_masked = t_masked - jnp.min(t_masked)

            if 'gp' in detrending_type:
                planet_model_masked = compute_transit_model(bestfit_params_wl, t_masked)
                mu_masked = mu[~wl_mad_mask] 
                total_trend_at_points = mu_masked - planet_model_masked
                detrended_flux = f_masked - (total_trend_at_points - 1.0)
                gp_stochastic_at_masked = gp_stochastic_component[~wl_mad_mask]

            elif detrending_type == 'linear':
                trend = bestfit_params_wl["c"] + bestfit_params_wl["v"] * t_norm_masked
                detrended_flux = f_masked - trend + 1.0

            elif detrending_type == 'quadratic':
                trend = (bestfit_params_wl["c"] + 
                         bestfit_params_wl["v"] * t_norm_masked + 
                         bestfit_params_wl["v2"] * t_norm_masked**2)
                detrended_flux = f_masked - trend + 1.0

            elif detrending_type == 'cubic':
                trend = (bestfit_params_wl["c"] + 
                         bestfit_params_wl["v"] * t_norm_masked + 
                         bestfit_params_wl["v2"] * t_norm_masked**2 + 
                         bestfit_params_wl["v3"] * t_norm_masked**3)
                detrended_flux = f_masked - trend + 1.0

            elif detrending_type == 'quartic':
                trend = (bestfit_params_wl["c"] + 
                         bestfit_params_wl["v"] * t_norm_masked + 
                         bestfit_params_wl["v2"] * t_norm_masked**2 + 
                         bestfit_params_wl["v3"] * t_norm_masked**3 + 
                         bestfit_params_wl["v4"] * t_norm_masked**4)
                detrended_flux = f_masked - trend + 1.0

            elif detrending_type == 'explinear':
                trend = (bestfit_params_wl["c"] + 
                         bestfit_params_wl["v"] * t_norm_masked + 
                         bestfit_params_wl['A'] * jnp.exp(-t_norm_masked / bestfit_params_wl['tau']))
                detrended_flux = f_masked - trend + 1.0

            elif detrending_type == 'linear_discontinuity':
                jump = jnp.where(t_masked > bestfit_params_wl["t_jump"], bestfit_params_wl["jump"], 0.0)
                trend = bestfit_params_wl["c"] + bestfit_params_wl["v"] * t_norm_masked + jump
                detrended_flux = f_masked - trend + 1.0

            elif detrending_type == 'spot':
                spot = spot_crossing(t_masked, bestfit_params_wl["spot_amp"], 
                                     bestfit_params_wl["spot_mu"], bestfit_params_wl["spot_sigma"])
                trend = bestfit_params_wl["c"] + bestfit_params_wl["v"] * t_norm_masked + spot
                detrended_flux = f_masked - trend + 1.0

            elif detrending_type == 'none':
                detrended_flux = f_masked 
                
            else:
                trend = bestfit_params_wl["c"] + bestfit_params_wl["v"] * t_norm_masked
                detrended_flux = f_masked - trend + 1.0 

            plt.scatter(t_masked, detrended_flux, c='k', s=6, alpha=0.5)
            plt.title(f'Detrended WLC: Sigma {round(wl_sigma_post_clip*1e6)} PPM')
            plt.savefig(f'{output_dir}/14_{instrument_full_str}_whitelightdetrended.png')
            plt.close()

            transit_only_model = compute_transit_model(bestfit_params_wl, t_masked) + 1.0
            residuals_detrended = detrended_flux - transit_only_model 

            fig = plt.figure(figsize=(16, 14))
            b_time, b_flux = jax_bin_lightcurve(jnp.array(data.wl_time), 
                                                jnp.array(data.wl_flux), 
                                                bestfit_params_wl['duration'])
            
            b_time_det, b_flux_det = jax_bin_lightcurve(jnp.array(t_masked), 
                                                        jnp.array(detrended_flux), 
                                                        bestfit_params_wl['duration'])
            b_time_det, b_res_det = jax_bin_lightcurve(jnp.array(t_masked), 
                                            jnp.array(residuals_detrended), 
                                            bestfit_params_wl['duration'])
            bin_style = dict(c='darkviolet', edgecolors='darkslateblue', s=40,  zorder=10, label='Binned (8/dur)')

            gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.5], 
                                   width_ratios=[1, 1], hspace=0.3, wspace=0.3)

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.scatter(data.wl_time, data.wl_flux, c='k', s=4, alpha=0.2)
            ax1.scatter(np.array(b_time), np.array(b_flux), **bin_style)
            ax1.set_title('Raw Light Curve', fontsize=14)
            ax1.set_ylabel('Flux', fontsize=12)
            ax1.tick_params(labelbottom=False)

            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax2.scatter(data.wl_time, data.wl_flux, c='k', s=4, alpha=0.2)
            ax2.scatter(np.array(b_time), np.array(b_flux), **bin_style)
            ax2.plot(data.wl_time, wl_transit_model, color="mediumorchid", lw=2, zorder=3)
            ax2.set_title('Raw Light Curve + Best-fit Model', fontsize=14)
            ax2.set_ylabel('Flux', fontsize=12)
            ax2.tick_params(labelbottom=False)

            gs_nested = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs[2, 0], 
                height_ratios=[2, 1],
                hspace=0.0
            )

            ax3_top = fig.add_subplot(gs_nested[0], sharex=ax1)

            ax3_top.scatter(t_masked, detrended_flux, c='k', s=4, alpha=0.2, label='Detrended Data')
            ax3_top.plot(t_masked, transit_only_model, color="mediumorchid", lw=2, zorder=3, label='Transit Model')
            ax3_top.scatter(np.array(b_time_det), np.array(b_flux_det), **bin_style)
            ax3_top.set_ylabel('Normalized Flux', fontsize=12)
            ax3_top.set_title('Detrended Light Curve', fontsize=14)
            plt.setp(ax3_top.get_xticklabels(), visible=False)

            ax3_bot = fig.add_subplot(gs_nested[1], sharex=ax3_top)
    

            ax3_bot.scatter(t_masked, residuals_detrended * 1e6, c='k', s=4, alpha=0.2)
            ax3_bot.axhline(0, color='mediumorchid', lw=4, zorder=3, linestyle='--')
            ax3_bot.scatter(np.array(b_time_det), np.array(b_res_det) * 1e6 , **bin_style)
            ax3_bot.set_ylabel('Res. (ppm)', fontsize=10)
            ax3_bot.set_xlabel('Time (BJD)', fontsize=12)

            dt = np.median(np.diff(data.wl_time)) * 86400 
            residuals_arr = np.array(wl_residual[~wl_mad_mask])

            beta, bin_sizes_min, measured_rms, expected_rms = calculate_beta_metrics(residuals_arr, dt)
            mc_betas, rms_lo_1, rms_hi_1, rms_lo_2, rms_hi_2 = run_beta_monte_carlo(residuals_arr, dt, n_sims=500)

            mu_sim, std_sim = norm.fit(mc_betas)
            z_score = (beta - mu_sim) / std_sim

            print(f"Beta: {beta:.4f}")
            print(f"Measured Beta: {beta:.3f}")
            print(f"MC Mean Beta:  {mu_sim:.3f}")
            print(f"MC Std Dev:    {std_sim:.3f}")
            print(f"Significance:  {z_score:.2f} sigma")

            ax_rms = fig.add_subplot(gs[0:2, 1])
            ax_rms.loglog(bin_sizes_min, expected_rms * 1e6, 'k--', lw=1.5, label='Theory $1/\sqrt{N}$')
            ax_rms.fill_between(bin_sizes_min, rms_lo_2 * 1e6, rms_hi_2 * 1e6, color='gray', alpha=0.2, label='White Noise ($2\sigma$)')
            ax_rms.fill_between(bin_sizes_min, rms_lo_1 * 1e6, rms_hi_1 * 1e6, color='gray', alpha=0.4, label='White Noise ($1\sigma$)')
            ax_rms.loglog(bin_sizes_min, measured_rms * 1e6, color='teal', lw=2, marker='o', markersize=5, label=f'Data (Beta={beta:.2f})')
            ax_rms.set_xlabel('Bin Size (minutes)', fontsize=12)
            ax_rms.set_ylabel('RMS (ppm)', fontsize=12)
            ax_rms.set_title('Time-Correlated Noise', fontsize=14)
            ax_rms.grid(True, which="both", alpha=0.2)

            ax_beta = fig.add_subplot(gs[2, 1])

            n, bins, patches = ax_beta.hist(mc_betas, bins=30, color='silver', alpha=0.6, density=True, label='Simulated White Noise')

            xmin, xmax = ax_beta.get_xlim()
            x_plot = np.linspace(xmin, xmax, 100)
            p_plot = norm.pdf(x_plot, mu_sim, std_sim)
            ax_beta.plot(x_plot, p_plot, 'k--', linewidth=2, label='Gaussian Fit')

            ax_beta.axvline(beta, color='teal', lw=3, label=f'Measured: {beta:.2f}')

            sig_color = 'green' if abs(z_score) < 2.0 else ('orange' if abs(z_score) < 3.0 else 'firebrick')
            ax_beta.text(0.95, 0.85, f"Significance: {z_score:.1f}$\sigma$", 
                       transform=ax_beta.transAxes, ha='right', fontsize=14, color=sig_color, fontweight='bold')

            ax_beta.set_xlabel('Beta Factor', fontsize=12)
            ax_beta.set_ylabel('Probability Density', fontsize=12)
            ax_beta.set_title("Beta Significance Test", fontsize=14)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/15_{instrument_full_str}_whitelight_summary.png')
            plt.close(fig)

            np.save(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy', arr=wl_mad_mask)
            
            if 'gp' in detrending_type:
                df = pd.DataFrame({
                    'wl_flux': data.wl_flux, 
                    'gp_flux': mu,
                    'gp_err': jnp.sqrt(var), 
                    'gp_trend': gp_stochastic_component
                }) 
                df.to_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
            
            rows = []
            for i in range(n_planets):
                row = {
                    'planet_id': i,
                    'period': bestfit_params_wl['period'][i],
                    'duration': bestfit_params_wl['duration'][i],
                    't0': bestfit_params_wl['t0'][i],
                    'b': bestfit_params_wl['b'][i],
                    'rors': bestfit_params_wl['rors'][i],
                    'depths': bestfit_params_wl['depths'][i],
                }
                if detrending_type != 'none':
                    row['c'] = bestfit_params_wl['c']
                    if 'v' in bestfit_params_wl: row['v'] = bestfit_params_wl['v']
                
                if 'c1' in bestfit_params_wl:
                    row['u1'] = bestfit_params_wl['c1']
                    row['u2'] = bestfit_params_wl['c2']
                    row['c1'] = bestfit_params_wl['c1']
                    row['c2'] = bestfit_params_wl['c2']
                else:
                    row['u1'] = bestfit_params_wl['u'][0]
                    row['u2'] = bestfit_params_wl['u'][1]
                
                if 'explinear' in detrending_type:
                    row['A'] = bestfit_params_wl['A']
                    row['tau'] = bestfit_params_wl['tau']
                elif 'quadratic' in detrending_type:
                    row['v2'] = bestfit_params_wl['v2']
                if 'gp' in detrending_type:
                    row['GP_log_sigma'] = bestfit_params_wl['GP_log_sigma']
                    row['GP_log_rho'] = bestfit_params_wl['GP_log_rho']
                
                rows.append(row)
        
            df = pd.DataFrame(rows)
            df.to_csv(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv', index=False)
            bestfit_params_wl_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv')

            DURATION_BASE = bestfit_params_wl_df['duration'].values
            T0_BASE = bestfit_params_wl_df['t0'].values
            B_BASE = bestfit_params_wl_df['b'].values
            RORS_BASE = bestfit_params_wl_df['rors'].values
            DEPTH_BASE = RORS_BASE**2
        else:
            print(f'GP trends already exist...')
            wl_mad_mask = np.load(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy')
            bestfit_params_wl_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv')
            DURATION_BASE = bestfit_params_wl_df['duration'].values
            T0_BASE = bestfit_params_wl_df['t0'].values
            B_BASE = bestfit_params_wl_df['b'].values
            RORS_BASE = bestfit_params_wl_df['rors'].values
            DEPTH_BASE = RORS_BASE**2
    else:
        print(f'Whitelight outliers and bestfit parameters already exist...')
        wl_mad_mask = np.load(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy')
        bestfit_params_wl_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv')
        DURATION_BASE = bestfit_params_wl_df['duration'].values
        T0_BASE = bestfit_params_wl_df['t0'].values
        B_BASE = bestfit_params_wl_df['b'].values
        RORS_BASE = bestfit_params_wl_df['rors'].values
        DEPTH_BASE = RORS_BASE**2

    key_lr, key_hr, key_map_lr, key_mcmc_lr, key_map_hr, key_mcmc_hr, key_prior_pred = jax.random.split(key_master, 7)
    need_lowres_analysis = interpolate_trend or interpolate_ld or need_lowres
    
    trend_fixed_hr = None
    ld_fixed_hr = None
    best_poly_coeffs_c, best_poly_coeffs_v = None, None
    best_poly_coeffs_u1, best_poly_coeffs_u2 = None, None
    spot_trend, jump_trend = None, None

    # ----------------------------------------------------
    # LOW RES ANALYSIS
    # ----------------------------------------------------
    if need_lowres_analysis:
        print(f"\n--- Running Low-Resolution Analysis (Binned to {lr_bin_str}) ---")
        time_lr = jnp.array(data.time[~wl_mad_mask])
        flux_lr = jnp.array(data.flux_lr[:, ~wl_mad_mask])
        flux_err_lr = jnp.array(data.flux_err_lr[:, ~wl_mad_mask])
        num_lcs_lr = jnp.array(data.flux_err_lr.shape[0])

        if 'gp' in detrending_type:
            gp_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
            gp_trend = jnp.array(gp_df['gp_trend'].values)[~wl_mad_mask]
            
            detrend_type_multiwave = detrending_type.replace('gp', 'gp_spectroscopic')
        else:
            detrend_type_multiwave = detrending_type
            gp_trend = None

        print(f"Low-res: {num_lcs_lr} light curves.")
        DEPTHS_BASE_LR = jnp.tile(DEPTH_BASE, (num_lcs_lr, 1))

        if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'MIRI/LRS']:
            U_mu_lr = get_limb_darkening(sld, data.wavelengths_lr, data.wavelengths_err_lr , instrument, ld_profile=ld_profile)
        elif instrument == 'NIRISS/SOSS':
            U_mu_lr = get_limb_darkening(sld, data.wavelengths_lr, data.wavelengths_err_lr, instrument, order=order, ld_profile=ld_profile)

        init_params_lr = {
            "u": U_mu_lr,
            "depths": jnp.tile(DEPTH_BASE, (num_lcs_lr, 1))
        }
        if detrend_type_multiwave != 'none':
            init_params_lr['c'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['c'].values[0])
            if 'v' in bestfit_params_wl_df.columns:
                 init_params_lr['v'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['v'].values[0])

        if 'explinear' in detrend_type_multiwave:
            init_params_lr['A'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['A'].values[0])
            init_params_lr['tau'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['tau'].values[0])
        
        lr_trend_mode = 'free'
        lr_ld_mode = 'fixed' if flags.get('fix_ld', False) else 'free'
        
        lr_model_for_run = create_vectorized_model(
            detrend_type=detrend_type_multiwave,
            ld_mode=lr_ld_mode,
            trend_mode=lr_trend_mode,
            n_planets=n_planets,
            ld_profile=ld_profile
        )

        model_run_args_lr = {
            'mu_duration': DURATION_BASE,
            'mu_t0': T0_BASE,
            'mu_b': B_BASE,
            'mu_depths': DEPTHS_BASE_LR,
            'PERIOD': PERIOD_FIXED,
        }

        if lr_ld_mode == 'fixed': model_run_args_lr['ld_fixed'] = U_mu_lr
        if lr_ld_mode == 'free': model_run_args_lr['mu_u_ld'] = U_mu_lr
        
        if 'gp_spectroscopic' in detrend_type_multiwave:
            model_run_args_lr['gp_trend'] = gp_trend
            init_params_lr['A_gp'] = jnp.ones(num_lcs_lr)
        if 'spot_spectroscopic' in detrend_type_multiwave:
            model_run_args_lr['spot_trend'] = spot_trend
            init_params_lr['A_spot'] = jnp.ones(num_lcs_lr)
        if 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
            model_run_args_lr['jump_trend'] = jump_trend
            init_params_lr['A_jump'] = jnp.ones(num_lcs_lr)

        samples_lr = get_samples(lr_model_for_run, key_mcmc_lr, time_lr, flux_err_lr, flux_lr, init_params_lr, **model_run_args_lr)

        ld_u_lr = np.array(samples_lr["u"])
        if detrend_type_multiwave != 'none':
            trend_c_lr = np.array(samples_lr["c"])
            if 'v' in samples_lr: trend_v_lr = np.array(samples_lr["v"])
        if 'explinear' in detrend_type_multiwave:
            trend_A_lr = np.array(samples_lr["A"])
            trend_tau_lr = np.array(samples_lr["tau"])

        map_params_lr = {
            "duration": DURATION_BASE, "t0": T0_BASE, "b": B_BASE,
            "rors": jnp.nanmedian(samples_lr["rors"], axis=0), 
            "period": PERIOD_FIXED,
        }
        
        if ld_profile == 'power2':
            c1_med = jnp.nanmedian(samples_lr['c1'], axis=0)
            c2_med = jnp.nanmedian(samples_lr['c2'], axis=0)
            POLY_DEGREE = 12
            MUS = jnp.linspace(0.0, 1.00, 300, endpoint=True)
            
            def compute_u_from_c(c1, c2):
                 profile = get_I_power2(c1, c2, MUS)
                 return calc_poly_coeffs(MUS, profile, poly_degree=POLY_DEGREE)
            
            map_params_lr['u'] = jax.vmap(compute_u_from_c)(c1_med, c2_med)
        else:
            map_params_lr['u'] = jnp.nanmedian(ld_u_lr, axis=0)

        if detrend_type_multiwave != 'none':
            map_params_lr['c'] = jnp.nanmedian(samples_lr["c"], axis=0)
            if 'v' in samples_lr: 
                map_params_lr['v'] = jnp.nanmedian(samples_lr["v"], axis=0)
        if 'v2' in samples_lr:
            map_params_lr['v2'] = jnp.nanmedian(samples_lr['v2'], axis=0)
        if 'v3' in samples_lr:
            map_params_lr['v3'] = jnp.nanmedian(samples_lr['v3'], axis=0)
        if 'v4' in samples_lr:
            map_params_lr['v4'] = jnp.nanmedian(samples_lr['v4'], axis=0)
        if 'explinear' in detrend_type_multiwave:
            map_params_lr['A'] = jnp.nanmedian(samples_lr['A'], axis=0)
            map_params_lr['tau'] = jnp.nanmedian(samples_lr['tau'], axis=0)


        selected_kernel = COMPUTE_KERNELS[detrend_type_multiwave]
        in_axes_map = {'rors': 0, 'u': 0}
        if detrend_type_multiwave != 'none':
            in_axes_map['c'] = 0
            if 'v' in samples_lr: in_axes_map['v'] = 0
        if 'explinear' in detrend_type_multiwave:
            in_axes_map.update({'A': 0, 'tau': 0})
        
        final_in_axes = {k: in_axes_map.get(k, None) for k in map_params_lr.keys()}

        if 'gp_spectroscopic' in detrend_type_multiwave:
            map_params_lr['A_gp'] = jnp.nanmedian(samples_lr['A_gp'], axis=0)
            final_in_axes['A_gp'] = 0
            model_all = jax.vmap(selected_kernel, in_axes=(final_in_axes, None, None))(map_params_lr, time_lr, gp_trend)
        else:
            model_all = jax.vmap(selected_kernel, in_axes=(final_in_axes, None))(map_params_lr, time_lr)

        residuals = flux_lr - model_all
        plot_noise_binning(residuals, f"{output_dir}/25_{instrument_full_str}_{lr_bin_str}_noisebin.png")

        # Outlier rejection for LR
        medians = np.nanmedian(residuals, axis=1, keepdims=True)
        sigmas    = 1.4826 * np.nanmedian(np.abs(residuals - medians), axis=1, keepdims=True)
        point_mask = np.abs(residuals - medians) > spectroscopic_sigma * sigmas
        time_mask = np.any(point_mask, axis=0)
        valid = ~time_mask
        time_lr = time_lr[valid]
        flux_lr = flux_lr[:, valid]
        flux_err_lr = flux_err_lr[:, valid]
        if gp_trend is not None: gp_trend = gp_trend[valid]
        if spot_trend is not None: spot_trend = spot_trend[valid]
        if jump_trend is not None: jump_trend = jump_trend[valid]
        
        print("Plotting low-resolution fits and residuals...")
        median_total_error_lr = np.nanmedian(samples_lr['total_error'], axis=0)
        plot_wavelength_offset_summary(time_lr, flux_lr, median_total_error_lr, data.wavelengths_lr,
                                     map_params_lr, {"period": PERIOD_FIXED},
                                     f"{output_dir}/22_{instrument_full_str}_{lr_bin_str}_summary.png",
                                     detrend_type=detrend_type_multiwave, gp_trend=gp_trend)

        # Polynomial Fitting
        poly_orders = [1, 2, 3, 4]
        wl_lr = np.array(data.wavelengths_lr)

        if detrending_type != 'none':
            print("Fitting polynomials to trend coefficients...")
            best_poly_coeffs_c, best_order_c, _ = fit_polynomial(wl_lr, trend_c_lr, poly_orders)
            if 'v' in samples_lr:
                best_poly_coeffs_v, best_order_v, _ = fit_polynomial(wl_lr, trend_v_lr, poly_orders)
                plot_poly_fit(wl_lr, trend_v_lr, best_poly_coeffs_v, best_order_v, "Wavelength", "v", "Trend Slope v", f"{output_dir}/2opt_v.png")

        if interpolate_ld:
            print("Fitting polynomials to limb darkening coefficients...")
            if ld_profile == 'power2':
                # Use c1 and c2 from samples
                c1_lr = np.array(samples_lr['c1'])
                c2_lr = np.array(samples_lr['c2'])
                best_poly_coeffs_u1, best_order_u1, _ = fit_polynomial(wl_lr, c1_lr, poly_orders)
                best_poly_coeffs_u2, best_order_u2, _ = fit_polynomial(wl_lr, c2_lr, poly_orders)
                plot_poly_fit(wl_lr, c1_lr, best_poly_coeffs_u1, best_order_u1, "Wavelength", "c1", "Limb Darkening c1", f"{output_dir}/2opt_c1.png")
                plot_poly_fit(wl_lr, c2_lr, best_poly_coeffs_u2, best_order_u2, "Wavelength", "c2", "Limb Darkening c2", f"{output_dir}/2opt_c2.png")
            else:
                best_poly_coeffs_u1, best_order_u1, _ = fit_polynomial(wl_lr, ld_u_lr[:, :, 0], poly_orders)
                best_poly_coeffs_u2, best_order_u2, _ = fit_polynomial(wl_lr, ld_u_lr[:, :, 1], poly_orders)
                plot_poly_fit(wl_lr, ld_u_lr[:, :, 0], best_poly_coeffs_u1, best_order_u1, "Wavelength", "u1", "Limb Darkening u1", f"{output_dir}/2opt_u1.png")
                plot_poly_fit(wl_lr, ld_u_lr[:, :, 1], best_poly_coeffs_u2, best_order_u2, "Wavelength", "u2", "Limb Darkening u2", f"{output_dir}/2opt_u2.png")

        plot_transmission_spectrum(wl_lr, samples_lr["rors"], f"{output_dir}/24_{instrument_full_str}_{lr_bin_str}_spectrum")
        save_results(wl_lr, data.wavelengths_err_lr, samples_lr, f"{output_dir}/{instrument_full_str}_{lr_bin_str}.csv")
        save_detailed_fit_results(time_lr, flux_lr, flux_err_lr, data.wavelengths_lr, data.wavelengths_err_lr, samples_lr, map_params_lr, {"period": PERIOD_FIXED}, detrend_type_multiwave, f"{output_dir}/{instrument_full_str}_{lr_bin_str}", median_total_error_lr, gp_trend=gp_trend, spot_trend=spot_trend, jump_trend=jump_trend)

    # ----------------------------------------------------
    # HIGH RES ANALYSIS
    # ----------------------------------------------------
    print(f"\n--- Running High-Resolution Analysis (Binned to {hr_bin_str}) ---")
    time_hr = jnp.array(data.time[~wl_mad_mask])
    flux_hr = jnp.array(data.flux_hr[:, ~wl_mad_mask])
    flux_err_hr = jnp.array(data.flux_err_hr[:, ~wl_mad_mask])

    if 'gp' in detrending_type:
        gp_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
        gp_trend = jnp.array(gp_df['gp_trend'].values)[~wl_mad_mask]
        detrend_type_multiwave = detrending_type.replace('gp', 'gp_spectroscopic')
    else:
        detrend_type_multiwave = detrending_type
        gp_trend = None

    if need_lowres_analysis:
        time_hr = time_hr[valid]
        flux_hr = flux_hr[:, valid]
        flux_err_hr = flux_err_hr[:, valid]
        if gp_trend is not None: gp_trend = gp_trend[valid]
        if spot_trend is not None: spot_trend = spot_trend[valid]
        if jump_trend is not None: jump_trend = jump_trend[valid]

    num_lcs_hr = flux_err_hr.shape[0]
    DEPTHS_BASE_HR = jnp.tile(DEPTH_BASE, (num_lcs_hr, 1))
    
    hr_ld_mode = 'free'
    if flags.get('interpolate_ld', False): hr_ld_mode = 'interpolated'
    elif flags.get('fix_ld', False): hr_ld_mode = 'fixed'
    hr_trend_mode = 'fixed' if flags.get('interpolate_trend', False) else 'free'

    model_run_args_hr = {}
    wl_hr = np.array(data.wavelengths_hr)

    if hr_ld_mode == 'interpolated':
        u1_interp_hr = np.polyval(best_poly_coeffs_u1, wl_hr)
        u2_interp_hr = np.polyval(best_poly_coeffs_u2, wl_hr)
        
        if ld_profile == 'power2':
            # u1_interp_hr corresponds to c1, u2 to c2
            # We need to reconstruct the polynomial u from these
            POLY_DEGREE = 12
            MUS = jnp.linspace(0.0, 1.00, 300, endpoint=True)
            
            def compute_one_lc_u_interp(c1_val, c2_val):
                 power2_profile = get_I_power2(c1_val, c2_val, MUS)
                 return calc_poly_coeffs(MUS, power2_profile, poly_degree=POLY_DEGREE)
            
            # Use vmap to do it for all wavelengths
            ld_interpolated_hr = jax.vmap(compute_one_lc_u_interp)(jnp.array(u1_interp_hr), jnp.array(u2_interp_hr))
        else:
            ld_interpolated_hr = jnp.array(np.column_stack((u1_interp_hr, u2_interp_hr)))
            
        model_run_args_hr['ld_interpolated'] = ld_interpolated_hr
    elif hr_ld_mode == 'fixed' or hr_ld_mode == 'free':
        if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'MIRI/LRS']:
            U_mu_hr_init = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument, ld_profile=ld_profile)
        elif instrument == 'NIRISS/SOSS':
            U_mu_hr_init = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument, order=order, ld_profile=ld_profile)
        if hr_ld_mode == 'fixed': model_run_args_hr['ld_fixed'] = U_mu_hr_init
        else: model_run_args_hr['mu_u_ld'] = U_mu_hr_init

    if hr_trend_mode == 'fixed':
        c_interp_hr = np.polyval(best_poly_coeffs_c, wl_hr)
        v_interp_hr = np.polyval(best_poly_coeffs_v, wl_hr)
        trend_fixed_hr = np.column_stack((c_interp_hr, v_interp_hr))
        model_run_args_hr['trend_fixed'] = jnp.array(trend_fixed_hr)

    model_run_args_hr['mu_duration'] = DURATION_BASE
    model_run_args_hr['mu_t0'] = T0_BASE
    model_run_args_hr['mu_b'] = B_BASE
    model_run_args_hr['mu_depths'] = DEPTHS_BASE_HR
    model_run_args_hr['PERIOD'] = PERIOD_FIXED

    init_params_hr = { "depths": DEPTHS_BASE_HR, "u": U_mu_hr_init if hr_ld_mode!='interpolated' else ld_interpolated_hr }
    if hr_trend_mode == 'free':
        if detrend_type_multiwave != 'none':
            init_params_hr["c"] = np.polyval(best_poly_coeffs_c, wl_hr)
            if 'v' in bestfit_params_wl_df.columns:
                 init_params_hr["v"] = np.polyval(best_poly_coeffs_v, wl_hr)

    hr_model_for_run = create_vectorized_model(
        detrend_type=detrend_type_multiwave,
        ld_mode=hr_ld_mode,
        trend_mode=hr_trend_mode,
        n_planets=n_planets,
        ld_profile=ld_profile
    )
    
    if 'gp_spectroscopic' in detrend_type_multiwave:
        model_run_args_hr['gp_trend'] = gp_trend
        init_params_hr['A_gp'] = jnp.ones(num_lcs_hr)
    if 'spot_spectroscopic' in detrend_type_multiwave:
        model_run_args_hr['spot_trend'] = spot_trend
        init_params_hr['A_spot'] = jnp.ones(num_lcs_hr)
    if 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
        model_run_args_hr['jump_trend'] = jump_trend
        init_params_hr['A_jump'] = jnp.ones(num_lcs_hr)

    samples_hr = get_samples(hr_model_for_run, key_mcmc_hr, time_hr, flux_err_hr, flux_hr, init_params_hr, **model_run_args_hr)

    map_params_hr = {
        "duration": DURATION_BASE, "t0": T0_BASE, "b": B_BASE,
        "rors": jnp.nanmedian(samples_hr["rors"], axis=0), 
        "period": PERIOD_FIXED
    }
    if "u" in samples_hr and ld_profile != 'power2': 
        map_params_hr["u"] = jnp.nanmedian(np.array(samples_hr["u"]), axis=0)
    elif "u" in samples_hr and ld_profile == 'power2':
        # Reconstruct u from c1/c2 medians for high-res
        # Note: samples_hr['u'] contains polynomial coefficients, but we want to use c1/c2 if available
        # But wait, create_vectorized_model samples c1/c2 for power2.
        # Let's check if c1/c2 are in samples_hr
        if 'c1' in samples_hr:
            c1_med_hr = jnp.nanmedian(samples_hr['c1'], axis=0)
            c2_med_hr = jnp.nanmedian(samples_hr['c2'], axis=0)
            POLY_DEGREE = 12
            MUS = jnp.linspace(0.0, 1.00, 300, endpoint=True)
            def compute_u_from_c(c1, c2):
                 profile = get_I_power2(c1, c2, MUS)
                 return calc_poly_coeffs(MUS, profile, poly_degree=POLY_DEGREE)
            map_params_hr['u'] = jax.vmap(compute_u_from_c)(c1_med_hr, c2_med_hr)
        else:
            # Fallback if c1/c2 not found (should not happen for power2)
             map_params_hr["u"] = jnp.nanmedian(np.array(samples_hr["u"]), axis=0)
    if "c" in samples_hr: 
        map_params_hr["c"] = jnp.nanmedian(samples_hr["c"], axis=0)
    if "v" in samples_hr: 
        map_params_hr["v"] = jnp.nanmedian(samples_hr["v"], axis=0)
    if "v2" in samples_hr: 
        map_params_hr["v2"] = jnp.nanmedian(samples_hr["v2"], axis=0)
    if "v3" in samples_hr: 
        map_params_hr["v3"] = jnp.nanmedian(samples_hr["v3"], axis=0)
    if "v4" in samples_hr: 
        map_params_hr["v4"] = jnp.nanmedian(samples_hr["v4"], axis=0)

    in_axes_map_hr = {"rors": 0, "u": 0}
    if detrend_type_multiwave != 'none':
        in_axes_map_hr.update({"c": 0})
        if "v" in samples_hr: in_axes_map_hr.update({"v": 0})
    
    final_in_axes_hr = {k: in_axes_map_hr.get(k, None) for k in map_params_hr.keys()}
    selected_kernel_hr = COMPUTE_KERNELS[detrend_type_multiwave]
    
    if 'gp_spectroscopic' in detrend_type_multiwave:
        map_params_hr['A_gp'] = jnp.nanmedian(samples_hr['A_gp'], axis=0)
        final_in_axes_hr['A_gp'] = 0
        model_all_hr = jax.vmap(selected_kernel_hr, in_axes=(final_in_axes_hr, None, None))(map_params_hr, time_hr, gp_trend)
    else:
        model_all_hr = jax.vmap(selected_kernel_hr, in_axes=(final_in_axes_hr, None))(map_params_hr, time_hr)

    residuals_hr = np.array(flux_hr - model_all_hr)
    plot_noise_binning(residuals_hr, f"{output_dir}/36_{instrument_full_str}_{hr_bin_str}_noisebin.png")

    median_total_error_hr = np.nanmedian(samples_hr['total_error'], axis=0)
    plot_wavelength_offset_summary(time_hr, flux_hr, median_total_error_hr, data.wavelengths_hr,
                                    map_params_hr, {"period": PERIOD_FIXED},
                                    f"{output_dir}/34_{instrument_full_str}_{hr_bin_str}_summary.png",
                                    detrend_type=detrend_type_multiwave, gp_trend=gp_trend)

    plot_transmission_spectrum(wl_hr, samples_hr["rors"], f"{output_dir}/31_{instrument_full_str}_{hr_bin_str}_spectrum")
    save_results(wl_hr, data.wavelengths_err_hr, samples_hr,  f"{output_dir}/{instrument_full_str}_{hr_bin_str}.csv")
    save_detailed_fit_results(time_hr, flux_hr, flux_err_hr, data.wavelengths_hr, data.wavelengths_err_hr, samples_hr, map_params_hr, {"period": PERIOD_FIXED}, detrend_type_multiwave, f"{output_dir}/{instrument_full_str}_{hr_bin_str}", median_total_error_hr, gp_trend=gp_trend, spot_trend=spot_trend, jump_trend=jump_trend)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

