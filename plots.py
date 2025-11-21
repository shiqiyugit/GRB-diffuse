import numpy as np
import AMES 
import math
import time
import copy
from sources import theta_dict


colors = ['C0', 'C1', 'C2', 'C3']
gamma_colors = [
    "#e377c2", "#f7b6d2", "#c5b0d5",
    "#db7093", "#dda0dd", "#ee82ee", "#da70d6"]

blue_colors = ["#3182bd","#6baed6","#9ecae1","#08519c","#4292c6", "#74a9cf","#bdd7e7","#c6dbef","#a6cee3" ]

def plot_dynamic(grb):
    t_min = np.log10(1)
    t_max = np.log10(1e10)
    time_array = np.logspace(t_min, t_max, 91)
    grb.TestDynamic(time_array)

    fig, ax = plt.subplots()
    data = np.loadtxt('result/dynamic.dat')
    ax.plot(data[:, 1], data[:, 2] * np.sqrt(1. - 1. / data[:,2] / data[:,2]), '-', lw=3, zorder=4)
    ax.plot(data[:, 3], data[:, 4] * np.sqrt(1. - 1. / data[:,4] / data[:,4]), '--', lw=3, zorder=4)

    data = np.loadtxt('ref_Granot23_1.dat')
    ax.plot(data[:, 0], data[:, 1], 'bo')
    data = np.loadtxt('ref_Granot23_2.dat')
    ax.plot(data[:, 0], data[:, 1], 'ro')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e13, 1e19])
    ax.set_ylim([1e-1, 2e3])
    ax.set_xlabel('T [s]', fontsize=15)
    ax.set_ylabel(r'$\Gamma$', fontsize=15)
    ax.legend(fontsize=7)
    plt.show()

def rebin_data_custom_edges(x, y, dx, dy_hi, dy_lo, new_edges):
    """
    Rebin flux (erg cm⁻² s⁻¹) with asymmetric errors.
    Assumes errors are absolute (not per-unit-energy).
    """
    new_edges = np.asarray(new_edges)
    assert np.all(np.diff(new_edges) > 0), "Bin edges must be increasing"

    # Filter data within bin range
    in_range = (x >= new_edges[0]) & (x < new_edges[-1])
    x = x[in_range]
    dx = dx[in_range]
    y = y[in_range]
    dy_lo = dy_lo[in_range]
    dy_hi = dy_hi[in_range]

    # find the bin_ind of the old x in the new_eddges
    bin_indices = np.clip(
        np.searchsorted(new_edges, x, side='right') - 1,
        0, len(new_edges) - 2
    )

    x_new, y_new, dx_new, dy_lo_new, dy_hi_new = [], [], [], [], []
    new_edges = np.asarray(new_edges)
    assert np.all(np.diff(new_edges) > 0), "Bin edges must be increasing"

    for i in range(len(new_edges) - 1):
        mask = (bin_indices == i)
        new_width = (new_edges[i+1] - new_edges[i])/2

        if not np.any(mask):
            x_new.append((new_edges[i] + new_edges[i+1]) / 2)
            y_new.append(0.0)
            dy_lo_new.append(0.0)
            dy_hi_new.append(0.0)

            dx_new.append(new_width)
            continue

        total_y = max(0, np.sum(y[mask]))
        total_dy_lo = np.sqrt(np.sum(dy_lo[mask]**2)) 
        total_dy_hi = np.sqrt(np.sum(dy_hi[mask]**2))

        x_new.append((new_edges[i] + new_edges[i+1]) / 2)
        y_new.append(total_y)
        dy_lo_new.append(total_dy_lo)
        dy_hi_new.append(total_dy_hi)

        dx_new.append(new_width)
    
    return (
        np.array(x_new),
        np.array(y_new),
        np.array(dx_new),
        np.array(dy_hi_new), np.array(dy_lo_new))

def rebin_adjacent_counts(x, y, dx, dy_hi, dy_lo, N=2):
    """
    Rebin counts-like data (bins in log scale) by merging every N adjacent bins.
    -------
    x_new : array
        New bin centers (geometric mean of merged edges).
    y_new : array
        Rebinned counts (sums).
    dx_new : array
        New half-widths (linear scale).
    dy_hi_new, dy_lo_new : array
        Rebinned errors (quadrature sums).
    """
    n_bins = len(x)
    n_groups = n_bins // N  # drop leftovers if not divisible

    # Construct edges from centers + dx
    edges = np.zeros(n_bins + 1)
    edges[:-1] = x - dx
    edges[1:] = x + dx

    # Reshape into groups
    y = y[:n_groups * N].reshape(n_groups, N)
    dy_hi = dy_hi[:n_groups * N].reshape(n_groups, N)
    dy_lo = dy_lo[:n_groups * N].reshape(n_groups, N)

    # Grouped edges
    edges_grouped = edges[::N]
    if len(edges_grouped) <= n_groups:
        edges_grouped = np.append(edges_grouped, edges[-1])

    # New bin edges
    edges_new = edges_grouped
    x_new = np.sqrt(edges_new[:-1] * edges_new[1:])      # geometric mean
    dx_new = 0.5 * (edges_new[1:] - edges_new[:-1])      # half-width in linear

    # Combine counts and errors
    y_new = np.sum(y, axis=1)
    dy_hi_new = np.sqrt(np.sum(dy_hi**2, axis=1))
    dy_lo_new = np.sqrt(np.sum(dy_lo**2, axis=1))

    return x_new, y_new, dx_new, dy_hi_new, dy_lo_new


def rebin_loglog_data(x, y, dx, dy_hi, dy_lo, rebin_factor=2):
    """
    Rebin data where both x and y are logarithmic quantities.
    
    Parameters:
        x: Log-spaced energy values (keV)
        y: Log-spaced flux values
        dx: Bin widths in log space
        dy: Errors on log flux
        rebin_factor: How many original bins to combine
        
    Returns:
        x_new, y_new, dx_new, dy_new: Rebinned quantities
    """
    n_new = len(x) // rebin_factor
    x_reshaped = x[:n_new*rebin_factor].reshape(n_new, rebin_factor)
    y_reshaped = y[:n_new*rebin_factor].reshape(n_new, rebin_factor)
    dy_hi_reshaped = dy_hi[:n_new*rebin_factor].reshape(n_new, rebin_factor)
    dy_lo_reshaped = dy_lo[:n_new*rebin_factor].reshape(n_new, rebin_factor)

    dx_reshaped = dx[:n_new*rebin_factor].reshape(n_new, rebin_factor)

    # New x as geometric mean
    x_new = np.exp(np.mean(np.log(x_reshaped), axis=1))
    
    # New y as weighted mean in log space
    weights = 1/np.sqrt(dy_lo_reshaped**2 + dy_hi_reshaped**2)
    log_y = np.log(y_reshaped)
    y_new = np.exp(np.sum(log_y*weights, axis=1) / np.sum(weights, axis=1))
    
    # New error (weighted combination)
    dy_lo_new = np.sqrt(1/np.sum(1/dy_lo_reshaped**2, axis=1))
    dy_hi_new = np.sqrt(1/np.sum(1/dy_hi_reshaped**2, axis=1))
    # New dx (logarithmic width)
    log_edges = np.log(x) - np.log(dx)#/2
    log_edges = np.append(log_edges, log_edges[-1] + np.log(dx[-1]))  # Add final right edge

    # Select edges to match rebinning
    new_log_edges = log_edges[::rebin_factor]

    # If the number of bins doesn't divide perfectly, we need one more edge
    if len(new_log_edges) <= len(x) // rebin_factor:
        new_log_edges = np.append(new_log_edges, log_edges[-1])

    # Compute new dx
    dx_new = np.diff(np.exp(new_log_edges)/2)  # Now dx_new length == x_new

    return x_new, y_new, dx_new, dy_hi_new, dy_lo_new
def rebin_sed(energy, flux, flux_err, N_bin=None, bin_width=None):
    """
    对 SED 数据进行对数重分箱，同时返回每一个bin的宽度（energy误差）
    """
    if N_bin is not None and bin_width is not None:
        raise ValueError("N_bin 和 bin_width 只能提供其中之一.")
    if N_bin is None and bin_width is None:
        raise ValueError("需要提供 N_bin 或者 bin_width.")
        
    log_energy = np.log10(energy)

    if N_bin is not None:
        bin_edges = np.linspace(log_energy.min(), log_energy.max(), N_bin + 1)
    else:
        bin_count = int((log_energy.max() - log_energy.min()) / bin_width)
        bin_edges = np.linspace(log_energy.min(), log_energy.max(), bin_count + 1)

    bin_indices = np.digitize(log_energy, bin_edges)

    bin_center = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    weights = 1 / flux_err**2

    bin_flux = np.ones(len(bin_center)) * np.nan
    bin_error = np.ones(len(bin_center)) * np.nan
    
    for i in range(1, len(bin_center) + 1):
        mask = bin_indices == i
        if mask.any():
            w = weights[mask]
            wnorm = w / w.sum()
            bin_flux[i-1] = np.sum(wnorm * flux[mask])

            variance = np.sum(wnorm**2 * flux_err[mask]**2)
            bin_error[i-1] = np.sqrt(variance)

    new_energy = 10**bin_center
    new_energy_err = 0.5 * (10**bin_edges[1:] - 10**bin_edges[:-1])

    return new_energy, new_energy_err, bin_flux, bin_error
def generate_parameter_band(mo_func, samples, theta_keys, theta_dict, M, s, cr, syn, IC, ph, pm, pa):
    """
    Generate a spectral band by evaluating M points between parameter bounds.
    
    Args:
        mo_func: Your model function that takes parameters and returns spectra
        M: Number of parameter sets to evaluate between bounds
        
    Returns:
        Tuple of (energy_photon, min_spectrum, max_spectrum, 
                 energy_neutrino, min_neutrino, max_neutrino)
    """
    if M is not None and len(samples) > M:
        idx = np.random.choice(len(samples), M, replace=False)
        samples = samples[idx]

    all_photon_spectra = []
    all_neutrino_spectra = []

    for p in samples:
        theta_holder = copy.deepcopy(theta_dict)  # fresh copy for each evaluation

        # assign sample values to theta
        for val, k in zip(p, theta_keys):
            theta_holder[k].value = val

        # evaluate model
        result = mo_func(theta_holder, s, cr, syn, IC, ph, pm, pa)
        e_ph = result[0]
        s_ph = result[1]*result[0]**2*AMES.eV2erg
        e_nu = result[2]
        s_nu = result[3]*result[2]**2*AMES.eV2erg

        all_photon_spectra.append(s_ph)
        all_neutrino_spectra.append(s_nu)

    all_photon_spectra = np.array(all_photon_spectra)
    all_neutrino_spectra = np.array(all_neutrino_spectra)

    # Compute envelopes
    min_ph = np.min(all_photon_spectra, axis=0)
    max_ph = np.max(all_photon_spectra, axis=0)
    min_nu = np.min(all_neutrino_spectra, axis=0)
    max_nu = np.max(all_neutrino_spectra, axis=0)

    return e_ph, min_ph, max_ph, e_nu, min_nu, max_nu

def plot_spectral_band(ax, energy, min_spec, max_spec, color, label, alpha=0.3):
    """Helper function to plot a spectral band"""
    ax.fill_between(energy, min_spec, max_spec, color=color, alpha=alpha, label=label)
    # Plot median line
    median_spec = (min_spec + max_spec) / 2
    ax.plot(energy, median_spec, color=color, lw=1.5)

