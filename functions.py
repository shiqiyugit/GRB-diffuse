import numpy as np    
import pandas as pd
from scipy.interpolate import interp1d
import AMES

from scipy.integrate import dblquad
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
from scipy import integrate
import matplotlib.pyplot as plt
from astropy.constants import c
from astropy import units as u
# Define a power-law spectrum (example)

#Import some fun things
import numpy as np
import scipy as sci
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pickle
import math as m
from astropy.cosmology import LambdaCDM

import scipy.stats as stats
from scipy.stats import poisson
from scipy.stats import norm

from astropy.coordinates import Angle
from astropy.io import fits
import scipy.interpolate as spint
from ic_diffuse import *

ERG_TO_EV = 6.2415e+11
TEV_TO_EV = 1e12
LB = 1e47 *ERG_TO_EV
'''
L_s = np.logspace(np.log10(1e45), np.log10(1e49), 50)
Lb=1e47
alpha1=0.0
alpha2=3.5
term1 = (L_s / Lb) ** alpha1
term2 = (L_s / Lb) ** alpha2
term = term1+term2
dum = L_s*term
#for i, x in enumerate(eps_nu):
#    dum.append(x * spec[i])
A0 = 1 / integrate.simpson(dum, term)
print(A0)
'''

def dnde_pl_spectrum(eps_nu, lum_nu=1e46, index = -2.2, E0 = 1e14, norm=1): #input eV, not erg
    """Power-law neutrino spectrum dN/dE"""
    spec = (eps_nu/E0)**index
    return norm * (eps_nu/E0)**index 

def powerlaw(E, A, alpha):
    return A * E**(-alpha)

def cutoffpl(E, A, alpha, E_cut):
    return A * E**(-alpha) * np.exp(-E / E_cut)

def bknlaw(E, A, alpha1, alpha2, E_break):
    if E < E_break:
        return A * E**(-alpha1)
    else:
        return A * E_break**(alpha1 - alpha2) * E**(-alpha2)

def power_law_spectrum(eps_nu, norm=1, index=2.2, E0=1e14):
    """Power-law neutrino spectrum """
    # return eV cm-2 s-1 @ E0
    return norm * (eps_nu/E0)**(-1*index)

from matplotlib.ticker import LogFormatterSciNotation

def plot_ps_km3(plot_flux=False):
    
    # Data from the paper
    energy_low = 72e6  # 72 PeV in GeV (1 PeV = 1e6 GeV)
    energy_center = 220e6 #220PeV
    energy_high = 2.6e9  # 2.6 EeV in GeV (1 EeV = 1e9 GeV)

    
    # Data from the paper
    flux_limit = 1.2e-9  # Upper limit at 1 GeV in GeV^-1 cm^-2 s^-1
    spectral_index = -2  # Assumed E^-2 spectrum

    ax = plt.gca()
    t = 335 *24 *3600
    if plot_flux:
        plt.errorbar(energy_center, flux_limit * (energy_center)**(-spectral_index), xerr=[[energy_center - energy_low], [energy_high - energy_center]],uplims=True,
         color='black', markersize=30, label='KM3NeT 90% CL Upper Limit')

    else:
    # Plot the upper limit point
        plt.errorbar(energy_center, flux_limit * (energy_center)**(-spectral_index)*t, xerr=[[energy_center - energy_low], [energy_high - energy_center]],uplims=True,
         color='black', markersize=30, label='KM3NeT 90% CL Upper Limit')

def plot_diffuse_km3():

    # Data from the paper
    energy_low = 72e6  # 72 PeV in GeV (1 PeV = 1e6 GeV)
    energy_high = 2.6e9  # 2.6 EeV in GeV (1 EeV = 1e9 GeV)
    energy_center = 220e6 #220PeV

    flux_central = 5.8e-8  # Central value
    flux_err_low = 3.7e-8  # Lower error
    flux_err_high = 10.1e-8  # Upper error
    conf_95 = [0.30e-8, 29.8e-8]  # 95% CI
    conf_997 = [0.02e-8, 47.7e-8]  # 99.7% CI

    ax = plt.gca()

    # Plot confidence intervals as shaded regions
    plt.errorbar(energy_center, flux_central,
             xerr=[[energy_center - energy_low], [energy_high - energy_center]],
             yerr=[[flux_central- conf_997[0]], [conf_997[1]-flux_central]],lw=4, color='lightblue',alpha=0.5, label='99.7% CI')
    plt.errorbar(energy_center, flux_central,
             xerr=[[energy_center - energy_low], [energy_high - energy_center]],
             yerr=[[flux_central-conf_95[0]], [conf_95[1]-flux_central]],lw=4, alpha=0.5, color='skyblue', label='95% CI')

    # Plot central value with asymmetric error bars
    plt.errorbar(energy_center, flux_central, 
             xerr=[[energy_center - energy_low], [energy_high - energy_center]],
             yerr=[[flux_central-flux_err_low], [flux_err_high-flux_central]], lw=4, color='blue', alpha=0.5, label='Central value (90% CL)')

    #ax.set_yscale('log#')
    #ax.set_xlabel(r'Energy [GeV]', fontsize=12)
    #ax.set_ylabel(r'$E^2 \Phi(E)$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]', fontsize=12)
    #ax.set_title('KM3NeT ARCA Isotropic Flux (72 PeV - 2.6 EeV)\n335 days livetime', pad=20)

    # Customize ticks
    ax.xaxis.set_major_formatter(LogFormatterSciNotation()) 
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())

def plot_diffuse_track():
    energies = np.logspace(np.log10(15e3), 6.0, 100) # GeV input 
    diffuse, low, upper = diffuse_flux(energies)
    #print(np.log10(1e-9*E_nus), np.array(fluxes)*ERG_TO_EV*1e-9)
    plt.plot(energies, energies**2*diffuse, 'r-', linewidth=2)
    y_lower = energies**2*low
    y_upper = energies**2*upper
    plt.fill_between(energies, y_upper, y_lower, edgecolor='red',alpha=0.5, linewidth=0.5)

def plot_diffuse_cascade():
    indices_UL = np.array([5,8,9, 10, 11, 12])
    x, x_err_low, x_err_high, y, x_new, x_err_low_new, x_err_high_new, y_new, y_err_low_new, y_err_high_new = diffuse_cascade()
    ax = plt.gca()
    lw=2

    (_, caps, _) = ax.errorbar(x[indices_UL], y[indices_UL], xerr=[x_err_low[indices_UL], x_err_high[indices_UL]], yerr=y[indices_UL]*0.5, fmt='go', linewidth=lw+1.0, zorder=30, uplims=True, markersize=0)
    (_, caps2, _) = ax.errorbar(x_new, y_new, xerr=[x_err_low_new, x_err_high_new], yerr=[y_err_low_new, y_err_high_new], fmt='go', label=r'Diffuse $\nu_{e}/{\nu}_{\tau}$', linewidth=lw+1.0, zorder=31, markersize=0)
    (_, caps4, _) = ax.errorbar(x_new, y_new, yerr=[y_err_low_new, y_err_high_new], fmt='go',  linewidth=lw+1.0, zorder=31, markersize=0, capsize=0)
    for cap in caps:
        cap.set_markeredgewidth(2)

def diffuse_flux(energies = np.logspace(np.log10(15e3), 6.0+3.0, 100), E0=100e3):
    #Diffuse Flux at 100 TeV, input needs to be GeV, output needs to be GeV
    diff_norm = 1.44*1e-18 #TeV-1cm-2s-1sr-1 
    diff_flx = power_law_spectrum(energies,diff_norm,2.37, E0)
    diff_norm_up = (1.44+0.25)*1e-18
    diff_norm_lo = (1.44-0.26)*1e-18

    diff_flx_up = power_law_spectrum(energies,diff_norm_up,2.37+0.09, E0)
    diff_flx_up2 = power_law_spectrum(energies,diff_norm_up,2.37-0.09,E0)
    diff_flx_lo = power_law_spectrum(energies,diff_norm_lo,2.37-0.09,E0)
    diff_flx_lo2 = power_law_spectrum(energies,diff_norm_lo,2.37+0.09,E0)

    d_err_up = [max(i,j) for i,j in zip(diff_flx_up,diff_flx_up2)]
    d_err_lo = [min(x,y) for x,y in zip(diff_flx_lo,diff_flx_lo2)]
    return [diff_flx, d_err_lo, d_err_up]

def diffuse_cascade(indices_UL = np.array([5,8,9, 10, 11, 12])):
    dat = np.loadtxt('ic/segmented_flux_fitresult.txt', skiprows = 13)
    # GeV unit 
    x = dat[:,0]
    x_err_low = x - dat[:,1]
    x_err_high = dat[:,2] - x

    # from fit
    tnorm = 1.e-8 #*4*np.pi
    #convert unit to fov
    y = tnorm * dat[:,3]

    y_max = tnorm * dat[:,-1]
    y_min = tnorm * dat[:,-2]

    y_err_high = y_max - y
    y_err_low = y - y_min

    # visualize upper limits with corresponding marker
    y[indices_UL] = y[indices_UL]+y_err_high[indices_UL]

    uplims = np.zeros(x.shape)
    uplims[[indices_UL]]=True

    # plot the points with non-zero best fit (i.e. not upper-limits)
    x_new = np.delete(x, indices_UL)
    x_err_high_new = np.delete(x_err_high, indices_UL)
    x_err_low_new = np.delete(x_err_low, indices_UL)
    
    y_new = np.delete(y, indices_UL)
    y_err_high_new = np.delete(y_err_high, indices_UL)
    y_err_low_new = np.delete(y_err_low, indices_UL)

    return [x, x_err_low, x_err_high, y, x_new, x_err_low_new, x_err_high_new, y_new, y_err_low_new, y_err_high_new]

# isflux flag needs to be carefully checked. currently seems not used
def neutrino_flux(z_min, z_max, L_min, L_max, E_obs, E2dNdE_nu, sp, L_bol=1, ismodel=False, L_nu0=1e46, z0=0.1, Gamma=5, dl0=1., norm_L=True):
    """
    Calculate the diffuse neutrino flux from LL GRBs.
    
    Parameters:
    z_min, z_max (float): Redshift integration bounds
    L_min, L_max (float): Luminosity integration bounds
    
    Returns:
    float: The calculated neutrino flux E_nu^2 * Phi_nu
    """

    z_list = np.logspace(np.log10(z_min), np.log10(z_max), 40)
    L_list = np.logspace(np.log10(L_min), np.log10(L_max), 40)
    L_list = L_list * ERG_TO_EV
    L_bol = L_bol * ERG_TO_EV
    nu_calc = NeutrinoSpectrumCalculator(H0=67.3, Om0=0.315)

    eeflux = []
    for e_nu in E_obs:
        flux_z = []
        for z in z_list:
            dum = []
            for L in L_list:
                dum.append(luminosity_function(L) * nu_calc.estimate_nu_flux(z, E_obs = e_nu, L_bol=L, sp=sp, E2dNdE_nu=E2dNdE_nu, L_nu0=L_nu0, norm_L=norm_L, dl=dl0))
            flux_z.append(nu_calc.func_z(z) * redshift_distribution(z) * integrate.simpson(dum, L_list))

        integral = integrate.simpson(flux_z, z_list)
        eeflux.append(integral)

    return np.array(eeflux) 

# Example functions (you'll need to define these based on your specific model)
def luminosity_function(L, A0=300, Lb=LB, alpha1=0.0, alpha2=3.5, L_min = 1e45, L_max = 1e49):
    """
    Calculate the luminosity function for LL GRBs.
    
    Parameters:
    L (float or array): Luminosity value(s) in erg/s
    A0 (float): Normalization parameter
    Lb (float): Break luminosity in erg/s (default: 1e47)
    alpha1 (float): First power-law index (default: 0.0)
    alpha2 (float): Second power-law index (default: 3.5)
    
    Returns:
    float or array: Value(s) of the luminosity function in unit of Gpc^-3 yr^-1
    """
    term1 = (L / Lb) ** alpha1
    term2 = (L / Lb) ** alpha2
    Gpc = 3.08e27 
    yr = 365*24*3600
    A0 = A0 * Gpc**-3 * yr**-1

    return A0 / Lb / (term1 + term2)

def target_photon_spectrum(energy_eV, alpha, beta, epsilon_b, epsilon_min=1, epsilon_max=1e7, L_iso=1e49, Gamma=5, r=1.2E15):

    """
    Calculate the GRB prompt emission spectrum (broken power-law with exponential cutoff).

    Parameters:
    -----------
    energy_eV : array_like
        Photon energies in eV
    L_iso : float
        Isotropic-equivalent luminosity in erg/s
    Gamma : float
        Lorentz factor of the jet
    r : float
        Distance from central engine in cm
    epsilon_b : float, optional
        Break energy in eV (default: 500 eV)
    epsilon_min : float, optional
        Minimum energy in eV (default:  1 eV)
    epsilon_max : float, optional
        Maximum energy in eV (default:  10 MeV)

    Returns:
    --------
    dnde : array_like
        Differential photon number density in photons/eV/cm³
    """
    # Convert to eV units for calculation
    e = energy_eV  # Input already in eV
    L_iso *= ERG_TO_EV
  
    # Normalization factor (L_iso/5 is photon luminosity at epsilon_b)
    norm = (L_iso / 5) / (4 * np.pi * r**2 * Gamma**2 * c.cgs.value * epsilon_b**2)

    # Initialize spectrum
    dnde = np.zeros_like(e)

    # Lower energy band (index -1)
    mask_low = (e >= epsilon_min) & (e < epsilon_b)
    dnde[mask_low] = norm * (e[mask_low]/epsilon_b)**(alpha)

    # Higher energy band (index -2.2)
    mask_high = (e >= epsilon_b) & (e <= epsilon_max)
    dnde[mask_high] = norm * (e[mask_high]/epsilon_b)**(beta)

    # Exponential cutoff
    dnde *= np.exp(-e/epsilon_max)

    return dnde

def interpolated_flux(E, flux, e_low=1E-3, e_high=1E6):
    mask_E =(E>e_low)&(E<e_high)
    spflux = spint.RegularGridInterpolator(points = [np.array(E[mask_E])], values=flux[mask_E])
    return spflux

def redshift_distribution(z, eta=-10):
    #Sun et al. 2015 star formation history
    return ((1+z)**(3.4*eta)+((1+z)/5000)**(-0.3*eta)+((1+z)/9)**(-3.5*eta))**(1/eta)

def luminosity_distance(z, H0=67.3, Om0=0.315):
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    return cosmo.luminosity_distance(z).to('cm').value
    
class NeutrinoSpectrumCalculator:

    def __init__(self, H0=67.3, Om0=0.315):
        """
        Initialize neutrino spectrum calculator
        
        Parameters:
        H0 (float): Hubble constant in km/s/Mpc (default: 70)
        Om0 (float): Matter density parameter (default: 0.3)
        """
        self.H0 = H0
        self.Om0 = Om0
        self.Olambda = 1 - Om0  # Flat universe assumption

    def func_z(self, z):
        c = 2.99792458e5  # Speed of light in km/s
        Mpc = 3.08e24
        # Break down the equation into components
        prefactor = (c / (self.H0/Mpc))/ (1 + z)**2
        numerator = 4 * np.pi
        denominator = np.sqrt(self.Om0 * (1 + z)**3 + self.Olambda)

        return prefactor * numerator / denominator
    
    def dVc_dz(self, z):
        """
        Calculate comoving volume element dVc/dz
        
        Parameters:
        z (float or array): Redshift value(s)
        
        Returns:
        float or array: Comoving volume element in cm³
        """
        c = 2.99792458e5  # Speed of light in km/s
        Mpc = 3.08e24
        DL = luminosity_distance(z, self.H0, self.Om0)

        # Break down the equation into components
        prefactor = (c / (self.H0/Mpc))/ (1 + z)**2
        numerator = 4 * np.pi * DL**2
        denominator = np.sqrt(self.Om0 * (1 + z)**3 + self.Olambda)

        return prefactor * numerator / denominator
    
    def estimate_nu_flux(self, z, E_obs, E2dNdE_nu=dnde_pl_spectrum, Gamma=5, sp=None, L_bol=1, L_nu0=1e46, norm=1., norm_L=True, dl=1.):
        dL = dl #luminosity_distance(z)
        prefactor = Gamma*(1+z) / (4 * np.pi * dL**2)
        if sp is not None:
            #return prefactor * E_nu**2* dNdE_nu(E_nu,norm=norm, index=sp) #PL comoving frame
            return prefactor * dNdE_nu(E_obs,norm=norm, index=sp) #PL comoving frame
        else:
            spec = E2dNdE_nu(E_obs*(1+z))
            #norm = (L_bol / (L_nu0*ERG_TO_EV)) * (luminosity_distance(z0)/dL)**2
            if norm_L:
               norm = L_bol / (L_nu0*ERG_TO_EV)
            else:
               norm = norm
            #return spec * norm * E_nu**2
            return spec * norm 

    
def plot_diffuse_flux(E2dNdE_nu, z0, E_nu0, E_obs = np.logspace(12, 20, 25), L_range=[1e46, 1e50], z_range=[0.001, 5], style='b--', label="IS", out_name="combine", folder="../LLGRB-result", Gamma0=3, dl0=1., norm_L=True):
    #E_nus = np.logspace(np.log10(e_min), np.log10(e_max), 25) #eV
    
    ax = plt.gca()

    L_min = L_range[0]
    L_max = L_range[1]
    fluxes_IS = neutrino_flux(
        z_min=z_range[0], z_max=z_range[1],
        L_min=L_min, L_max=L_max,
        E_obs = E_obs, sp=None, Gamma=Gamma0,
        L_nu0=E_nu0, 
        E2dNdE_nu=E2dNdE_nu, z0=z0, dl0=dl0, norm_L=norm_L
    )

    ax.loglog(1e-9*E_obs, 1e-9* np.array(fluxes_IS), style, label = label+" neutrino", linewidth=2)
    plt.xlabel('Neutrino Energy (GeV)', fontsize=12)
    plt.ylabel(r'$E_\nu^2 \phi_\nu \ [\rm GeV \ cm^{-2} \ s^{-1}$]', fontsize=12)
    plt.title(f'Diffuse Neutrino Flux)', fontsize=14)
    plt.legend()
    ax.set_xlim([1e3, 1e11])
    ax.set_ylim([1e-13, 1e-6])
    #plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    #plt.savefig(folder + '/' + out_name+'_IS_diffuse_neutrino_flux.pdf')

    return ax, E_obs, fluxes_IS
    
def plot_luminosity_function():
    # plot luminosity function
    #plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()
    lumin = np.logspace(46, 55, 100)*ERG_TO_EV
    Gpc = 3.08e27 
    yr = 365*24*3600

    Lf = luminosity_function(lumin)*Gpc**3*yr

    rate_tot = integrate.simpson(Lf, lumin)
    print('rate_tot ', rate_tot)

    plt.plot(lumin/ERG_TO_EV, lumin*Lf, 'b-', linewidth=3, label='LL GRB')

    Lf_hl = luminosity_function(lumin, A0=10**-1.3, Lb=10**51.6*ERG_TO_EV, alpha1=0.5, alpha2=1.5, L_min = 1e48, L_max = 1e55)
    Lf_hl = Lf_hl * Gpc**3*yr
    print(lumin/ERG_TO_EV, lumin*Lf_hl)
    plt.plot(lumin/ERG_TO_EV, lumin*Lf_hl, 'C1--', linewidth=3, label='HL GRB')

    #EP240414a
    L_EP = 1.3e48 #peak luminosity
    rho_EP = [0.3-0.2, 0.3+0.7]
    y = np.array(np.linspace(rho_EP[0], rho_EP[1], 10))
    plt.plot(L_EP*y/y, y, 'k--', lw=2)

    #plt.scatter(df.logL
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([1e45, 1e55])
    ax.set_ylim([1e-3, 1e3])

    plt.ylabel(r'$log(L_{\gamma\,\rm iso}\Phi(z, L_{\gamma\,\rm iso})/[\rm erg~Gpc^{-3}yr^{-1}]$)', fontsize=12)
    plt.xlabel(r'$log(L_{\gamma\,\rm iso}/[erg\ s^{-1}$])', fontsize=12)
    plt.title(f'Luminosity Function', fontsize=14)
    #plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(frameon=False, fontsize=12, loc=0)
    plt.tight_layout()
    plt.savefig('figure/luminosity_function.pdf')
    plt.show()
    
    
if __name__ == "__main__":

    plot_luminosity_function()

    # Define parameters from the paper
    L_min = 1e45  # erg/s
    L_max = 1e49  # erg/s
    z = 0.0785

    Gamma = 3   # Lorentz factor
    
    # Calculate fluence of GRB 190829A from ANTARES/GCN 
    E_nus = np.logspace(np.log10(1e12), np.log10(1e20), 50) #eV

    # Plot
    
    
    plt.figure(figsize=(10, 6))
    neutrino_flux = nu_calc.estimate_nu_flux(0.05, E_nus)/ERG_TO_EV
    plt.plot(np.log10(1e-9*E_nus), np.log10(neutrino_flux), 'r-', lw=2)  
    plt.xlabel(r'$log(\rm Energy/GeV)$', fontsize=12)
    plt.ylabel(r'$log(\nu^2*\phi_\nu\/[\rm erg cm^{-2}])$', fontsize=12)
#    plt.title('GRB Prompt Emission Spectrum (Broken Power-Law)', fontsize=14)
    plt.legend()
    plt.grid(which='both', alpha=0.5)
    plt.xlim([1, 8])
    #plt.ylim([-12, -7])
    #plt.show()
    plt.savefig('nu_fluence.pdf')
    
        
    # Energy grid (1 eV to 10 MeV in eV)
    energy = np.logspace(1, 7, 500)  # eV
    L_iso = 1e49 #erg
    r=1.2e15 
    epsilon_b=500
    # Calculate spectrum
    spectrum = target_photon_spectrum(energy, L_iso, Gamma, r)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.loglog(energy, energy**2 * spectrum, 'r-', lw=2)  # Plot νFν in eV units
    plt.axvline(epsilon_b, color='k', ls='--', label=f'Break energy ({epsilon_b} eV)')
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel(r'$\nu F_\nu$ [arb. units]', fontsize=12)
    plt.title('GRB Prompt Emission Spectrum (Broken Power-Law)', fontsize=14)
    plt.legend()
#    plt.grid(which='both', alpha=0.5)
    plt.savefig('target_photon.pdf')
    

    # Calculate the flux
    #fluence_nu = power_law_spectrum(E_nus)
    fluxes_2 = []
    for ind in range(len(E_nus)):
        flux = neutrino_flux(
          z_min=0.001, z_max=5,
          L_min=L_min, L_max=L_max,
          E_nus = E_nus, E_ind = ind, sp = -2.2,
          dNdE_nu =  dnde_pl_spectrum
        )
        fluxes_2.append(flux)
    fluxes_3 = []
    for ind in range(len(E_nus)):
        flux = neutrino_flux(
            z_min=0.001, z_max=5,
            L_min=L_min, L_max=L_max,
            E_nus = E_nus, E_ind = ind, 
            sp = -2.5, dNdE_nu = dnde_pl_spectrum,
          ) 
        fluxes_3.append(flux)

       
    #print(f"Neutrino flux: {fluxes}")
        
    plt.figure(figsize=(10, 6))
    plt.plot(1e-9*E_nus, np.array(fluxes_2)*1e-9, 'b-', linewidth=2, label=r'$\gamma=-2.2$')
    plt.plot(1e-9*E_nus, np.array(fluxes_3)*1e-9, color='gray',linestyle='-', linewidth=2, label=r'$\gamma=-2.5$')
    #print(1e-9*E_nus, E_nus**2*np.array(fluxes_3)*1e-9)
    
    # diffuse track
    plot_diffuse_track()

    # diffuse casacde data points
    plot_diffuse_cascade()

    # plot diffuse km3 data point
    plot_diffuse_km3()

    plt.xlabel('Neutrino Energy [GeV]', fontsize=12)
    plt.ylabel(r'$E_\nu^2 \phi_\nu$ [$\rm GeV \ cm^{-2} \ s^{-1} \ sr^{-1}$]', fontsize=12)
    plt.title(f'Diffuse Neutrino Flux (Γ={Gamma})', fontsize=14)
    #plt.grid(True, which="both", ls="--", alpha=0.5)
    ax=plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlim([1e3, 1e11])
    #plt.ylim([1e-12, 1e-6])
    plt.legend()
    plt.tight_layout()
    plt.savefig('diffuse_neutrino_flux.pdf')
    plt.show()
    
    
