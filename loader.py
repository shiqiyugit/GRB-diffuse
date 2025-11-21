import numpy as np
from astropy.table import Table
import pandas as pd

from plots import *

eV2erg = 1.6e-12

GRB_nu = {
'name': ['120422A', '171205A', '190829A', '201015A'],
'e_center': [1e12, 1e12, 1e12, 1e12],
'e_low' : [3e12, 20e12, 3e12, 10e12],
'e_high' : [3e15, 20e15, 3e15, 10e15],
'fluence' : [0.086e9, 0.14e9, 1.5e9, 0.059e9], # eV*cm-2
}
GRB_nu = pd.DataFrame(GRB_nu)

def get_data(GRB_name='190829A', fermi=True, bat=True, gbm=False, xlim=None, plot=False):
    x_combined = np.array([])
    y_combined = np.array([])
    xerr_combined = np.array([])
    yerr_hi_combined = []
    yerr_lo_combined = []
    yerr_combined = np.array([])
    is_upperlimit = np.array([], dtype=bool)

    grb_list = ['190829A', '171205A', '201015A', '100316D']

    if fermi & (GRB_name in grb_list):
        #print('loading fermi for ', GRB_name) 
        # Get Fermi data and mark as upper limits
        x_fermi, y_fermi, xerr_fermi, yerr_fermi, uplims = get_fermi(GRB_name, plot=plot)
        x_combined=np.append(x_combined, x_fermi)
        y_combined=np.append(y_combined, y_fermi)
        xerr_combined = np.append(xerr_combined, xerr_fermi)
        yerr_combined = np.append(yerr_combined, yerr_fermi[:,0])
        is_upperlimit=np.append(is_upperlimit, uplims)
    if bat:
        # Get BAT data (not upper limits)
        x_bat, y_bat, xerr_bat, yerr, uplims = get_bat(GRB_name, xlim=xlim, plot=plot)
        """
        y_bat /= x_bat**2
        yerr_hi /= x_bat**2
        yerr_lo /= x_bat**2
        x_bat, y_bat, xerr_bat, yerr_hi, yerr_lo = rebin_adjacent_counts(x_bat, y_bat, xerr_bat, yerr_hi, yerr_lo, 3)
        y_bat *=x_bat**2
        yerr_hi *=x_bat**2
        yerr_lo *=x_bat**2
        """
        #if len(newbin)>3:
        #    x_bat, y_bat, xerr_bat, yerr_hi, yerr_lo = rebin_data_custom_edges(x_bat, y_bat, xerr_bat, yerr_hi, yerr_lo, newbin)
        #else:
        #    x_bat, y_bat, xerr_bat, yerr_hi, yerr_lo = rebin_loglog_data(x_bat, y_bat, xerr_bat, yerr_hi, yerr_lo, newbin[0])
        x_combined=np.append(x_combined, x_bat)
        y_combined=np.append(y_combined, y_bat)
        xerr_combined=np.append(xerr_combined, xerr_bat)
        yerr_combined = np.append(yerr_combined, yerr)
        uplims = np.zeros(len(x_bat), dtype=bool)
        is_upperlimit=np.append(is_upperlimit, uplims)

    if gbm:
        x_bat, y_bat, xerr_bat, yerr_hi, uplims = get_gbm(GRB_name)
        """
        y_bat /= x_bat**2
        yerr_hi /= x_bat**2
        yerr_lo /= x_bat**2
        x_bat, y_bat, xerr_bat, yerr_hi, yerr_lo = rebin_adjacent_counts(x_bat, y_bat, xerr_bat, yerr_hi, yerr_lo, 3)
        y_bat *=x_bat**2
        yerr_hi *=x_bat**2
        yerr_lo *=x_bat**2
        """
        idx = y_bat>0
        x_bat=x_bat[idx]
        y_bat=y_bat[idx]
        x_err=xerr_bat[idx]
        yerr_hi=yerr_hi[idx]
        #yerr_lo=yerr_lo[idx]

        x_combined=np.append(x_combined, x_bat)
        y_combined=np.append(y_combined, y_bat)
        xerr_combined =np.append(xerr_combined, np.abs(xerr_bat))
        #yerr_lo_combined.append(yerr_lo)
        yerr_hi_combined.append(yerr_hi)
        is_upperlimit=np.append(is_upperlimit, np.zeros(len(y_bat), dtype=bool))

    #yerr_combined = np.concatenate(yerr_lo_combined)
    #yerr_combined = np.stack((yerr_lo_combined, yerr_hi_combined), axis=0)
    x_neutrino, y_neutrino, xerr_neutrino, yerr_neutrino = get_GRB_neutrino_fluence(GRB_name, False)
    #print(x_combined, y_combined, is_upperlimit)
    ret = {
        'x_gamma': x_combined,'y_gamma': y_combined,
        'xerr_gamma':xerr_combined, 'yerr_gamma':yerr_combined, 'isul_gamma':is_upperlimit, 
        'x_neutrino':x_neutrino, 'y_neutrino':y_neutrino, 'xerr_neutrino':xerr_neutrino, 'yerr_neutrino':yerr_neutrino}
    return ret

def get_fermi(src_name = '190829A', filename=None,plot=False):
    #print("loading fermi spectral data:", src_name)
    #print("returning: e_ctr, e2dnde, in unit of eV") 
    file_path = 'data/sed_'+src_name+'.npy'
    if filename:
        file_path = filename
    sed = np.load(file_path, allow_pickle=True).item()
    ebin_error =  np.abs(sed['e_max']*1e6 - sed['e_min']*1e6)
    ebin_ctr = sed['e_ctr']*1e6 # MeV to eV
    if plot:
        eflux=sed['e2dnde_ul']*1e6*AMES.eV2erg   #MeVcm^-2s^-1 to eV
        eflux_lo=sed['e2dnde_err_lo'] * 1e6*AMES.eV2erg  #MeV to eV
        eflux_hi=sed['e2dnde_err_hi']* 1e6*AMES.eV2erg  #MeV to eV
        eflux_ul = sed['e2dnde_ul']* 1e6*AMES.eV2erg  #MeV to eV
    else:
        eflux=sed['dnde_ul']*1e-6   #MeVcm^-2s^-1 to eV
        eflux_lo=sed['dnde_err_lo'] * 1e-6  #MeV to eV
        eflux_hi=sed['dnde_err_hi']* 1e-6  #MeV to eV
        eflux_ul = sed['dnde_ul']* 1e-6  #MeV to eV
    
    tss=sed['ts'] 
    uplims = tss<3
    eflux_hi[uplims]=eflux_ul[uplims] 
    eflux_lo[uplims]=0
    #yerr=[np.array(eflux_lo), np.array(eflux_hi)]
    yerr=np.stack((eflux_lo, eflux_lo), axis=1)
    uplims = np.ones(len(eflux_lo), dtype=bool)
    return (np.array(ebin_ctr), np.array(eflux), np.array(ebin_error), yerr, uplims)

def get_gbm(src_name = '190829A', xlim=[8e3, 30e3]):
        # x, dx, y, +err_hi, -err_lo
        filename = "data/gbm_190829A_n6.dat"

        # Read manually to skip header
        data = []
        with open(filename, "r") as f:
            for line in f:
                # Skip comments, headers, empty lines
                if line.strip().startswith("!") or line.strip() == "":
                    continue
                if line.strip().startswith("READ") or line.strip().startswith("@"):
                    continue
                parts = line.strip().split()
                if len(parts) == 5:
                    data.append([float(p) for p in parts])
        # negative values will be rebinned.

        data = np.array(data)*1e3
        if xlim is not None:
            data = data[(data[:, 0] > xlim[0]) & (data[:, 0] <= xlim[1])]

        for i in [3,2]: data[:,i] = data[:,i]*1e-3/data[:, 0]**2
        uplims = np.zeros(len(data[:, 0]), dtype=bool)
        #merged_data = rebin_merge_negative_flux(data)
        #return x, dx, y, dy
        return (data[:, 0],  data[:, 2], data[:, 1], data[:,3],  uplims)

def get_bat(src_name = '190829A', xlim=None, filename=None, plot=False):
    #print("loading BAT spectral data: ", src_name)
    #eeufspec
    if filename is None:
    	filename = "data/bat_"+src_name+"_flux.dat"
    if plot:
        filename = "data/bat_"+src_name+"_rebinned.dat"

    # Read manually to skip header
    data = []
    with open(filename, "r") as f:
        for line in f:
            # Skip comments, headers, empty lines
            if line.strip().startswith("!") or line.strip() == "":
                continue
            if line.strip().startswith("READ") or line.strip().startswith("@"):
                continue
            parts = line.strip().split()
            if len(parts) == 5:
                data.append([float(p) for p in parts])
    # negative values will be rebinned.

    data = np.array(data)
    for i in [0, 1]: data[:,i] *= 1e3
    if xlim is not None:
        data = data[(data[:, 0] > xlim[0]) & (data[:, 0] <= xlim[1])]

    if plot:
        for i in [3,2]: data[:,i] = data[:,i] * 1e3 *AMES.eV2erg
    else:
        for i in [3,2]: data[:,i] = data[:,i] * 1e-3 #eV2erg 
    uplims = np.zeros(len(data[:, 0]), dtype=bool)
    #merged_data = rebin_merge_negative_flux(data)
    return (data[:, 0],  data[:, 2], data[:, 1], data[:,3], uplims)


def get_GRB_neutrino_fluence(name, make_plot = False):
    
    # Data from the paper
    grb = GRB_nu[GRB_nu.name == name]
    energy_low = grb.e_low
    energy_high = grb.e_high
    energy_center = grb.e_center
    energy_err = energy_high - energy_low

    # Data from the paper
    flux_limit = grb.fluence * eV2erg  # Upper limit in eV cm^-2 
    spectral_index = -2  # Assumed E^-2 spectrum
    stat_err = np.sqrt(flux_limit**2)

    if make_plot:
        ax = plt.gca()
        # Plot the upper limit point
        plt.errorbar(energy_center, flux_limit, xerr=[[energy_center - energy_low], [energy_high - energy_center]], yerr=[[stat_err],[stat_err]], 
        fmt='o',color='black', linewidth=2.0, zorder=30, uplims=True, markersize=0,
        label='GRB {name} 90% CL Upper Limit')
        plt.plot(energy_center, flux_limit, marker=r'$\downarrow$', markersize=10, capsize=5,
         color='black', linestyle='none')
    return (energy_center, flux_limit, energy_err, stat_err)
