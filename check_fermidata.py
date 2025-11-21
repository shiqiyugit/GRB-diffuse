import numpy as np
import matplotlib.pyplot as plt
from loader import get_bat, get_data, get_gbm, get_fermi
from plots import rebin_data_custom_edges, rebin_loglog_data

eV2erg = 1.6e-12
erg2eV = 1/1.6e-12
src_name = "190829A"
filename = "data/sed_171205A.npy"#sed_190829A_new.npy"

x_bat, y_bat, x_err,  yerr, uplims = get_fermi(src_name, filename)

plt.errorbar(x_bat*1e-3, y_bat*1e-3*erg2eV, yerr=yerr.reshape((2,-1))*1e-3*erg2eV, xerr=x_err*1e-3, fmt='o', capsize=3, color='b', label='fixedall fermi data')
filename = "data/fermi_pm5ks/sed_171205A.npy"#sed_190829A_new.npy"

x_bat, y_bat, x_err,  yerr, uplims = get_fermi(src_name,filename)

plt.errorbar(x_bat*1e-3, y_bat*1e-3*erg2eV, yerr=yerr.reshape((2,-1))*1e-3*erg2eV, xerr=x_err*1e-3, fmt='o', capsize=3, color='r', label='free3deg fermi data')

#plt.errorbar(x_bat*1e-3, y_bat*1e-3*erg2eV, yerr=yerr.reshape((2,8))*1e-3*erg2eV, xerr=x_err*1e-3, fmt='o', capsize=3, color='b', label='new fermi data')

#x_bat, y_bat, x_err,  yerr, uplims = get_fermi(src_name)
#plt.errorbar(x_bat*1e-3, y_bat*1e-3*erg2eV, yerr=yerr.reshape((2,8))*1e-3*erg2eV, xerr=x_err*1e-3, fmt='o', capsize=3, color='b', label='old fermi data')
# Plotting
#plt.errorbar(x_bat*1e-3, y_bat*1e-3*erg2eV, yerr=[yerr_lo*1e-3*erg2eV, yerr_hi*1e-3*erg2eV], fmt='o', capsize=3, color='c', label='GBM data', markersize=8)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy (keV)')
plt.ylabel('E^2 Flux (keV/cm²/s)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.legend()
plt.show()
plt.savefig('data/'+src_name+'_fermi_spec.pdf')

