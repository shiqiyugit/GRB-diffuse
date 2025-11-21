import numpy as np
import matplotlib.pyplot as plt
from loader import get_bat, get_data, get_gbm
from plots import rebin_data_custom_edges, rebin_loglog_data

eV2erg = 1.6e-12
erg2eV = 1/1.6e-12
#src_name = "060218"
#src_name = "201015A"
#src_name = "171205A"
#src_name = "120422A"

src_name = "100316D"
filename="data/bat_100316D_e2flux.dat"
x_bat, y_bat, x_err,  yerr, uplims = get_bat(src_name, filename=filename)
plt.errorbar(x_bat*1e-3, y_bat*1e-3*erg2eV, yerr=yerr*1e-3*erg2eV, xerr=x_err*1e-3, fmt='o', capsize=3, color='r', label='100316D')
x_bat, y_bat, x_err,  yerr, uplims = get_bat(src_name, filename="data/bat_100316D_rebinned.dat")
plt.errorbar(x_bat*1e-3, y_bat*1e-3*erg2eV, yerr=yerr*1e-3*erg2eV, xerr=x_err*1e-3, fmt='+', capsize=3, color='b', label='100316D')
'''
filename = "data/bat_120422A_flux.dat"
x_bat, y_bat, x_err,  yerr, y_ul= get_bat('120422A', filename=filename)
mask = y_ul.astype(bool)
plt.errorbar(x_bat*1e-3, y_bat*1e-3*erg2eV, xerr=x_err*1e-3, yerr=yerr*1e-3*erg2eV, fmt='o', capsize=3, color='c', label='120422A', markersize=8)

filename = "data/bat_190829A_flux.dat"
x_bat, y_bat, x_err,  yerr, y_ul= get_bat('190829A', filename=filename)
mask = y_ul.astype(bool)
plt.errorbar(x_bat*1e-3, y_bat*1e-3*erg2eV, xerr=x_err*1e-3, yerr=yerr*1e-3*erg2eV, fmt='o', capsize=3, color='b', label='190829A', markersize=8)

filename = "data/bat_171205A_flux.dat"
x_bat, y_bat, x_err,  yerr, y_ul= get_bat('171205A', filename=filename)
mask = y_ul.astype(bool)
plt.errorbar(x_bat*1e-3, y_bat*1e-3*erg2eV, yerr=yerr*1e-3*erg2eV, xerr=x_err*1e-3, fmt='o', capsize=3, color='g', label='171205A', markersize=8)
'''
'''
filename = "data/bat_060218_e2flux.dat"
x_bat, y_bat, x_err,  yerr, y_ul= get_bat('060218', filename=filename)
mask = y_ul.astype(bool)
# Plotting
plt.errorbar(x_bat*1e-3, y_bat*1e-3*erg2eV, yerr=yerr*1e-3*erg2eV, xerr=x_err*1e-3, fmt='o', capsize=3, color='g', label='060218', markersize=8)
x_bat, y_bat, x_err,  yerr, y_ul= get_bat('060218', filename='data/060218_xrt_wt.dat')
plt.errorbar(x_bat*1e-3, y_bat*1e-3*erg2eV, yerr=yerr*1e-3*erg2eV, xerr=x_err*1e-3, fmt='+', capsize=3, color='g', label='060218 xrt', markersize=8)
'''
'''
filename = "data/bat_201015A_flux.dat"
x_bat, y_bat, x_err,  yerr, y_ul= get_bat('201015A', filename=filename)
mask = y_ul.astype(bool)
# Plotting
plt.errorbar(x_bat*1e-3, y_bat*1e-3*erg2eV, xerr=x_err*1e-3, yerr=yerr*1e-3*erg2eV, fmt='o', capsize=3, color='g', label='201015A', markersize=8)
'''




plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy (keV)')
plt.ylabel('E^2 Flux (keV/cm²/s)')
plt.title('BAT Spectrum')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.legend()
plt.show()
plt.savefig('data/'+src_name+'_BAT_spec.pdf')

