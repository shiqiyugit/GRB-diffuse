import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

def spl_flux(energy, norm, gamma):
    E=energy
    return (norm)*((E/(100000.))**-gamma)

def plot_diffuse_nu(ax, estes=False):
    print('here')
# Segmented fit with GF binning (new, June 16th)
    x = np.asarray([316.228, 1467.799, 3162.278, 6812.921, 14677.993, 31622.777, 68129.207, 146779.927,
                      316227.766, 681292.069, 1467799.268, 3162277.66, 6812920.691, 31622776.602])
    x_low = np.asarray([100.0, 1000.0, 2154.435, 4641.589, 10000.0, 21544.347, 46415.888, 100000.0,
                      215443.469, 464158.883, 1000000.0, 2154434.69, 4641588.834, 10000000.0,])
    x_high = np.asarray([1000.0, 2154.435, 4641.589, 10000.0, 21544.347, 46415.888, 100000.0, 215443.469,
                    464158.883, 1000000.0, 2154434.69, 4641588.834, 10000000.0, 100000000.0])
    x_err_low = x - x_low
    x_err_high = x_high - x
    # from fit
    tnorm = 1.e-8
    fitres={'cscd_piece0': 0.0, 'cscd_piece1': 0.0, 
            'cscd_piece2': 0.0, 'cscd_piece3':  3.149, 
            'cscd_piece4':  3.5, 'cscd_piece5': 4.433, 
            'cscd_piece6': 2.031, 'cscd_piece7': 1.814, 
            'cscd_piece8': 0.093, 'cscd_piece9': 0.855, 
            'cscd_piece10': 0.413, 'cscd_piece11': 0.0, 'cscd_piece12':0.016, 
            'cscd_piece13': 0.072}
    
    flux_central = [0.0, 0.0, 0.0, 	3.1054259733921152, 3.6228643920829797, 4.582023741602006, 2.0532411851950414, 1.8183859659755437	,
                    0.10411855881422984, 0.8459423855701095, 0.42584756278897035, 0.0, 0.015157580532533957, 0.07789995538115546]
    flux_lower = [0.0, 0.0, 0.0, 1.0896478477585938, 2.9071097404086768, 4.063582541142068, 1.6251341002189292, 1.4382602916177802,
                  0.0, 0.43881768724380293, 0.16205429697038198, 0.0, 0.0, 0.0]
    flux_upper = [27.095080522339288, 14.680829700272398, 3.7496160318164584, 4.937410655697931, 4.812082509105001, 5.123011002694284, 2.4413438490719273, 
                  2.2056715180845003, 0.4060860463971494, 1.2724773756766201, 0.771, 0.213, 0.087, 0.363]
    y=[]
    y_err_high=np.asarray(flux_upper)-np.asarray(flux_central)
    y_err_low=np.asarray(flux_central)-np.asarray(flux_lower)
    
            
    # print(Sigmas)
    indices_UL = np.where(np.asarray(flux_central)<0.02)[0]
    
    y = tnorm *np.asarray(flux_central)
    y_err_high = tnorm *np.asarray(y_err_high).flatten()
    y_err_low =tnorm * np.asarray(y_err_low).flatten()
    
    
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
    
    
    #fig = plt.figure(figsize=(6, 6), dpi=200) 
    #ax = plt.axes()
    
    ax.set_xscale('log')
    ax.set_yscale('log',nonpositive='clip')
    
    (_, caps, _) = ax.errorbar(x[indices_UL], y[indices_UL], xerr=[x_err_low[indices_UL], x_err_high[indices_UL]], yerr=y[indices_UL]*0.5, fmt='o',color='black', linewidth=2.0, zorder=30, uplims=True, markersize=0)
    (_, caps2, _) = ax.errorbar(x_new, y_new, xerr=[x_err_low_new, x_err_high_new], yerr=[y_err_low_new, y_err_high_new], fmt='o',color='black', label="CombinedFit (2024)", linewidth=2.0, zorder=31, markersize=0)
    
    x_bins =  np.logspace(2,8,12+1)[:-2]
    
    center = 10**((np.log10(x_bins[:-1]) + np.log10(x_bins[1:])) / 2)
    best_fits = center*center*spl_flux(center,[150.0, 0.0146663, 0.0000000, 13.331804534129844, 3.8569, 2.5965, 0.97, 1.02, 0, 0.82,],gamma=2)*10**-18
    onesigma_up = center*center*spl_flux(center,[150.0, 50.76989, 10.18, 3.67, 0.85, 0.427, 0.327, 0.49, 0.28, 1.04, ],gamma=2)*10**-18
    
    if estes:
        estes_uplims = best_fits < 1e-11
        ax.errorbar(center,best_fits, uplims=estes_uplims,
                     yerr=[onesigma_up,onesigma_up],
                     xerr=[center-x_bins[:-1],x_bins[1:]-center], 
                     fmt='o', color='tab:blue',alpha=0.5,capsize=3, lw=2.5, zorder=1000, label=r'ESTES (2023)')
        ax.errorbar(center[estes_uplims], best_fits[estes_uplims] + onesigma_up[estes_uplims], uplims=estes_uplims[estes_uplims],
                     yerr=[0.5*onesigma_up[estes_uplims],onesigma_up[estes_uplims]],
                     xerr=[(center-x_bins[:-1])[estes_uplims],(x_bins[1:]-center)[estes_uplims]],
                     fmt='', linestyle='',alpha=0.5,capsize=3, color='tab:blue', lw=2.5, zorder=1000)
        
        x_bins_hese =  np.logspace(np.log10(4.20*1e4),np.log10(7.67*1e6),7+1)
        center_hese = 10**((np.log10(x_bins_hese[:-1]) + np.log10(x_bins_hese[1:])) / 2)
        best_hese = (center_hese*center_hese)*(1e-18)*np.array([5.7, 3.89, 7.1e-2, 8.1e-3, 1.3e-2,  7.7e-4, 0.0e-5])/3.0
        best_up =   (center_hese*center_hese)*(1e-18)*np.array([9.5,  .99, 14e-2,  28e-3,   .86e-2, 17e-4,  6.7e-5])/3.0
        best_down = (center_hese*center_hese)*(1e-18)*np.array([5.7,  .87, 7.1e-2, 8.1e-3,  .64e-2, 7.7e-4, 0.0e-5])/3.0
        hese_uplims = best_hese < 1e-11
        ax.errorbar(center_hese,best_hese, uplims=hese_uplims,
                     yerr=[best_up,best_up],
                     xerr=[center_hese-x_bins_hese[:-1],x_bins_hese[1:]-center_hese], 
                     fmt='^', color='tab:red',alpha=0.5,capsize=3, lw=2.5, zorder=1000, label=r'HESE (2021)')


    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=[0.97, 0.99], loc='upper right', prop={'size':14}, fancybox=False, frameon=True, ncol=1)
    
    #ax.set_ylim(ymin = 4.e-10)
    #ax.set_ylim(ymax = 1e-6)
    #ax.set_xlim([1.e3, 1e8])
    
    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.yaxis.set_tick_params(right='on',which='both')

    yticks = ax.yaxis.get_majorticklocs()[2:-2]
    ax.set_yticks(yticks)
    
    ax.tick_params(axis='both', which='major', length=5, colors='0.0')
    ax.yaxis.set_ticks_position('both')
    ax.set_ylabel(r'$\mathrm{E_\nu^2\, \Phi_{\nu + \bar{\nu}}^{\mathrm{per\,flavor}} [GeV\, cm^{-2}\, s^{-1}\, sr^{-1}]}$',fontsize=16)
    ax.set_xlabel(r'Neutrino Energy [GeV]',fontsize=16)
    
#    plt.savefig('../plots/SegmentedFit_Comparsion_CombinedFit.pdf',dpi=300,bbox_inches='tight')
#    plt.show()
    
    
    
    
    
    
    
    
