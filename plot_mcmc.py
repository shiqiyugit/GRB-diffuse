import argparse
import numpy as np
import emcee
import corner
import json
import sys
import pandas
import scipy as sp
import AMES 
import math
import time
from functions import *
from loader import *
from fitter import *
from sources import *
from grb import GRBsinglezone
from run_fit import setup_run
import json
import ic_diffuse

if __name__ == '__main__':

    # Initialize parser
    parser = argparse.ArgumentParser(description='Example script with flags')

    # Add flag arguments
    parser.add_argument('--source', '-src', type=str, required=True, help='source name')
    parser.add_argument('--outdir', '-od', type=str,default='../LLGRB-result', help='Output directory path')
    parser.add_argument('--infile', '-if', type=str,default=None, help='input file to make plot')
    parser.add_argument('--burnin', '-bn', type=int, default=1)
    parser.add_argument('--thin', '-tn', type=int, default=1)
    parser.add_argument('--seed', '-i', type=int, default=None)
    parser.add_argument('--test_llh', '-tl', action='store_true')
    parser.add_argument('--external_shock', '-es', action='store_true')
    parser.add_argument('--best_fit', '-bfp', action='store_true')
    parser.add_argument('--calc_flux', '-cf', action='store_true')
    parser.add_argument('--shock_breakout', '-sbo', action='store_true')
    parser.add_argument('--config', '-cg', default=None)

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)

    src_name = args.source
    out_dir = args.outdir
    infile = args.infile
    burnin = args.burnin
    thin = args.thin
    #data_dict, src_list, models = setup_run(src_name, plot=True, config=["190829A", "201015A"])
    data_dict, src_list, models = setup_run(src_name, plot=True, config=config)
    #data_dict, src_list, models = setup_run(src_name, plot=True, config=None)
    if args.shock_breakout: src_name= "sbo_"+src_name

    f = fit(data_dict, nwalkers=None, seed=args.seed, src_param=src_list, models=models, out_name=src_name, folder=out_dir, test_llh = args.test_llh)
    #f.load_result(out_name=src_name, folder=out_dir, infile=infile, thin=thin, burnin=burnin, raw=True)
    #f.plot_mcmc_result(bfp=args.best_fit, burnin=burnin, out_name=src_name, folder=out_dir, raw=True)
    #f.plot_sed_result(models, out_name=src_name, folder=out_dir)
    #f.plot_llh(src_name, burnin=burnin, out_name=src_name, folder=out_dir, infile=infile, raw=True)
    #f.plot_sed_draft(models, out_name=src_name, folder=out_dir)
    #quit()
    #f.plot_sed_draft(models, out_name=src_name, folder=out_dir)
    #f.plot_data(out_name=src_name, folder=out_dir)

    # more plots for diffuse
    diffuse=True
    if (src_name == 'combine') or (src_name == 'sbo'):
        diffuse = True
    if diffuse:
        bfp = f.get_bfp(out_dir, src_name)
        
        '''
        result = models[0].model(bfp, s=s, cr=cr)
        # result 3 is E^2 * flux
        # Here, the input is neutrino total fluence
        neutrino_energy = result[2]

        # update by Bing 20250827
        neutrino_fluence = result[3] * (4*np.pi*(src_list[0]['dl'])**2) * src_list[0]['T_90'] #Total energy

        E2dNdE_nu = interp1d(neutrino_energy, neutrino_fluence*ERG_TO_EV, fill_value=0, kind='linear', bounds_error=False) # EV
        '''
        #using all to draw the band
        styles=['C0--','C1--','C2--','C3--', 'C4--', 'C5--', 'C6--', 'C7--']
        fluxes = []
        E_obs = np.logspace(12, 20, 25)
        fig, ax = plt.subplots()
        df = 'IS'
        if 'sbo' in src_name:
            df = 'SBO'

        for i, src in enumerate(src_list):
            print(i, src)
            s = AMES.Source()
            update_s(s)
            s.InitSource()

            # update by Bing 20250812
            #reset redshift to 0
            cr = AMES.CRDistribution(s)
            update_cr(cr)

            syn = AMES.Synchrotron(s)
            IC = AMES.InverseCompton(s)
            ph = AMES.Photonbackground(s)
            pm = AMES.Photomeson(s)
            pa = AMES.Photopair(s)
            models[i].src_param['z'] = 0
            result = models[i].model(bfp, s=s, cr=cr,syn=syn, IC=IC, ph=ph, pm=pm, pa=pa)
            neutrino_energy = result[2]
            neutrino_fluence = result[3] * (4*np.pi*(src['dl'])**2) * src['T_90'] *neutrino_energy**2 

            #check purpose
            """
            fig, ax = plt.subplots()
            ax.plot(neutrino_energy, result[3]*neutrino_energy**2*AMES.eV2erg * src['T_90'], '-')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim([1e11, 1e17])
            ax.set_ylim([1e-10, 1e-2])
            ax.set_xlabel(r'$E_\\nu [\\rm eV]$', fontsize=15)
            ax.set_ylabel(r"$E_\\nu^2 \\Phi_\\nu [\\mathrm{erg cm^{-2}}]$", fontsize=15)
            #ax1.set_yscale('log')
            #ax1.set_ylim([1e44, 1e52])
            #ax1.set_ylabel(r'$\\mathcal{E}_\\nu [\\rm erg$]', fontsize=15)
            ax.legend(frameon=False)
            plt.show()
            """

            E2dNdE_nu = interp1d(neutrino_energy, neutrino_fluence, fill_value=0, kind='linear', bounds_error=False) # EV
            _, _, flux = plot_diffuse_flux(E2dNdE_nu, src['z'], 10**src['logLiso'], Gamma0=bfp['Gamma_j'].value, E_obs = E_obs, style=styles[i], label=src['name']+"-style "+df, out_name=src_name, folder=out_dir, dl0=src['dl'])
            fluxes.append(flux)
            ax.loglog(1e-9*E_obs, 1e-9* np.array(flux), styles[i], label = src['name']+"-style")

        median_flux = np.mean(fluxes, axis=0)
        ax = plt.gca()
        ax.loglog(1e-9*E_obs, 1e-9* np.array(median_flux), 'k--', label = "median neutrino from "+df, linewidth=4)
        plot_diffuse_km3()
        #plot_diffuse_track()
        #plot_diffuse_cascade()
        ic_diffuse.plot_diffuse_nu(ax, True)
        ic_diffuse.plot_diffuse_nu(ax, False)
        plt.legend()
        plt.tight_layout()
        name = config['sources'][0]+"_"+config['sources'][1]
        #name = src_name
        plt.savefig(out_dir + '/' + name +'_'+df+'_diffuse_neutrino_flux.pdf')
        plt.show()

    if args.external_shock:
        bfp = f.get_bfp(out_dir, src_name)
        from external_shock import *
        print("start external shock")
        t_min = np.log10(1e1)
        t_max = np.log10(1e7)
        dT = 1e7-1e1
        time_array = np.logspace(t_min, t_max, 20)

        eta_k = 10 #convertian factor, Ek = eta_k E_giso
        #external_shock = GRB(Gamma0 = bfp['Gamma_j'].value, z = src_list[0]['z'], dl = src_list[0]['dl'], E_ej = (10**src_list[0]['logLiso'])*src_list[0]['T_90']*eta_k, L_ph = 0., time_array=time_array, calc_flux =args.calc_flux)
        external_shock = GRB(Gamma0 = 10, z = src_list[0]['z'], dl = src_list[0]['dl'], E_ej = 1e52, L_ph = 0., time_array=time_array, calc_flux =args.calc_flux)

        # get external shock fluence 
        fig, ax = plt.subplots()
        energy, fluence_fs, fluence_rs = external_shock.plot_neutrino_spectrum() #fluence in eV-1*cm-2

        ax = plt.gca()
        # get diffuse GeV/cm^-2/s^-1
        if diffuse:
           ax.loglog(1e-9*E_obs, 1e-9* np.array(median_flux), 'k--', label = "median IS neutrino", linewidth=4)
        # get spectral flux eV-1 cm-2 s-1
        #manually cutoff, need to update. use IS proton bf to run external shock? //syu
        avg_dNdE_nu_es_fs = interp1d(energy, fluence_fs * (1.+src_list[0]['z'])*4.*np.pi*(src_list[0]['dl']**2), fill_value=0, kind='linear', bounds_error=False) # need neutrino spectrum eV
        avg_dNdE_nu_es_rs = interp1d(energy, fluence_rs * (1.+src_list[0]['z'])*4.*np.pi*(src_list[0]['dl']**2), fill_value=0, kind='linear', bounds_error=False) # need neutrino spectrum eV
        _,_,_ = plot_diffuse_flux(avg_dNdE_nu_es_fs, src_list[0]['z'], 10**src_list[0]['logLiso'], style='C3--', label="ES (fs)", out_name=src_name, folder=out_dir, Gamma0=bfp['Gamma_j'].value,norm_L=False)
        _,_,_ = plot_diffuse_flux(avg_dNdE_nu_es_rs, src_list[0]['z'], 10**src_list[0]['logLiso'], style='C4:', label="ES (rs)", out_name=src_name, folder=out_dir, Gamma0=bfp['Gamma_j'].value,norm_L=False)

        plot_diffuse_km3() #need the diffuse version
        #plot_diffuse_track()
        #plot_diffuse_cascade()
        ic_diffuse.plot_diffuse_nu(ax, True)
        ic_diffuse.plot_diffuse_nu(ax, False)
        plt.legend()
        plt.tight_layout()

        B = (32*np.pi*0.01*10*AMES.mp*AMES.c_cnst**2)**0.5*14
        print('B = ', (32*np.pi*0.01*10*AMES.mp*AMES.c_cnst**2)**0.5*14)
        print('Epmax, syn = ', (6*np.pi*4.8e-10/(AMES.sigma_T*B*10))**0.5*(AMES.mp**2*AMES.c_cnst**2/AMES.me)*14*AMES.erg2eV/1e18)

        # averaged flux, can also use all fluxes to get band //syu
        # need to have km3 diffuse flux
        plt.xlim([1e2, 1e11])
        plt.ylim([1e-12, 1e-5])
        plt.xlabel('E [GeV]')
        plt.ylabel(r'$E^2 \Phi_{\nu} [\rm GeV \ cm^{-2}\ s^{-1}\ sr^{-1}]$')
        #plt.title("s")
        plt.loglog()
        plt.legend(loc=3, ncol=1, frameon=False)
        plt.savefig(out_dir+'/'+name+'_IS_RS_FS_neutrino_flux.pdf')
        plt.show()
