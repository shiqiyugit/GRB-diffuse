# Copyright to LL GRB team: Shiqi Yu and Bing Zhang

import numpy as np
import matplotlib.pyplot as plt
import os
import AMES
from functions import *
from scipy import integrate
from es_param import es_param
import copy
from grb import update_s_es, update_cr

source = AMES.Source()
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'r', 'g', 'b', 'c', 'y', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'r', 'g', 'b', 'c', 'y', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'r', 'g', 'b', 'c', 'y']

class GRB:
    #190829A
    def __init__(self, source=source, z = 0.0785, dl=300*AMES.Mpc, E_ej = 1.5e52,  L_ph=1e50, Gamma0=5, time_array=[1, 10, 100, 1000, 1500] , calc_flux = False, folder='../LLGRB-result/'):

        p = es_param
        new_pars = {'z':           z,
                    'dl':          dl,
                    #'dl':          luminosity_distance(z),
                    'E_ej':          E_ej,
                    'Gamma0':          Gamma0,
                    'L_ph':          L_ph,
                    }
        p.update(new_pars)
        self.p = p
        param = [p['z'], p['dl'], p['E_ej'], p['Gamma0'], p['n_ex'], p['k_ex'], p['spectral_e'],
                 p['epsilon_e'], p['fraction_e'], p['eta_acc_e'], p['epsilon_B'], p['open_angle'], p['view_angle'], p['gaussian_cone'], p['jet_index'], p['T_ej']]

        self.source = AMES.Source()
        update_s_es(self.source)
        self.source.InitSource()
        #self.grb = AMES.GRBAfterglowFSRSHadronic(self.source)
        #self.grb.setGRBAfterglowFSRSHadronicParam(param)
        self.grb = AMES.GRBAfterglow(self.source)
        self.grb.setGRBAfterglowParam(param)
        self.folder = folder
        # acc_p , eta_acc_p, spectral_p, L_ph     
        param_hadronic = [p['acc_p'], p['eta_acc_p'], p['spectral_p'], p['L_ph']]
        #self.grb.setGRBAfterglowFSRSHadronicParamHadronic(param_hadronic)
        self.grb.setGRBAfterglowParam(param_hadronic)

        RS_param = [p['spectral_e'], p['epsilon_e'], p['fraction_e'],  p['eta_acc_e'], p['epsilon_B_RS'], p['acc_p'], p['eta_acc_p'], p['spectral_p']]
        #self.grb.setGRBAfterglowFSRSHadronicRSParam(RS_param)
        self.grb.setGRBAfterglowRSParam(RS_param)

        print(param, param_hadronic, RS_param)

        self.time_array = time_array 
        self.energy_array_min = [1e9 / AMES.eV2Hz, 1e3, 1e11]
        self.energy_array_max = [1e9 / AMES.eV2Hz, 1e3, 1e11]

        #self.grb.setOutputFolder("../LLGRB-result")

        self.grb.haveAttenuSSA(True)
        self.grb.haveSSCSpec(False)
        self.grb.haveAttenuGGSource(False)
        #self.grb.haveNeutrino(True) 
        #self.grb.haveProtonSyn(True)
        self.grb.haveReverseShock(True)

        if calc_flux:
           calc_flux()
    def calc_flux(self):
        self.flux_vector = self.grb.Flux(ED, syn, IC, gg, ph, self.time_array, self.energy_array_min, self.energy_array_max)

    def calc_spectrum(self, T):
        self.T = T
        self.grb.haveSSCSpec(True)
        self.grb.haveAttenuSSA(True)
        self.grb.haveAttenuGGSource(True)
        self.grb.haveReverseShock(False)
        self.grb.showInfo(False)

        have_onezone=True
        print("calc spectrum")
        if have_onezone:
           self.grb.haveOneZone(have_onezone)
           #self.grb.haveEdgeEffect(False)
        self.flux_vector = self.grb.Spectrum(self.T)

    def plot_spectrum(self):
        fig, ax = plt.subplots()
        energy = np.array(self.source.getPhoton().getEnergy())

        self.spectrum_GS02(ax, self.p, self.T, energy, 'C7', label='GS02')
        #spectrum_afterglowpy(ax, self.p, self.T, energy, 'C8', label='afterglowpy')
        #spectrum_vegasafterglow(ax, self.p, self.p_RS, self.T, energy, 'C9', label='vegasafterglow')

        time = self.time_array
        ax.plot(energy, self.flux_vector[0], '--', lw=2, label='syn (FS)')
        ax.plot(energy, self.flux_vector[1], '--', lw=2, label='ssc (FS)')
        #ax.plot(energy, self.flux_vector[2], '--', lw=2, label='EIC (FS)')
        ax.plot(energy, self.flux_vector[3], ':', lw=2, label='syn (RS)')
        ax.plot(energy, self.flux_vector[4], ':', lw=2, label='ssc (RS)')
        #ax.plot(energy, self.flux_vector[6], '--', lw=2, label='EIC (RS)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1e-8, 1e15])
        ax.set_ylim([1e-15, 1e-3])
        ax.set_xlabel('E [eV]', fontsize=15)
        ax.set_ylabel(r'Flux $[\rm erg \ cm^{-2} \ s^{-1}]$', fontsize=15)
        ax.legend(loc=0, fontsize=7)
        plt.savefig(self.folder+'/spectrum.png')
        plt.show()
    def spectrum_GS02(self, ax, time_obs, i):
        from GRB_GS02 import GS02
        g = GS02(self.p['E_ej'], self.p['n_ex'], self.p['epsilon_e'],
                 self.p['epsilon_B'], self.p['spectral_e'], self.p['dl'], self.p['z'])
        tdays = time_obs / 3600 / 24.
        energy = np.array(source.getPhoton().getEnergy())
        nu = energy * AMES.eV2Hz
        spectrum = g.gen_spectrum(nu, tdays)*1e-3/AMES.erg2Jy*nu
        ax.plot(energy, spectrum, '-', c=colors[i], lw=2., alpha=0.5, label='Granot & Sari, 2002')

    def plot_photon_spectrum(self):
        data = np.loadtxt(self.folder + '/spectrum.dat')
        num = len(s.getPhoton().getMomentum())
        for i, x in enumerate(self.time_array):
            idx1 = num * i
            idx2 = num * (i + 1)
            ax.plot(data[idx1:idx2, 0], data[idx1:idx2, 1], '--', c=colors[i], lw=2, label=str(x) + ' s, EATS')
            #ax.plot(data[idx1:idx2, 0], data[idx1:idx2, 2], '-', c=colors[i], lw=2)

        for i, x in enumerate(self.time_array):
            self.spectrum_GS02(ax, self.time_array[i], i)
            self.spectrum_afterglowpy(ax, self.time_array[i], i)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1e-8, 1e15])
        ax.set_ylim([1e-27, 1e-5])
        ax.set_xlabel('E [eV]', fontsize=15)
        ax.set_ylabel(r'Flux $[\rm erg \ cm^{-2} \ s^{-1}]$', fontsize=15)
        #ax.legend(fontsize=7)
        plt.savefig(self.folder + '/photon_spectrum.pdf')
        plt.show()


    def plot_neutrino_flux(self):

        fig, ax = plt.subplots()

        data_fs = np.loadtxt(self.folder + '/spectrum_neutrino.dat')
        data = np.loadtxt(self.folder + '/spectrum_neutrino_RS.dat')

        num = len(source.getNeutrino().getMomentum())

        for i, x in enumerate(self.time_array):
            idx1 = num * i
            idx2 = num * (i + 1)
            ax.plot(data_fs[idx1:idx2, 0]*1e-9, 1e-9*data_fs[idx1:idx2, 1]*ERG_TO_EV, '--', c=colors[i], lw=2, label=str(int(x)) + ' s (FS)')
            ax.plot(data[idx1:idx2, 0]*1e-9, 1e-9*data[idx1:idx2, 1]*ERG_TO_EV, '-.', c=colors[i], lw=2, label=str(int(x)) + ' s (RS)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1e-8, 1e21])
        ax.set_ylim([1e-15, 1e-5])
        ax.set_xlabel('E [GeV]', fontsize=15)
        ax.set_ylabel(r'Flux $[\rm GeV \ cm^{-2} \ s^{-1}]$', fontsize=15)
        ax.legend(fontsize=7)
        plot_ps_km3(True)
        plt.savefig(self.folder+'/neutrino_flux_FS_RS.pdf')
        plt.show()

    def plot_neutrino_spectrum(self):

        ax = plt.gca()

        data_fs = np.loadtxt(self.folder + '/spectrum_neutrino.dat')
        data_rs = np.loadtxt(self.folder + '/spectrum_neutrino_RS.dat')

        num = len(self.source.getNeutrino().getMomentum())
        energies = data_fs[0:num, 0] #*1e-9 #161
        print(num, len(energies), len(data_fs), len(self.time_array))
        #integrate.simpson(spec, eps_nu)
        data_fs = data_fs.reshape([ len(self.time_array), num, 2])
        data_rs = data_rs.reshape([ len(self.time_array), num, 2])

        fluences_rs, fluences_fs = [], []
        for i in range(num):
            flux_rs = data_rs[:, i, 1]
            flux_fs = data_fs[:, i, 1]

            fluence_rs = integrate.simpson(flux_rs, self.time_array)
            fluence_fs = integrate.simpson(flux_fs, self.time_array)
            fluences_rs.append(fluence_rs)
            fluences_fs.append(fluence_fs)

        fluences_rs = np.array(fluences_rs)
        fluences_fs = np.array(fluences_fs)

        ax.loglog(energies*1e-9, fluences_rs*AMES.erg2eV*1e-9, '-', c='g', lw=2, label='RS neutrino')
        ax.loglog(energies*1e-9, fluences_fs*AMES.erg2eV*1e-9, '--', c='g', lw=2, label='FS neutrino')

        #ax.set_xscale('log')
        #ax.set_yscale('log')
        plot_ps_km3(plot_flux=False)

        ax.set_xlim([1e-8, 1e21])
        ax.set_ylim([1e-18, 1e-5])
        ax.set_xlabel('E [eV]', fontsize=15)
        ax.set_ylabel(r'Fluence $[\rm GeV \ cm^{-2} ]$', fontsize=15)
        ax.legend(fontsize=7)
        plt.show()
        # output fluence  
        return energies, fluences_fs*AMES.erg2eV, fluences_rs*AMES.erg2eV

    def plot_sub_spectrum(self, ax):

        #ax = plt.gca()
        
        data_fs = np.loadtxt(self.folder + '/spectrum_neutrino.dat')

        '''
        num = len(source.getPhoton().getMomentum())
        for i, x in enumerate(self.time_array):
            idx1 = num * i
            idx2 = num * (i + 1)
            ax.plot(data[idx1:idx2, 0]*1e-9, 1e-9*data[idx1:idx2, 1] *ERG_TO_EV, '--', c=colors[i], lw=2, label=str(int(x)) + ' s (sync)')
            ax.plot(data[idx1:idx2, 0]*1e-9, 1e-9*data[idx1:idx2, 2]*ERG_TO_EV, '-.', c=colors[i], lw=2, label=str(int(x)) + ' s (ssc)')
        '''
        
        data = np.loadtxt(self.folder + '/spectrum_neutrino_RS.dat')
        num = len(source.getNeutrino().getMomentum())
        energies = data[0:num, 0] #*1e-9 #161
        #source.getNeutrino().getMomentum()
        #integrate.simpson(spec, eps_nu)
        data_ts = data.reshape([ len(self.time_array), num, 2])
        data_fs_ts = data_fs.reshape([ len(self.time_array), num, 2])

        #print(data_ts.shape, data_ts[1,:,0])
        fluences, fluences_fs = [], []
        for i in range(num):
            flux = data_ts[:, i, 1] 
            flux_fs = data_fs_ts[:, i, 1]

            fluence = integrate.simpson(flux, self.time_array)
            fluence_fs = integrate.simpson(flux_fs, self.time_array)
            #fluence += fluence_fs
            fluences.append(fluence)
            fluences_fs.append(fluence_fs)

        fluences = np.array(fluences)
        fluences_fs = np.array(fluences_fs)

        """    
        for i, x in enumerate(self.time_array):
            idx1 = num * i
            idx2 = num * (i + 1)
            ax.plot(data[idx1:idx2, 0]*1e-9, data[idx1:idx2, 1] *ERG_TO_EV*1e-9, '-', c=colors[i], lw=2, label=str(int(x)) + ' s, EATS (nu)')
        """
        ax.plot(energies*1e-9, fluences*ERG_TO_EV*1e-9, '-', c='g', lw=2, label='RS neutrino')        
        ax.plot(energies*1e-9, fluences_fs*ERG_TO_EV*1e-9, '--', c='g', lw=2, label='FS neutrino')    

        for i, x in enumerate(self.time_array):
            self.spectrum_GS02(ax, self.time_array[i], i)
            self.spectrum_afterglowpy(ax, self.time_array[i], i)
        
  
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1e-8, 1e21])
        ax.set_ylim([1e-15, 1e-5])
        ax.set_xlabel('E [GeV]', fontsize=15)
        ax.set_ylabel(r'Fluence $[\rm erg \ cm^{-2} ]$', fontsize=15)
        ax.legend(fontsize=7)
        plt.savefig(self.folder+'/spectrum_neutrino_RS.pdf')
        plt.show()
        return energies, fluences*ERG_TO_EV/energies**2

    def plot_target_photon(self):
        fig, ax = plt.subplots()
        data = np.loadtxt(self.folder + '/test_density_fs.dat')
        time = self.time_array#()
        ax.plot(data[:, 0], data[:, 1], 'C0-')
        ax.plot(data[:, 0], data[:, 2], 'C1-')
        #energy = np.array(source.getTarget().getEnergy())
        #ax.plot(energy, spec*energy**2)
        #ax.set_xlim([1e-8, 1e21])
        #ax.set_ylim([1e-20, 1e10])
        ax.set_xlabel('E [eV]', fontsize=15)
        ax.set_ylabel('Target photon density')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1e-8, 1e15])
        ax.set_ylim([1e-17, 1e-5])
        plt.savefig(self.folder+'/target_photon.pdf')
        plt.show()

    def plot_flux(self):
        fig, ax = plt.subplots()

        data = np.loadtxt(self.folder + '/flux.dat')
        ax.plot(data[:, 0], data[:, 1], 'C0-')
        ax.plot(data[:, 0], data[:, 2], 'C1-')
        ax.plot(data[:, 0], data[:, 3], 'C2-')

        data = np.loadtxt(self.folder + '/flux_RS.dat')
        ax.plot(data[:, 0], data[:, 1], 'C0--')
        ax.plot(data[:, 0], data[:, 2], 'C1--')
        ax.plot(data[:, 0], data[:, 3], 'C2--')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1, 1e4])
        ax.set_ylim([1e-19, 1e-4])
        ax.set_xlabel('T [s]', fontsize=15)
        ax.set_ylabel(r'Flux $[\rm erg \ cm^{-2} \ s^{-1}]$', fontsize=15)
        ax.legend(fontsize=7)
        plt.savefig(self.folder+'/photon_flux.pdf')
        plt.show()

    def external_density(self):
        radius = np.logspace(12, 19, 81)
        n_ex = []
        #for r in radius:
        #    n_ex.append(self.grb.ExternalDensity(r))

        fig, ax = plt.subplots()
        ax.plot(radius, n_ex, '-', c='#3D59AB', lw=2.5)
        ax.set_xlabel(r'Radius [$\rm cm$]', fontsize=15)
        ax.set_ylabel(r'External medium density [$\rm~cm^{-3}$]', fontsize=15)
        ax.text(2e15, 30, r'$n_{\rm ex}(R) \propto \rm const$', fontsize=15)
        ax.text(4e16, 3e0, r'$n_{\rm ex}(R) \propto R^{-2}$', fontsize=15)
        ax.set_xlim([1e15, 1e18])
        ax.set_ylim([1e-1, 1e3])
        ax.loglog()
        plt.savefig('figure/external_density.pdf')
        plt.show()


    def plot_dynamic(self):
        #t_min = np.log10(1)
       # t_max = np.log10(1e10)
       # self.time_array = np.logspace(t_min, t_max, 91)
        self.grb.TestDynamic(self.time_array)

        fig, ax = plt.subplots()
        data = np.loadtxt(self.folder + '/dynamic.dat')
        ax.plot(data[:, 1], data[:, 2] * np.sqrt(1. - 1. / data[:,2] / data[:,2]), '-', lw=3, zorder=4)
        ax.plot(data[:, 3], data[:, 4] * np.sqrt(1. - 1. / data[:,4] / data[:,4]), '--', lw=3, zorder=4)

        data = np.loadtxt('refs/ref_Granot23_1.dat')
        ax.plot(data[:, 0], data[:, 1], 'bo')
        data = np.loadtxt('refs/ref_Granot23_2.dat')
        ax.plot(data[:, 0], data[:, 1], 'ro')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1e13, 1e19])
        ax.set_ylim([1e-1, 2e3])
        ax.set_xlabel('T [s]', fontsize=15)
        ax.set_ylabel(r'$\Gamma$', fontsize=15)
        ax.legend(fontsize=7)
        plt.show()

    def plot_dynamic2(self):
        #t_min = np.log10(1)
        #t_max = np.log10(1e11)
        #self.time_array = np.logspace(t_min, t_max, 91)
        self.grb.TestDynamic(self.time_array)

        fig, ax = plt.subplots()
        data = np.loadtxt(self.folder + '/dynamic.dat')
        ax.plot(self.time_array, data[:, 2] * np.sqrt(1. - 1. / data[:,2] / data[:,2]), '-', lw=3, zorder=4)

        t = self.time_array
        C_BM = np.sqrt(17.*self.p['E_ej']/(8*np.pi*self.p['n_ex']*AMES.mp*AMES.c_cnst**5))
        C_ST = 2./5*1.15*(self.p['E_ej']/(self.p['n_ex']*AMES.mp*AMES.c_cnst**5))**(1./5)
        GammaBeta2 = 0.5*C_BM**2*t**(-3) + 9./16*C_ST**2*t**(-6./5)
        Gamma = np.sqrt(GammaBeta2)
        """
        Beta = np.sqrt(1/(1+1/GammaBeta2))
        radius = []
        for t1 in t:
            I = quad(self.integrand, 0, t1, args=(C_BM, C_ST))
            radius.append(I[0])              
        """
        ax.plot(t, Gamma, '--', lw=2, label='Gamma')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1, 1e11])
        ax.set_ylim([1e-4, 2e3])
        ax.set_xlabel('T [s]', fontsize=15)
        ax.set_ylabel(r'$\Gamma$', fontsize=15)
        ax.legend(fontsize=7)
        plt.show()
 
if __name__ == "__main__":        
    t_min = np.log10(1e2)
    t_max = np.log10(1e5)
    time_array=np.logspace(t_min, t_max, 6)
    g = GRB(time_array=time_array,Gamma0=3, calc_flux=False)
    print('finished')
    g.calc_spectrum(1e2)
    g.plot_spectrum()
    #_, _ =g.plot_neutrino_spectrum()
    #g.plot_target_photon()
    #g.plot_neutrino_flux()
    #g.plot_dynamic()
    #g.plot_dynamic2()
    #g.external_density()
