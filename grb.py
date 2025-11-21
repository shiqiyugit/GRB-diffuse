import argparse
import numpy as np
import emcee
import corner
import sys
import pandas
import scipy as sp
import AMES 
import math
import time
from functions import *
from loader import *
from sources import *
import copy
from functions import luminosity_distance
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

PI = AMES.PI
c_cnst = AMES.c_cnst
erg2eV = AMES.erg2eV
eV2erg = AMES.eV2erg

"""
Leptohadronic model for GRB prompt emission
"""

'''
Set particle momentum grid: total number of grids, minimum momentum, maximum momentum
The binning in log10 space is 0.05 or smaller to keep accuracy
Warning: please set same binning for photons and electrons if Klein-Nishina effect is important
num_p, min_p, max_p = 101, 1e9, 1e19
'''
def update_s_es(s):
 A, Z = 1, 1
 s.setNucleus(A, Z)
 num_p, num_muon, min_p, max_p = 141, 241, 1e7, 1e21
 s.getTarget().setMomentum(151, 1e-7, 1e8)
 s.getPhoton().setMomentum(281, 1e-7, 1e21)
 s.getElectron().setMomentum(151, 1e4, 1e21)
 s.getNeutrino().setMomentum(num_p, min_p, max_p)
 s.getNeutron().setMomentum(num_p, min_p, max_p)
 s.getProton().setMomentum(num_p, min_p, max_p)
 s.getMuonminusL().setMomentum(num_muon, min_p, max_p)
 s.getMuonminusR().setMomentum(num_muon, min_p, max_p)
 s.getMuonplusL().setMomentum(num_muon, min_p, max_p)
 s.getMuonplusR().setMomentum(num_muon, min_p, max_p)
 s.getPionminus().setMomentum(num_p, min_p, max_p)
 s.getPionplus().setMomentum(num_p, min_p, max_p)
 s.getPionzero().setMomentum(num_p, min_p, max_p)
 s.getNucleus().setMomentum(num_p, min_p, max_p)
 s.getParam().setGeometry("shell")

def update_s(s):
 A, Z = 1, 1
 s.setNucleus(A, Z)
 """
 num_p, num_muon, min_p, max_p = 121, 241, 1e7, 1e19
 s.getTarget().setMomentum(121, 1e-2, 1e8)
 s.getPhoton().setMomentum(211, 1e-2, 1e19)
 s.getElectron().setMomentum(151, 1e4, 1e19)
 #s.getPhoton().setMomentum(161, 1e-2, 1e14)
 #s.getElectron().setMomentum(101, 1e4, 1e14)
 """
 #added by Bing 9/30
 num_p, num_muon, min_p, max_p = 121, 241, 1e7, 1e19
 s.getTarget().setMomentum(121, 1e-7, 1e8)
 s.getPhoton().setMomentum(211, 1e-7, 1e19)
 s.getPhotonSyn().setMomentum(211, 1e-7, 1e19)
 s.getPhotonIC().setMomentum(211, 1e-7, 1e19)
 s.getPhotonPM().setMomentum(211, 1e-7, 1e19)
 s.getPhotonPA().setMomentum(211, 1e-7, 1e19)
 s.getElectron().setMomentum(151, 1e4, 1e19)
 s.getElectronPrim().setMomentum(151, 1e4, 1e19)
 s.getElectronPM().setMomentum(151, 1e4, 1e19)
 s.getElectronPA().setMomentum(151, 1e4, 1e19)
 s.getNeutrino().setMomentum(num_p, min_p, max_p)
 s.getNeutron().setMomentum(num_p, min_p, max_p)
 s.getProton().setMomentum(num_p, min_p, max_p)
 s.getMuonminusL().setMomentum(num_muon, min_p, max_p)
 s.getMuonminusR().setMomentum(num_muon, min_p, max_p)
 s.getMuonplusL().setMomentum(num_muon, min_p, max_p)
 s.getMuonplusR().setMomentum(num_muon, min_p, max_p)
 s.getPionminus().setMomentum(num_p, min_p, max_p)
 s.getPionplus().setMomentum(num_p, min_p, max_p)
 s.getNucleus().setMomentum(num_p, min_p, max_p)
 s.getParam().setGeometry("shell")

def update_cr(cr, have_proton=True, have_timescale=False):
 cr.HaveMomentumSpace(True) #[default]
 cr.HaveSource(True) # if False, particle will evolve without source term
 cr.HaveElectron(True) # [default]
 cr.HaveNeutrino(True) # [default]
 cr.HaveEscape(True) # if False, no escape for both charged particles and neutral particles
                    # if True, no escape for charged particles, t_esc = R/c for neutral particles
 cr.HaveAdiabatic(True) # if False, no adiabatic cooling [default, t_ad = clt]
 cr.HaveSynchrotron(True) # if True, synchrotron process included [All charged particles]
 cr.HaveInverseCompton(True) # if True, inverse-Compton process included [Electrons]
 cr.HaveGammaGamma(True) # if True, gamma-gamma pair production process include [Photons]
 
 cr.HaveEBLAttenuation(True) # if True, EBL attentuation process for photons included
 cr.HaveTargetUpdate(True) # if True, target photons updated at each time step
 cr.HaveSteadyState(True) # if True, AMES will output steady-state results. If False, AMES will output results at t = clt
 cr.OutputTimescale(have_timescale)
 cr.OutputResult(False)
 cr.OutputComponent(have_timescale)
 cr.setOutputFolder('../LLGRB-result/')

 cr.HaveProton(have_proton) # [default]
 cr.HaveMuon(have_proton)  # if False, no pion and muon cooling 
                    # if True, pion and muon decay (and its cooling ?) are included
 cr.HaveDecay(have_proton) # if True, neutron decay included [secondary proton included, except secondary neutrino and electron]
 cr.HavePhotopair(have_proton) # if True, Bethe-Heitler pair production process included
 cr.HavePhotopairSecondary(have_proton) # if True, secondaries from Bethe-Heitler pair production process included
 cr.HavePhotomeson(have_proton) # if True, Photomeson process included
 cr.HavePhotomesonSecondary(have_proton) # if True, secondaries from Photomeson process included
 
class GRBsinglezone:
      """
      Hadronic model
      """
      def __init__(self, src_param):
          s = AMES.Source()
          update_s(s)
          s.InitSource()

          self.src_param = src_param
          self.shared_param = self.src_param['shared_param']
          self.L_gamma = 10**self.src_param['logLiso']

          self.energy_proton = np.array(s.getNucleus().getEnergy())
          self.energy_electron = np.array(s.getElectron().getEnergy())
          self.energy_photon = np.array(s.getPhoton().getEnergy())
          self.energy_neutrino = np.array(s.getNeutrino().getEnergy())

      def test_model(self, theta, s, cr):
          #print("running test model")
          spectral_p = theta['spectral_p'].value
          log_xi_p = theta['log_xi_p'].value
          energy_photon = self.energy_photon
          #spectrum_photon = 1e-8 * (energy_photon/1e9)**(2-spectral_p) * 10**log_xi_p
          return [energy_photon, spectrum_photon, energy_photon, spectrum_photon]

      def model(self, theta, s, cr, syn, IC, ph, pm, pa, have_proton=True):

          s.getParam().setRedshift(self.src_param['z'])
          s.getParam().setLumDistance(self.src_param['dl'])

          log_xi_B = theta['log_xi_B'].value
          log_xi_e = theta['log_xi_e'].value
          log_Radius = theta['log_Radius'].value
          Gamma_j = theta['Gamma_j'].value
          Gamma_j_z = Gamma_j/(1+self.src_param['z'])
          spectral_e = theta['spectral_e'].value
          #gamma_e_min = 10**theta['log_gamma_e_min'].value
          eps_e = theta['eps_e'].value
          log_xi_p = theta['log_xi_p'].value
          spectral_p = theta['spectral_p'].value
          eta = theta['eta'].value
          #spectral_p = spectral_e

          s.getParam().setDopplerFactor(Gamma_j)
          beta_j = np.sqrt(1 - 1./Gamma_j**2)
          s.getParam().setVelocity(beta_j)
          
          r_diss = 10**log_Radius
          s.getParam().setEmissionRadius(r_diss)
          s.getParam().setShellWidth(r_diss / Gamma_j)  ##comoving frame shell width
          utility = AMES.Utility()

          #input
          #Luminosity, E_pk_obs
          eps_min = 0.1   #comoving   
          eps_max = 1e7   #comoving
          eps_pk_obs = self.src_param['Epk_obs']  #obs
          alpha = self.src_param['alpha'] ## dN/(dtde) \propto e^(-alpha) or e^(-beta) , unit is number/energy/time 
          try:
              beta = self.src_param['beta'] ## dN/(dtde) \propto e^(-alpha) or e^(-beta) , unit is number/energy/time 
          except:
              beta = 2.5
              #print('not using band function')
          u_ph = self.L_gamma/4/PI/r_diss**2/Gamma_j**2/AMES.c_cnst* AMES.erg2eV  #to comoving, unit is eV/cm^-3
          eps_pk = eps_pk_obs / Gamma_j_z
          energy = np.array(s.getTarget().getEnergy())
          #ph = AMES.Photonbackground(s)
          if self.src_param['func_form'] == 'CPL':
            ph.Powerlaw(eps_min, eps_pk, alpha, u_ph)
          elif self.src_param['func_form'] == 'BPL':
            ph.BrokenPowerlaw(eps_min, eps_pk, eps_max, alpha, beta, u_ph)
          elif self.src_param['func_form'] == 'Band':
            ph.BandFunction(eps_min, eps_pk, eps_max, alpha, beta, u_ph)  # You can use Band function.
          else:
            print('Value Error!')
          spectrum = np.array(s.getTarget().getSpectrum())
         
          #renormalize the target photon for give energy range
          dum = energy * spectrum
          energy = energy.tolist()
          dum = dum.tolist()
          x_begin = self.src_param['E_min'] / Gamma_j_z
          x_end   = self.src_param['E_max'] / Gamma_j_z
          norm = utility.Integrate(energy, dum, x_begin, x_end)
          spectrum *= u_ph / norm


          L_BB = 3e46
          T_BB = 3e5
          E_BB = L_BB / (4 * np.pi * r_diss**2 * Gamma_j**2 * AMES.c_cnst)*AMES.erg2eV
          ph.GreyBody(T_BB, E_BB)
          spectrum_BB = np.array(s.getTarget().getSpectrum())
          for i, x in enumerate(energy):
              s.getTarget().setSpectrum(i, spectrum[i] + spectrum_BB[i]) 
          spectrum = np.array(s.getTarget().getSpectrum())

          #Output flux band
          energy = np.array(s.getTarget().getEnergy())
          flux_Band = energy**2 * spectrum * r_diss**2 * Gamma_j_z**2 *AMES.c_cnst*AMES.eV2erg / self.src_param['dl']**2 # observed
          energy_obs = energy * Gamma_j_z
          self.flux_internal = flux_Band
          self.energy_obs = energy_obs

          xi_B = 10**log_xi_B
          Bprime =np.sqrt(8.0*PI*u_ph* eV2erg*xi_B) # fomulation by xi_B = U_B/U_ph.
          cr.setMagStrength(Bprime)

          xi_p = 10**log_xi_p
          u_p = u_ph*xi_p
          eps_p_min = 10 * AMES.proton_mass # proton comoving minimum energy (we fixed to gamma_p = 10)

          t_dyn = r_diss / Gamma_j / beta_j / c_cnst #dynamical time
          '''
          syn = AMES.Synchrotron(s)
          IC = AMES.InverseCompton(s)
          pm = AMES.Photomeson(s)
          pa = AMES.Photopair(s)
          '''
          if have_proton:
              # Calculate proton maximum energy
              ## loss timescale
              pm.Losstime(s.getProton())
              losstime_pm = s.getProton().getLosstime()

              pa.Losstime(s.getProton())
              losstime_pa = s.getProton().getLosstime()     

              syn.LosstimeProton(Bprime)
              losstime_syn_p = s.getProton().getLosstime()
              
              # Adiabatic loss timescale
              losstime_adi = np.full_like(self.energy_proton, 1/t_dyn, dtype=float)

              # Acceleration timescale (Bohm limit, true timescale not inverse)
              t_acc = (self.energy_proton * eV2erg) / (eta * AMES.e_charge * Bprime * c_cnst)

              # Total proton loss timescale
              t_loss = np.array(losstime_pa) + np.array(losstime_pm) + np.array(losstime_syn_p) + losstime_adi

              # Difference (in log-space to treat power laws properly)
              diff = np.log(t_acc) - np.log(t_loss)

              # Interpolate the difference as a function of proton energy
              f_diff = interp1d(self.energy_proton, diff, kind='linear', fill_value="extrapolate")

              # Minimize |diff| to find crossing (where t_acc ~ t_loss), replacement of interpolation
              res = minimize_scalar(lambda E: abs(f_diff(E)),
                       bounds=(self.energy_proton.min(), self.energy_proton.max()),
                       method='bounded')

              if res.success:
                 eps_p_max = res.x
              else:
                 # Fallback: pick closest grid point
                 eps_p_max = self.energy_proton[np.argmin(np.abs(diff))]

              momentum_min_p, momentum_max_p, u_inj, isEne = \
                  s.getProton().Energy2Momentum(eps_p_min), \
                  s.getProton().Energy2Momentum(eps_p_max), \
                  u_p / t_dyn, True
              s.getProton().setSpectrumPL(momentum_min_p, momentum_max_p, spectral_p, u_inj, isEne)

          # Electron
          # Calculate electron maximum energy
          ## loss timescale
          syn.Losstime(Bprime)
          losstime_syn = s.getElectron().getLosstime()

          IC.Losstime()
          losstime_IC = s.getElectron().getLosstime()     

          # Adiabatic loss timescale
          losstime_adi = np.full_like(self.energy_electron, 1/t_dyn, dtype=float)

          # Acceleration timescale (Bohm limit, true timescale not inverse)
          t_acc = eta*AMES.e_charge*Bprime*c_cnst/(self.energy_electron *eV2erg)

          # Total proton loss timescale
          t_loss = np.array(losstime_syn)+np.array(losstime_IC)+np.array(losstime_adi)

          # Difference (in log-space to treat power laws properly)
          diff = np.log(t_acc) - np.log(t_loss)

          # Interpolate the difference as a function of proton energy
          f_diff = interp1d(self.energy_electron, diff, kind='linear', fill_value="extrapolate")

          # Minimize |diff| to find crossing (where t_acc ~ t_loss), replacement of interpolation
          res = minimize_scalar(lambda E: abs(f_diff(E)),
                   bounds=(self.energy_electron.min(), self.energy_electron.max()),
                   method='bounded')

          if res.success:
             eps_e_max = res.x
          else:
             # Fallback: pick closest grid point
             eps_e_max = self.energy_electron[np.argmin(np.abs(diff))]

          xi_e = 10**log_xi_e
          u_e = u_ph*xi_e
          Gamma_sh=theta['Gamma_sh'].value
          if spectral_e == 2:
              A = eps_e * (AMES.proton_mass/AMES.electron_mass) * (Gamma_sh - 1)
              eps_e_min = A / np.log(eps_e_max/AMES.electron_mass / A) *AMES.electron_mass
          else:
              eps_e_min = eps_e*((spectral_e-2)/(spectral_e-1))*(AMES.proton_mass)*(Gamma_sh -1) #gamma_e_min * AMES.electron_mass
          # Electron injection
          momentum_min_e, momentum_max_e, u_inj_e, isEne = \
              s.getElectron().Energy2Momentum(eps_e_min), \
              s.getElectron().Energy2Momentum(eps_e_max), \
              u_e / t_dyn, True
          s.getElectron().setSpectrumPL(momentum_min_e, momentum_max_e, spectral_e, u_inj_e, isEne)

          #run the code
          clt = t_dyn # same as dynamical timescale
          minimum_time_step, maximum_time_step, steady_state_error, t_stop = 1e-1, 1e-1, 5e-2, 7*clt

          #reset target photon density to 0
          #for i in range(len(s.getTarget().getSpectrum())):
          #    s.getTarget().setSpectrum(i, 0.0)

          show_info, show_monitor = False, False

          cr.setOutputFolder('../LLGRB-result/' + self.src_param['name'])
          cr.LeptohadronicFullCascade(clt, minimum_time_step, maximum_time_step, steady_state_error, t_stop, show_info, show_monitor)

          #output result
          #here we only return photon and neutrino
          try:
              flux_vector = AMES.VecVecdouble()
              cr.getFlux(ph, flux_vector)
          except:
              flux_vector = cr.getFlux(ph)
          spectrum_photon = np.array(flux_vector[0][3:])
          #spectrum_photon *= self.energy_photon**2 * eV2erg
          spectrum_neutrino = np.array(flux_vector[2][3:])
          #spectrum_neutrino *= self.energy_neutrino**2* eV2erg 
          #return [self.energy_photon, spectrum_photon, self.energy_neutrino, spectrum_neutrino]

          if flux_vector[-1][0]==1102:
             spectrum_photon_syn = np.array(flux_vector[-7][3:])
             spectrum_photon_IC = np.array(flux_vector[-6][3:])
             spectrum_photon_pm = np.array(flux_vector[-5][3:])
             spectrum_photon_pa = np.array(flux_vector[-4][3:])
             return [self.energy_photon, spectrum_photon, self.energy_neutrino, spectrum_neutrino, spectrum_photon_syn, spectrum_photon_IC, spectrum_photon_pm, spectrum_photon_pa]
          else:
             spectrum_photon_syn = np.zeros(len(spectrum_photon))
             spectrum_photon_IC = np.zeros(len(spectrum_photon))
             spectrum_photon_pm = np.zeros(len(spectrum_photon))
             spectrum_photon_pa = np.zeros(len(spectrum_photon))

             return [self.energy_photon, spectrum_photon, self.energy_neutrino, spectrum_neutrino]
