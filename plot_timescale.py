import matplotlib.pyplot as plt
import numpy as np
from sources import *
from run_fit import setup_run
import AMES

class lh:
    def __init__(self):
        pass

    def read_file(self, filename):
        # Open the file
        infile = open(filename, 'r')
        lines = infile.readlines()
        data = []
        for line in lines:
            # separates line into a list of items.  ',' tells it to split the lines at the commas
            sline = line.split(' ')
            data.append([float(x) for x in sline])
        infile.close()  # Always close the file!
        return data

    def plot_timescale(self, src_name):
        fig, ax = plt.subplots() 

        output_folder = '../LLGRB-result/' + src_name
        filename = output_folder + "/LeptohadronicEnergy.dat"
        energy = self.read_file(filename)
        filename = output_folder + "/timescale_escape.dat"
        timescale_escape = self.read_file(filename)
        filename = output_folder + "/timescale_syn.dat"
        timescale_syn = self.read_file(filename)
        filename = output_folder + "/timescale_IC.dat"
        timescale_IC = self.read_file(filename)
        filename = output_folder + "/timescale_gg.dat"
        timescale_gg = self.read_file(filename)
        filename = output_folder + "/timescale_photopair.dat"
        timescale_pair = self.read_file(filename)
        filename = output_folder + "/timescale_photomeson.dat"
        timescale_photomeson = self.read_file(filename)

        s = AMES.Source()
        energy_proton = np.array(s.getProton().getEnergy())
        energy_electron = np.array(s.getElectron().getEnergy())
        c_cnst = AMES.c_cnst
        erg2eV = AMES.erg2eV
        eV2erg = AMES.eV2erg
        PI = np.pi

        folder='figure'
        data_dict, src_list, models = setup_run(src_name)
        src_param = source_dict[src_name]
        param_bf = np.loadtxt(folder + '/' + src_name + '_best_fit.txt')
        shared_param = src_list[0]['shared_param']
        free_keys = []
        theta = {}
        for k in shared_param.keys():
            theta[k] = shared_param[k]
            if shared_param[k].is_free:
               free_keys.append(k)
        for i, k in enumerate(free_keys):
            theta[k].value = param_bf[i]

        L_gamma = 10**src_param['logLiso'] 
        log_Radius = theta['log_Radius'].value
        Gamma_j = theta['Gamma_j'].value
        Gamma_j_z = Gamma_j/(1+src_param['z'])
        beta_j = np.sqrt(1 - 1./Gamma_j**2)
        r_diss = 10**log_Radius
        spectral_e = theta['spectral_e'].value
        log_xi_p = theta['log_xi_p'].value
        log_xi_B = theta['log_xi_B'].value
        log_xi_e = theta['log_xi_e'].value

        u_ph = L_gamma/4/PI/r_diss**2/Gamma_j**2/AMES.c_cnst* AMES.erg2eV  #to comoving, unit is eV/cm^-3
        log_xi_B = theta['log_xi_B'].value
        xi_B = 10**log_xi_B
        Bprime =np.sqrt(8.0*PI*u_ph* eV2erg*xi_B) # fomulation by xi_B = U_B/U_ph.
        eta_p = 10.
        acctime = eta_p*AMES.e_charge*Bprime*c_cnst/(energy_proton *eV2erg)
        ax.plot(energy_proton, 1./acctime, 'k-', lw=3.5, label='Acc')

        ax.plot(np.array(energy[1][3:])*Gamma_j_z, 1./np.array(timescale_syn[1][3:]), 'C0-', lw=3, label='Syn, e')
        ax.plot(np.array(energy[1][3:])*Gamma_j_z, 1./np.array(timescale_IC[1][3:]), 'C0-.', lw=3, label='IC, e')
        ax.plot(np.array(energy[0][3:])*Gamma_j_z, 1./np.array(timescale_gg[0][3:]), 'C5--', lw=3, label=r'$\gamma \gamma$')
        ax.plot(np.array(energy[4][3:])*Gamma_j_z, 1./np.array(timescale_syn[4][3:]), 'C4-', lw=3, label='Syn, p')
        ax.plot(np.array(energy[6][3:])*Gamma_j_z, 1./np.array(timescale_syn[6][3:]), 'C6-', lw=3, label=r'Syn, $\pi^\pm$')
        t_dec_pic = np.array(energy[6][3:])*Gamma_j_z/AMES.pic_Mass *AMES.pic_Life
        ax.plot(np.array(energy[6][3:])*Gamma_j_z, t_dec_pic, 'C6--', lw=3, label=r'Decay, $\pi^\pm$')
        ax.plot(np.array(energy[7][3:])*Gamma_j_z, 1./np.array(timescale_syn[7][3:]), 'C7-', lw=3, label=r'Syn, $\mu^\pm$')
        t_dec_mu = np.array(energy[7][3:])*Gamma_j_z/AMES.mu_Mass *AMES.mu_Life
        ax.plot(np.array(energy[7][3:])*Gamma_j_z, t_dec_mu, 'C7--', lw=3, label=r'Decay, $\mu^\pm$')
        ax.plot(np.array(energy[4][3:])*Gamma_j_z, 1./np.array(timescale_photomeson[4][3:]), 'C4--', lw=3, label='pm, p')
        ax.plot(np.array(energy[4][3:])*Gamma_j_z, 1./np.array(timescale_pair[4][3:]), 'C4-.', lw=3, label='pa, p')


        print(src_name)
        print('Gamma, ', Gamma_j)
        print('Radius, ', log_Radius)
        print('xi_B, ', log_xi_B, 'B = G ', Bprime)
        print('xi_p, ', log_xi_p)
        print('xi_e, ', log_xi_e)
        print('spectral_e, ', spectral_e)

        pm = AMES.Photomeson(s)
        ph = AMES.Photonbackground(s)
        utility = AMES.Utility()

        #input
        #Luminosity, E_pk_obs
        eps_min = 0.1   #comoving   
        eps_max = 1e7   #comoving
        eps_pk_obs = src_param['Epk_obs']  #obs
        alpha = src_param['alpha'] ## dN/(dtde) \propto e^(-alpha) or e^(-beta) , unit is number/energy/time 
        try:
            beta = src_param['beta'] ## dN/(dtde) \propto e^(-alpha) or e^(-beta) , unit is number/energy/time 
        except:
            print('not using band')
        L_gamma = 10**src_param['logLiso'] 
        u_ph = L_gamma/4/PI/r_diss**2/Gamma_j**2/AMES.c_cnst* AMES.erg2eV  #to comoving, unit is eV/cm^-3
        eps_pk = eps_pk_obs / Gamma_j_z
        energy = np.array(s.getTarget().getEnergy())
        #internal target photons density (comoving frame, eV^-1 cm^-3), setSpectrum
        #try: 
        #    ph_func = self.src_param['target_photon']['func']
        #    ph_spec = ph_func(self.src_param['target_photon']['param'], np.array(energy)) # eV/cm^-2/s^-1
        #    for i in range(len(energy)):
        #        s.getTarget().setSpectrum(i, ph_spec[i])
        #except:
        if src_param['func_form'] == 'CPL':
          ph.Powerlaw(eps_min, eps_pk, alpha, u_ph)
        elif src_param['func_form'] == 'BPL':
          ph.BrokenPowerlaw(eps_min, eps_pk, eps_max, alpha, beta, u_ph)
        elif src_param['func_form'] == 'Band':
          ph.BandFunction(eps_min, eps_pk, eps_max, alpha, beta, u_ph)  # You can use Band function.
        else:
          print('Value Error!')
        spectrum = np.array(s.getTarget().getSpectrum())
        #renormalize the target photon for give energy range
        dum = energy * spectrum
        norm = utility.Integrate(energy, dum, src_param['E_min']/Gamma_j_z, src_param['E_max']/Gamma_j_z)
        spectrum *= u_ph / norm
        for i, x in enumerate(energy):
            s.getTarget().setSpectrum(i, spectrum[i]) 

        energy_proton = s.getProton().getEnergy()
        pm = AMES.Photomeson(s)
        pm.Losstime(s.getProton())
        losstime_pm = np.array(s.getProton().getLosstime())

        pa = AMES.Photopair(s)
        pa.Losstime(s.getProton())
        losstime_pa = np.array(s.getProton().getLosstime())

        #leptonic process
        energy_photon = s.getPhoton().getEnergy()
        gg = AMES.GammaGamma(s)
        gg.Intetime()
        intetime_gg = np.array(s.getPhoton().getIntetime())

        energy_electron = s.getElectron().getEnergy()
        ic = AMES.InverseCompton(s)
        ic.Losstime()
        losstime_ic = s.getElectron().getLosstime()

        syn = AMES.Synchrotron(s)
        Bprime =np.sqrt(8.0*PI*u_ph* eV2erg*xi_B) # fomulation by xi_B = U_B/U_ph.
        syn.Losstime(Bprime)
        losstime_syn = s.getElectron().getLosstime()
        syn.LosstimeProton(Bprime)
        losstime_syn_proton = s.getProton().getLosstime()

        t_dyn = r_diss / Gamma_j / beta_j / AMES.c_cnst
        #ax.plot(energy_electron, np.array(energy_electron)/np.array(energy_electron)/t_dyn, 'k:', lw=2.5, label=r'$t_{\rm dyn}^{-1}$')
        ax.plot(energy_electron, np.array(energy_electron)/np.array(energy_electron)*t_dyn, 'k:', lw=3.5, label='Dyn')
        #ax.plot(energy_electron, losstime_syn, 'C0:', lw=2.5, label=r'$t_{\rm syn, e}^{-1}$ (ph)')
        #ax.plot(energy_proton, losstime_syn_proton, 'C1:', lw=2.5, label=r'$t_{\rm syn, p}^{-1}$ (ph)')
        #ax.plot(energy_electron, losstime_ic, 'C2:', lw=2.5, label=r'$t_{\rm IC}^{-1}$ (ph)')
        ax.plot(energy_photon, 1./intetime_gg, 'C3:', lw=2.5, label=r'$t_{\rm \gamma \gamma}^{-1}$ (ph)')
        ax.plot(energy_proton, 1./losstime_pm, 'C4:', lw=2.5, label=r'$t_{\rm pm}^{-1}$ (ph)')
        #ax.plot(energy_proton, losstime_pa, 'C5:', lw=2.5, label=r'$t_{\rm pa}^{-1}$ (ph)')

        ax.text(2e6, 2e4, 'GRB ' + src_name, fontsize=20)
    
        ax.set_xlim([1e6, 1e19])
        ax.set_ylim([1e-6, 1e5])
        ax.set_xlabel('E [eV]', fontsize=15)
        ax.set_ylabel(r'$\rm Timescale [seconds]$', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=13)
        #ax.legend(loc=0, ncol=2, handlelength=3, fontsize=9, frameon=False)
        #ax.grid(which='major', linestyle=':', linewidth=0.5, color='gray', alpha=0.7)
        ax.loglog()
        
        plt.tight_layout()
        plt.savefig('figure/' + src_name + '_timescale.pdf', bbox_inches='tight')
        plt.show()

lh = lh()
#src_name = '171205A'
#src_name = '190829A'
#src_name = '201015A'
#src_name = '120422A'
#src_name = '060218'
#src_name = '100316D'
src_name_l = ['050826', '060218', '100316D', '120422A', '171205A', '190829A', '201015A']
for src_name in src_name_l:
    lh.plot_timescale(src_name)
