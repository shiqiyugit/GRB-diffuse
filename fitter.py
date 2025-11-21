import numpy as np
import emcee, psutil, os
import corner
from multiprocessing import Pool, RLock, Value, current_process
import multiprocessing as mp
from scipy.stats import gaussian_kde
import threading
import matplotlib.pyplot as plt
from scipy import integrate
from loader import *
import glob
from scipy.stats import mode
from plots import *
from grb import GRBsinglezone, update_s, update_cr
import copy
from functools import partial
from functions import *
from sources import *
import random
import AMES
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample


class fit:
    def __init__(self, data_list, src_param, models, nsteps=300, nwalkers = 10, nchains=3, seed=None, out_name = 'output', folder='result', test_llh = False):
        self.nsteps = nsteps
        #self.i_start = i_start
        self.nwalkers = nwalkers
        self.nchains = nchains
        self.seed = seed
        self.filename = folder + "/sampler_" + out_name + "_seed_" + str(seed) 
        self.x_gamma = data_list['x_gamma']
        self.y_gamma = data_list['y_gamma']
        self.xerr_gamma = data_list['xerr_gamma']
        self.yerr_gamma = data_list['yerr_gamma']
        self.gamma_ul = data_list['isul_gamma']
        try:
            self.x_neutrino = np.array(data_list['x_neutrino'])
            self.y_neutrino = np.array(data_list['y_neutrino'])
            self.yerr_neutrino = np.array(data_list['yerr_neutrino'])
        except:
            print("neutrino UL not detected for this source")
            
        self.src_param = src_param
        self.n_src = len(models)
        shared_param = src_param[0]['shared_param']
        self.test_llh = test_llh

        self.free_keys, self.theta_l, self.theta_h = [], [], []
        self.theta_bf, self.theta_bf_l, self.theta_bf_h = [], [], []
        self.theta = {}
        self.theta_init = []
        
        for k in shared_param.keys():
            self.theta[k] = shared_param[k]
            if shared_param[k].is_free:
                self.free_keys.append(k)
                self.theta_init.append(shared_param[k].value)
                self.theta_l.append(shared_param[k].min)
                self.theta_h.append(shared_param[k].max)
                
        self.models = models
        self.theta_l = np.array(self.theta_l)
        self.theta_h = np.array(self.theta_h)

    def init_model(self):
        """Reconstruct GRBsinglezone objects from parameters (not the object itself)."""
        self._model = []
       
        for mod in self.models:  # now a list of src_param dicts
            if self.test_llh:
                self._model.append(mod.test_model)
            else:
                self._model.append(mod.model)

    def log_prob(self, theta):#, s, cr, syn, IC, ph, pm, pa):

        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        # Pass a detached parameter dict to likelihood impl
        theta_dict = self.theta.copy()
        theta_dict.update({
            k: Parameter(k, v, bounds=(low, high))
            for k, v, low, high in zip(self.free_keys, theta, self.theta_l, self.theta_h)
        })
        # multi-process needs to make sure isolated theta

        return lp + self._log_likelihood_impl(theta_dict)#, syn=syn, IC=IC, ph=ph, pm=pm, pa=pa)

    def _log_likelihood_impl(self, theta_dict):#, syn, IC, ph, pm, pa):
        s, cr, syn, IC, ph, pm, pa= self.init_worker()
        llh_gamma = 0
        for i, model in enumerate(self._model):
            model_copy = copy.deepcopy(model)
            energy_photon, spectrum_photon, energy_neutrino, spectrum_neutrino = model_copy(theta_dict, s=s, cr=cr, syn=syn, IC=IC, ph=ph, pm=pm, pa=pa, have_proton=self.have_proton);

            model_photon = np.interp(self.x_gamma[i], energy_photon, spectrum_photon)
            if np.any(np.isnan(model_photon)):
                return -np.inf
            sigma2_photon = self.y_gamma[i] 
            sigma2_photon = np.where(sigma2_photon <= 0, 1e-20, sigma2_photon)
            ul_mask = self.gamma_ul[i] 
            rl_mask = (~ul_mask) 
            # for ULs
            sigma_ul = self.y_gamma[i][ul_mask] / 1.64 #95% UL
            xi = model_photon[ul_mask] #- self.y_gamma[i][ul_mask] 
            ll_upper = -0.5 * (xi / sigma_ul)**2
            ll_upper = np.where(model_photon[ul_mask] <= self.y_gamma[i][ul_mask], 0, ll_upper) #not penalizing on below
            llh_gamma += np.nansum(ll_upper)
            
            residual = model_photon[rl_mask] - self.y_gamma[i][rl_mask]
            #pos_error = residual > 0
            yerr = self.yerr_gamma[i][rl_mask] ** 2 #(np.hstack(self.yerr_gamma[i][1])**2)[rl_mask]
            #yerr_low = self.yerr_gamma[i][1][rl_mask] ** 2 #(np.hstack(self.yerr_gamma[i][0])**2)[rl_mask]
            yerr = np.where(yerr <= 0, 1e-20, yerr)
            #yerr_low = np.where(yerr_low <= 0, 1e-20, yerr_low)
            direct = -0.5 * (residual**2 / yerr + np.log(2 * yerr))
            #direct[pos_error] = -0.5 * (residual[pos_error]**2 / yerr_high[pos_error] + np.log(2 * yerr_high[pos_error]))
            llh_gamma += np.nansum(direct)

            #direct = -0.5 * (residual**2 / sigma2_photon[rl_mask] + np.log(2 * sigma2_photon[rl_mask]))
            #direct = np.where(np.isfinite(direct), direct, -1e3)
            
            #llh_gamma += np.nansum(direct)
            if not np.isfinite(llh_gamma):
                return -np.inf
        #thread_id = threading.get_ident()           # numeric thread ID
        #thread_name = threading.current_thread().name  # readable thread name
        #print(f"[Thread {thread_name} | ID {thread_id}] llh_gamma = {llh_gamma}")
        s.clear()
        del s, cr, syn, IC, ph, pm, pa
        return llh_gamma
    
    def log_prior(self, theta):
        if np.any(self.theta_l>theta) or np.any(self.theta_h<theta):
              return -np.inf
        return 0.0

    def run_all_chains(self, resume=False, init=False, have_proton=True):
        self.init_model()
        processes = []
        self.have_proton=have_proton
        if self.nchains >1:
          ctx = mp.get_context("spawn")  # SAFER on macOS
          for i in range(self.nchains):
            file_i = self.filename + f"_chain_{i}.h5"
            backend = emcee.backends.HDFBackend(file_i)
            # uncomment for single job testing
            # IMPORTANT: only pass simple objects to subprocess
            p = ctx.Process(
                target=self.worker_wrapper,
                args=(i, backend, resume, init, have_proton)
            )
            processes.append(p)
            p.start()
    
          for p in processes:
            p.join()
        else:
            print("using a single chain, skip chain parallization")
            file_i = self.filename + f"_chain_0.h5"
            backend = emcee.backends.HDFBackend(file_i)
            self.worker_wrapper(0, backend, resume, init, have_proton)

    def init_worker(self):
        s = AMES.Source()
        update_s(s)
        s.InitSource()
        
        syn = AMES.Synchrotron(s)
        IC = AMES.InverseCompton(s)
        ph = AMES.Photonbackground(s)
        pm = AMES.Photomeson(s)
        pa = AMES.Photopair(s)
        
        cr = AMES.CRDistribution(s)
        update_cr(cr, self.have_proton)
        return s, cr, syn, IC, ph, pm, pa

    def worker_wrapper(self, chain_id, backend, resume, init, have_proton=True):
        if not resume:
            # Initialize initial walker positions
            rng = np.random.default_rng(self.seed + chain_id) #self.i_start + chain_id)
            theta_l = [self.theta[k].min for k in self.free_keys]
            theta_h = [self.theta[k].max for k in self.free_keys]

            initial = np.array([
                np.random.uniform(low=theta_l[i], high=theta_h[i], size=self.nwalkers)
                for i in range(len(self.free_keys))
                ]).T
            nwalkers, ndim = initial.shape
            if init: 
                print(f"[chain {chain_id}] Resetting backend")
                backend.reset(nwalkers, ndim)
        else:
            print(f"[chain {chain_id}] Continuing from last sample")
            initial = backend.get_last_sample().coords

        # one ames for each chain and all walkers in seriels

        self.run_chain_threaded(chain_id, initial, backend, resume=resume)# s=s, cr=cr, syn=syn, IC=IC, ph=ph, pm=pm, pa=pa, resume=resume)

    def run_chain_threaded(self, chain_id, initial, backend, 
            resume=False):#, n_threads=8):

        nwalkers, ndim = initial.shape
        n_threads = nwalkers
        remaining_steps = self.nsteps - backend.iteration if backend.iteration > 0 else self.nsteps
    
        log_prob_fn = self.log_prob
        #sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, backend=backend, threads=n_threads)
        if resume:
            initial=None    
        print(f"[chain {chain_id}] Starting threaded MCMC sampling with {n_threads} threads")
    
        print("sched affinity:", os.sched_getaffinity(0))
        moves = [
            (emcee.moves.StretchMove(a=2.2), 0.8),   # main move
            (emcee.moves.DEMove(gamma0=None), 0.2)]
        with ProcessPoolExecutor(max_workers=n_threads) as executor:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_prob_fn, backend=backend, pool=executor, moves=moves
            )
        
            sampler.run_mcmc(initial, remaining_steps, progress=True)
        print(f"[chain {chain_id}] Finished sampling")

    # ========= plotting functions ================

    def process_mcmc(self):
        all_log_probs, all_samples = [], []
        for afile in self.files:
            reader = emcee.backends.HDFBackend(afile)
            log_prob_samples = reader.get_log_prob(flat=False, thin=self.thin, discard=self.burnin)
            all_log_probs.append(log_prob_samples)
            sample = reader.get_chain(flat=False, thin=self.thin, discard=self.burnin)
            all_samples.append(sample)
        last_probs = np.array([np.nanmedian(prob, axis=1)[-1] for prob in all_log_probs])
        median_probs = np.nanmedian(last_probs)
        mask =last_probs>median_probs
        all_log_probs = [x for x, m in zip(all_log_probs, mask) if m]
        all_samples = [x for x, m in zip(all_samples, mask) if m]

        self.truncated_probs = all_log_probs
        self.truncated_samples = all_samples

    def load_result(self, out_name = 'output', folder='result', infile=None, thin=20, burnin=100, raw=True, have_proton=True):
        self.have_proton=have_proton
        if not infile:
            if self.seed is not None:
                filename = folder + "/sampler_" + out_name + "_seed_" + str(self.seed) +  "_chain_?.h5"
            else:
                filename = folder + "/sampler_" + out_name + "_seed_*_chain_*.h5"
            files = glob.glob(filename)
        else:
            files = [infile]
        self.files = files
        backend = emcee.backends.HDFBackend(files[0])
        self.nwalkers = backend.get_chain().shape[1]
        self.thin = thin
        self.burnin=burnin
        self.init_model()
        if not raw: self.process_mcmc()


    def plot_llh(self, src_name='190829A', burnin=100, out_name='output', folder='result', infile=None, raw = True):
        """Plot the log-likelihood evolution using the minimum chain length."""
        plt.figure(figsize=(12, 6))
        
        if raw:
            all_log_probs = []
            for afile in self.files:
                reader = emcee.backends.HDFBackend(afile)
                log_prob_samples = reader.get_log_prob(flat=False)
                all_log_probs.append(log_prob_samples)
        
            truncated_probs = all_log_probs #[x[:min_length] for x in all_log_probs]
        else:
            truncated_probs = self.truncated_probs
            steps = self.steps

        # Plot individual chains
        for i, prob in enumerate(truncated_probs):
            # Plot mean likelihood
            mean_log_prob = np.nanmedian(prob, axis=1)
            steps = np.arange(len(mean_log_prob))
            plt.plot(steps, mean_log_prob, 
                    alpha=0.8, 
                    label=f'Chain {i+1} (Mean)')
            for walker in range(self.nwalkers):
                plt.plot(steps, prob[:, walker], 
                        alpha=0.3, 
                        lw=0.5,
                        color=plt.gca().lines[-1].get_color())

    # Plot combined statistics (with burn-in removal)
        if len(self.files) > 1:
            all_medians = []
            all_lowers = []
            all_uppers = []

            for chain in truncated_probs:
                all_medians.append(np.percentile(chain, 50, axis=1))
                all_lowers.append(np.percentile(chain, 16, axis=1))
                all_uppers.append(np.percentile(chain, 84, axis=1))
        
            # Find maximum length among chains
            max_len = max(len(m) for m in all_medians)
            
            # Create arrays to hold the results
            median_combined = np.full(max_len, np.nan)
            lower_combined = np.full(max_len, np.nan)
            upper_combined = np.full(max_len, np.nan)
            
            # Combine results where available
            for i in range(max_len):
                # Collect available values at this step
                step_medians = [m[i] for m in all_medians if i < len(m)]
                step_lowers = [l[i] for l in all_lowers if i < len(l)]
                step_uppers = [u[i] for u in all_uppers if i < len(u)]
                
                if step_medians:  # Only calculate if we have data
                    median_combined[i] = np.median(step_medians)
                    lower_combined[i] = np.median(step_lowers)
                    upper_combined[i] = np.median(step_uppers)
            
            # Create matching steps array
            steps_combined = np.arange(max_len)
            
            # Plotting
            plt.fill_between(steps_combined, lower_combined, upper_combined,
                            alpha=0.2, color='gray', label='1σ range (all chains)')
            plt.plot(steps_combined, median_combined,
                    color='k', ls='--', label='Median (all chains)')
            
            
        if (burnin > 0) and raw:
            plt.axvline(burnin, color='red', linestyle=':', alpha=0.5)
            plt.text(burnin+10, plt.ylim()[0]+0.1*(plt.ylim()[1]-plt.ylim()[0]), 
                    'Burn-in', 
                    color='red',
                    ha='left')
        plt.gca().set_ylim(np.min(median_combined)*0.1,1.5*np.max(median_combined))
        
        plt.tight_layout()

        plt.savefig(f"{folder}/likelihood_evolution_{out_name}.pdf", dpi=300)
        #plt.show()

    def hpd_interval(self, samples, credible_level=0.68):
        s = np.sort(samples)
        n = len(s)
        interval_idx_inc = int(np.floor(credible_level * n))
        n_intervals = n - interval_idx_inc
        intervals = [(s[i], s[i + interval_idx_inc]) for i in range(n_intervals)]
        widths = [end - start for start, end in intervals]
        min_idx = np.argmin(widths)
        return intervals[min_idx]

    def plot_mcmc_result(self, bfp=False, burnin=100, out_name = 'output', folder='result', raw=True):
        if raw:
            all_samples, all_probs = [], []
            for afile in self.files:
                reader = emcee.backends.HDFBackend(afile)
                flat_sample = reader.get_chain(discard=burnin, flat=True)
                all_samples.append(flat_sample)
                flat_prob = reader.get_log_prob(discard=burnin, flat=True)
                all_probs.append(flat_prob)
            flat_samples = np.vstack(all_samples)  # shape: (total_samples, n_params)
            flat_log_probs = np.concatenate(all_probs)  # shape: (total_samples,)
        else:
            flat_samples = np.vstack(self.truncated_samples)
            flat_log_probs =  np.concatenate(self.truncated_probs)

        lower = np.zeros(flat_samples.shape[1])
        upper = np.zeros(flat_samples.shape[1])
        
        for j in range(flat_samples.shape[1]):
            lo, hi = self.hpd_interval(flat_samples[:, j], 0.68)
            lower[j], upper[j] = lo, hi

        if bfp:
            best_idx = np.argmax(flat_log_probs)
            best_fit = flat_samples[best_idx]
            # kNN density estimation
            
            print("Global BFP (multivariate KDE within HPD):", best_fit)

            fig = corner.corner(
                flat_samples,
                labels=list(self.free_keys),
                quantiles=[0.16,0.5, 0.84],  # Only show the 1-sigma quantiles
                plot_datapoints=False, # turn off scatter
                fill_contours=True,
                show_titles=True,
                label_kwargs=dict(fontsize=15),
                title_kwargs={"fontsize": 15},
                truths=best_fit,  # Show the mode as truth values
                truth_color='red'  # Color for the mode lines
            )
            

        else:
            best_fit = np.percentile(flat_samples, 50, axis=0) 
            param_name = [
                r'${\rm log}\,\xi_B$',
                r'${\rm log}\,\xi_e$',
                r'${\rm log}\,R$',
                r'$\Gamma_j$',
                r'$s_e$',
                r'$\epsilon_e$',
                r'${\rm log}\,\xi_p$'
            ]
            fig = corner.corner(
                flat_samples, plot_datapoints=False, fill_contours=True, labels=list(param_name), 
                #quantiles=[0.16, 0.5, 0.84], 
                show_titles=False, label_kwargs=dict(fontsize=15), truths=best_fit, truth_color='red')

        ndim = len(self.free_keys)
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            ax = axes[i, i]  # 1D marginal panel
        
            # vertical lines for HPD
            ax.axvline(lower[i], color="r", ls="--", lw=2, alpha=0.8, label="68% HPD" if i == 0 else "")
            ax.axvline(upper[i], color="r", ls="--", lw=2, alpha=0.8)
            ax.set_title("")
            # vertical line for median or MAP
            ax.axvline(best_fit[i], color="r", ls="-", lw=2, alpha=0.8, label="Best-fit" if i == 0 else "")
        
        # add legend only once (top-left panel)
        axes[0, 0].legend(fontsize=10)

        plt.savefig(folder + '/' + out_name + '_contour.pdf')
        #plt.show()
        self.flat_samples = flat_samples

        # sample from 1sigma region for band plot later
        mask = np.all((flat_samples >= lower) & (flat_samples <= upper), axis=1)
        sigma_samples = flat_samples[mask]

        print(f"Extracted {len(sigma_samples)} samples within 1σ region.")

        # --- Downsample to N points ---
        #N = 10000  # adjust
        #if len(sigma_samples) > N:
        #    idx = np.random.choice(len(sigma_samples), size=N, replace=False)
        #    sigma_samples = sigma_samples[idx]

        #  Save
        np.savetxt(folder + "/params_1sigma_"+out_name+".csv", sigma_samples, delimiter=",")

        self.samples_1sigma = sigma_samples
        #print(f"saved {N} 1sigma samples in params_1sigma.csv for band plotting")
        np.savetxt(folder + '/' + out_name + '_best_fit.txt', best_fit)
        np.savetxt(folder + '/' + out_name + '_best_fit_l.txt', lower)
        np.savetxt(folder + '/' + out_name + '_best_fit_h.txt', upper)

    def get_bfp(self, folder='result',out_name = 'output'):
        theta = np.loadtxt(folder + '/' + out_name + '_best_fit.txt')
        theta_dict = self.theta.copy()
        theta_dict.update({
            k: Parameter(k, v, bounds=(low, high))
            for k, v, low, high in zip(self.free_keys, theta, self.theta_l, self.theta_h)
        })
        return theta_dict

    def plot_sed_result(self, grb_l, out_name, folder='result', infile=None):
        fig, ax = plt.subplots()
        print("Loading best fits from: ", folder + '/' + out_name)
        param_bf = np.loadtxt(folder + '/' + out_name + '_best_fit.txt')
        self.samples_1sigma = np.loadtxt(folder +"/params_1sigma_"+out_name+".csv", delimiter=",")

        for i, k in enumerate(self.free_keys):
            self.theta[k].value = param_bf[i]
        
        for i, model in enumerate(self._model):
            # Set up source and CR distribution
            s = AMES.Source()
            update_s(s)
            s.InitSource()
            
            syn = AMES.Synchrotron(s)
            IC = AMES.InverseCompton(s)
            ph = AMES.Photonbackground(s)
            pm = AMES.Photomeson(s)
            pa = AMES.Photopair(s)
            
            cr = AMES.CRDistribution(s)
            update_cr(cr, self.have_proton)
        
            # Get photon and neutrino spectra from the model
            energy_photon, spectrum_photon, energy_neutrino, spectrum_neutrino = model(self.theta, s=s, cr=cr, syn=syn, IC=IC, ph=ph, pm=pm, pa=pa)
            spectrum_photon *= energy_photon**2*AMES.eV2erg
            spectrum_neutrino *= energy_neutrino**2*AMES.eV2erg
            # Plot the central model predictions
            ax.plot(energy_photon, spectrum_photon, '-', lw=2.5, c=gamma_colors[i],
                label=fr"model $\gamma$ ({self.src_param[i]['name']})")
            ax.plot(energy_neutrino, spectrum_neutrino, '-', lw=2.5, c=blue_colors[i],
                    label=fr"model $\nu$")
            # Plot uncertainty bands
            N=100
            print("plot band from 50 samples in 1 sigma region")
            energy_photon, spectrum_photon_l, spectrum_photon_h, \
            energy_neutrino, spectrum_neutrino_l, spectrum_neutrino_h = generate_parameter_band(
                model, self.samples_1sigma, self.free_keys, self.theta, N, s=s, cr=cr, syn=syn, IC=IC, ph=ph, pm=pm, pa=pa)
            spectrum_photon_l *= energy_photon**2*AMES.eV2erg
            spectrum_photon_h *= energy_photon**2*AMES.eV2erg
            spectrum_neutrino_l *= energy_neutrino**2*AMES.eV2erg
            spectrum_neutrino_h *= energy_neutrino**2*AMES.eV2erg
            ax.fill_between(energy_photon, spectrum_photon_l, spectrum_photon_h,
                           alpha=0.5, color=gamma_colors[i])
            ax.fill_between(energy_neutrino, spectrum_neutrino_l, spectrum_neutrino_h,
                            alpha=0.5, color=blue_colors[i])

        # Plot gamma-ray data with error bars
        for i, (x, y, e, ul) in enumerate(zip(self.x_gamma, self.y_gamma, self.yerr_gamma, self.gamma_ul)):
            ax.errorbar(np.array(x), np.array(y), yerr=np.array(e), uplims=ul,
                 fmt='o', markersize=2.5, ecolor=gamma_colors[i], color='black',  
                 capsize=2.5, elinewidth=2, markeredgewidth=1, label=fr"{self.src_param[i]['name']}", alpha=0.8)
        ax.set_xlim([1e0, 1e18])
        ax.set_ylim([1e-14, 1e-3])
        ax.set_xlabel(r'$\rm E [eV]$', fontsize=17)
        ax.set_ylabel(r'$\rm Flux (\rm erg cm^{-2} s^{-1})$', fontsize=17)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(loc=2, ncol=2, handlelength=3, fontsize=9, frameon=False)
        plt.tight_layout() 
        plt.savefig(folder + '/' + out_name + '_fit.pdf')
        #plt.show()

    def plot_data(self, out_name, folder='result' ):
        fig, ax = plt.subplots()
        print("err gamma: ", self.xerr_gamma)
        #self.xerr_gamma = np.abs(self.xerr_gamma)
        i = 0
        for x, xerr, y, yerr, ul in zip(self.x_gamma, self.xerr_gamma, self.y_gamma, self.yerr_gamma, self.gamma_ul):
            ax.errorbar(x, y, yerr=yerr, uplims=ul, fmt='o', ecolor=gamma_colors[i], markersize=2.5, color='black', capsize=2.5, elinewidth=2, markeredgewidth=1, label=out_name)
            i += 1
          
        #ax.set_xlim([1e0, 1e18])
        #ax.set_xlim(1e3, 260e3)
        #ax.set_ylim([1e-12, 1e-2])
        #ax.set_ylim([1e-14, 1e-3])
        ax.set_xlabel(r'$\rm E [eV]$', fontsize=17)
        ax.set_ylabel(r'$\rm Flux (\rm erg cm^{-2} s^{-1})$', fontsize=17)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(loc=2, ncol=2, handlelength=3, fontsize=9, frameon=False)
        plt.tight_layout()
        plt.savefig(folder + '/' + out_name + '_data_points.pdf')
        #plt.show()

    def plot_result_combine(self, grb_l, out_name, folder='result'):
        fig, ax = plt.subplots()
        param_bf = np.loadtxt(f"{folder}/{out_name}_best_fit.txt")
        #param_l  = np.loadtxt(f"{folder}/{out_name}_best_fit_l.txt")        
        #param_h  = np.loadtxt(f"{folder}/{out_name}_best_fit_h.txt")        
        colors = ['C0', 'C1', 'C2', 'C3']
        for i, (mo, color_style,src,grb_dum) in enumerate(zip(_model, colors,self.src_param,grb_l)):
            label_name = src['name']
            energy_photon, spectrum_photon, energy_neutrino, spectrum_neutrino = mo(param_bf, s, cr)
            ax.plot(energy_photon, spectrum_photon, '-',c=color_style, label=f'{label_name} photon')
            ax.plot(energy_neutrino, spectrum_neutrino, '--',c=color_style, label=f'{label_name} neutrino')
            ax.errorbar(self.x_gamma[i], self.y_gamma[i], yerr=self.yerr_gamma[i],
                uplims=True, fmt=".", capsize=5, c=color_style, label=f'{label_name} gamma obs')
            ax.errorbar(self.x_neutrino[i], self.y_neutrino[i]*1e-5, yerr=self.yerr_neutrino[i]*1e-5,
                        uplims=True, fmt="*", capsize=5, c=color_style, label=f'{label_name} neutrino obs')
            #ax.plot(grb_dum.energy_obs, grb_dum.flux_internal, c=color_style, label=f'{label_name} grb dum')                   
        ax.text(1e14, 1e-3, "combine", fontsize=20)
        ax.set_xlim([1e0, 1e18])
        ax.set_ylim([1e-14, 1e-5])
        ax.set_xlabel(r'$\rm E [eV]$', fontsize=17)
        ax.set_ylabel(r'$\rm Flux (\rm erg cm^{-2} s^{-1})$', fontsize=17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='upper right', ncol=2, handlelength=3, fontsize=9, frameon=False)
        plt.tight_layout()
        plt.savefig(f"{folder}/{out_name}_fit.pdf")
        #plt.show()
        
    def plot_sed_draft(self, grb_l, out_name, folder='result', infile=None):
        """
        add by Bing 20250930
        plot for figures in the draft
        """
        fig, ax = plt.subplots()
        #fig, (legend_ax, ax) = plt.subplots(2, 1, figsize=(10, 8), 
                                 #gridspec_kw={'height_ratios': [1, 4]})
        param_bf = np.loadtxt(folder + '/' + out_name + '_best_fit.txt')

        self.have_proton = True

        for i, k in enumerate(self.free_keys):
            self.theta[k].value = param_bf[i]
        #self.theta['Gamma_j'].value = 5
        #self.theta['log_Radius'].value = 12


        for i, model in enumerate(self.models):
            # Set up source and CR distribution
            s = AMES.Source()
            update_s(s)
            s.InitSource()
            syn = AMES.Synchrotron(s)
            IC = AMES.InverseCompton(s)
            ph = AMES.Photonbackground(s)
            pm = AMES.Photomeson(s)
            pa = AMES.Photopair(s)
            ph=AMES.Photonbackground(s)
            #ph.GreyBody(2e6, 1e-8)
            #target_energy, target_spectrum = np.array(s.getTarget().getEnergy()), np.array(s.getTarget().getSpectrum())
            #ax.plot(target_energy, target_energy**2*target_spectrum, '-')
        
            cr = AMES.CRDistribution(s)
            have_proton, have_timescale = self.have_proton, True
            update_cr(cr, have_proton, have_timescale)
        
            # Get photon and neutrino spectra from the model
            result = model.model(self.theta, s=s, cr=cr, syn=syn, IC=IC, ph=ph, pm=pm, pa=pa)
            energy_photon, spectrum_photon = result[0], result[0]**2*AMES.eV2erg*result[1]
            energy_neutrino, spectrum_neutrino = result[2], result[2]**2*AMES.eV2erg*result[3]
            spectrum_photon_syn, spectrum_photon_IC = result[0]**2*AMES.eV2erg*result[4], result[0]**2*AMES.eV2erg*result[5]
            spectrum_photon_pm = result[0]**2*AMES.eV2erg*result[6]
            spectrum_photon_pa = result[0]**2*AMES.eV2erg*result[7]

            np.savetxt(folder + '/' + out_name + '_fit_neutrino.txt', np.c_[energy_neutrino, spectrum_neutrino])
            #ratio_IC = np.nan_to_num(spectrum_photon_IC/spectrum_photon_syn, nan=0.0, posinf=0.0, neginf=0.0)
            #ratio_pm = np.nan_to_num(spectrum_photon_pm/spectrum_photon_syn, nan=0.0, posinf=0.0, neginf=0.0)
            #ratio_pa = np.nan_to_num(spectrum_photon_pa/spectrum_photon_syn, nan=0.0, posinf=0.0, neginf=0.0)
            #np.savetxt(folder + '/' + out_name + '_fit_ratio.txt', np.c_[energy_photon, ratio_IC, ratio_pm, ratio_pa])

            # Plot the central model predictions
            ax.plot(energy_photon, spectrum_photon, '-', lw=3, c=gamma_colors[i],
                label='Total')
            ax.plot(energy_photon, spectrum_photon_syn, '--', lw=3, c=gamma_colors[i],
                label='Syn')
            ax.plot(energy_photon, spectrum_photon_IC, '-.', lw=3, c=gamma_colors[i],
                label='IC')
            ax.plot(energy_photon, spectrum_photon_pm, '--', lw=3, c='C4',
                label='pm')
            ax.plot(energy_photon, spectrum_photon_pa, '--', lw=3, c='C5',
                label='pa')
            dum = [x1+x2+x3+x4 for x1, x2, x3, x4 in zip(spectrum_photon_syn, spectrum_photon_IC, spectrum_photon_pm, spectrum_photon_pa)]
            #ax.plot(energy_photon, dum, 'k--', lw=4,
            #    label=fr"model $\gamma$ ({self.src_param[i]['name']}), all")
            ax.plot(energy_neutrino, spectrum_neutrino, '-', lw=3, c=blue_colors[i],
                    label='Neutrino')

            have_proton, have_timescale = self.have_proton, False
            update_cr(cr, have_proton, have_timescale)

            sigma_samples = np.loadtxt(folder + "/params_1sigma_"+out_name+".csv", delimiter=",")
            self.samples_1sigma = sigma_samples

            energy_photon, spectrum_photon_l, spectrum_photon_h, \
            energy_neutrino, spectrum_neutrino_l, spectrum_neutrino_h = generate_parameter_band(
            model.model, self.samples_1sigma, self.free_keys, self.theta, 2000, s=s, cr=cr, syn=syn, IC=IC, ph=ph, pm=pm, pa=pa)
            ax.fill_between(energy_photon, spectrum_photon_l, spectrum_photon_h,
                            alpha=0.5, color=gamma_colors[i])
            ax.fill_between(energy_neutrino, spectrum_neutrino_l, spectrum_neutrino_h,
                            alpha=0.5, color=blue_colors[i])

            np.savetxt(folder + '/' + out_name + '_fit_neutrino_l.txt', np.c_[energy_neutrino, spectrum_neutrino_l])
            np.savetxt(folder + '/' + out_name + '_fit_neutrino_h.txt', np.c_[energy_neutrino, spectrum_neutrino_h])

        # Plot gamma-ray data with error bars
        for i, (x, y, e, ul) in enumerate(zip(self.x_gamma, self.y_gamma, self.yerr_gamma, self.gamma_ul)):
            idx = np.array(x) < 1e7
            idx_lat = np.array(x) > 1e7
            ax.errorbar(np.array(x)[idx], np.array(y)[idx], yerr=np.array(e)[idx], uplims=ul[idx], fmt='o', markersize=2.5, ecolor=gamma_colors[i], color='black', capsize=2.5, elinewidth=2, markeredgewidth=1, label=r'$\it{Swift}$-BAT', alpha=0.8)

            #ax.errorbar(np.array(x)[idx_lat], np.array(y)[idx_lat], yerr=[np.array(e[0])[idx_lat], np.array(e[1])[idx_lat]], uplims=ul[idx_lat], fmt='o', markersize=3.5, ecolor='C3', color='C3', capsize=2.5, elinewidth=2, markeredgewidth=1, label=fr"{self.src_param[i]['name']}" + ' (Fermi-LAT)', alpha=0.8)
            ax.errorbar(np.array(x)[idx_lat], np.array(y)[idx_lat], yerr=np.array(e)[idx_lat], uplims=ul[idx_lat], fmt='o', markersize=3.5, ecolor='C3', color='C3', capsize=2.5, elinewidth=2, markeredgewidth=1, label=r'$\it{Fermi}$-LAT', alpha=0.8)

        for i, grb_dum in enumerate(grb_l):
            ax.plot(grb_dum.energy_obs, grb_dum.flux_internal, '-.', c='k')

        data = np.loadtxt('data/thermal_op_060218.dat')
        ax.plot(10**data[:,0]*AMES.Hz2eV, 10**data[:,1], 'o')
        data = np.loadtxt('data/thermal_x_060218.dat')
        ax.plot(10**data[:,0]*AMES.Hz2eV, 10**data[:,1], 'o')

        ax.set_xlim([1e-2, 1e18])
        ax.set_ylim([1e-14, 1e-6])
        ax.set_xlabel(r'$\rm E [eV]$', fontsize=15)
        ax.set_ylabel(r'$\rm Flux (\rm erg~cm^{-2}~s^{-1})$', fontsize=15)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.text(2e-2, 3e-7, 'GRB ' + out_name, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=13)
        #ax.legend(loc=2, ncol=2, handlelength=3, fontsize=9, frameon=False)

        """
        # Create legend in the top subplot
        legend_ax.axis('off')  # Hide the axes for legend subplot
        
        # Get handles and labels from the main plot
        handles, labels = ax.get_legend_handles_labels()

        # Create legend in the top subplot with multiple columns
        legend_ax.legend(handles, labels, loc='center', ncol=3, 
                handlelength=3, fontsize=20, frameon=False)
        """

        plt.tight_layout() 
        plt.savefig(folder + '/' + out_name + '_fit.pdf')
        plt.show()
