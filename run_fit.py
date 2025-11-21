import argparse
import numpy as np
import sys
import scipy as sp
from functions import *
from loader import *
from fitter import fit
from grb import GRBsinglezone
from sources import *
import json

import os
#os.environ["OMP_NUM_THREADS"] = "1"

def setup_run(src_name, shared_params=None, plot=False, config=None):
    data_keys =['x_gamma', 'y_gamma',  'xerr_gamma', 'yerr_gamma', 'isul_gamma', 'x_neutrino', 'y_neutrino', 'yerr_neutrino']
    data_dict = {key: [] for key in data_keys}
    xlim=None #[15e3, 150e3]
    gbm = False
    bat = True
    fermi=True
    if (src_name == 'combine') or ('sbo' in src_name):

       src_list, data_sets, models = [], [], []
       x_gamma, y_gamma, yerr_gamma, isul_gamma, x_nu, y_nu, yerr_nu = [], [],[],[],[],[],[]
       i = 0
       if src_name == 'combine': 
           try: src_names = config['sources']
           except: src_names = config #["190829A", "171205A", "201015A", "120422A"]
           print(src_names, config['sources'])
           src_dict = {k : source_dict[k] for k in src_names}
       else: src_dict = sbo_dict
       for k in src_dict.keys():
           fermi=True
           if (k == '060218') or (k == '050826') or (k == '120422A'):
               fermi=False
           data = get_data(k, bat=bat, gbm=gbm, fermi=fermi, xlim=xlim, plot=plot)
           for dk in data_keys:
               data_dict[dk].append(data[dk])
           if shared_params:
               shared_param_dict = create_parameter_dict(shared_params)
               src_dict[k]['shared_param'] = shared_param_dict
           models.append(GRBsinglezone(src_param = src_dict[k]))
           src_list.append(src_dict[k])

    else:
       if (src_name == '060218') or (src_name == '050826') or (src_name == '120422A'):
           fermi=False
       data = get_data(src_name, gbm=gbm, bat=bat, fermi=fermi, xlim=xlim, plot=plot)

       for dk in data_keys:
           data_dict[dk].append(data[dk])
       if shared_params:
           shared_param_dict = create_parameter_dict(shared_params)
           source_dict[src_name]['shared_param'] = shared_param_dict
       src_list = [source_dict[src_name]]
       models = [GRBsinglezone(src_param = source_dict[src_name])]
    return data_dict, src_list, models

if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser(description='Example script with flags: python run_fit.py -src 190829A -c 1 -w 16 -n 300 -i 0 -init')

    # Add flag arguments
    parser.add_argument('--source', '-src', type=str, required=True, help='source name')

    parser.add_argument('--outdir', '-od', type=str,default='../LLGRB-result', help='Output directory path')
    parser.add_argument('--initial', '-init', action='store_true', help='run for the first time')
    parser.add_argument('--seed', '-i', type=int, default=0, help='seed')
    parser.add_argument('--nwalkers', '-w', type=int, default=18, help='num of walkers, at lesat 2* parameters')
    parser.add_argument('--nsteps', '-n', type=int, default=2, help='steps of mcmc')
    parser.add_argument('--nchains', '-c', type=int, default=1, help='how many chains to use for mcmc')
    parser.add_argument('--test_llh', '-tl', action='store_true', help='test llh')
    parser.add_argument('--shortbreakout', '-sbo', action='store_true', help='run sbo jobs')
    parser.add_argument('--larger_range', '-lr', action='store_true', help='use larger par ranges')
    parser.add_argument('--reseed_run', '-rs', action='store_true', help='run seeds to sample from scratch, will drop the prior knowledge')
    parser.add_argument('--have_proton', '-hp', action='store_false', help='run hadronic process=True, EM only=False')
    parser.add_argument('--config', '-cg', default=None)

    args = parser.parse_args()

    src_name = args.source
    initial_run = args.initial
    seed = args.seed
    #out_dir = args.outdir
    #nwalkers = args.nwalkers
    nsteps = args.nsteps
    nchains = args.nchains

# Load JSON
    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)

        shared_params = config["shared_params"]
        nwalkers = config["nwalkers"]
        out_dir = config["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        os.system(f"cp {args.config} {out_dir}/")
        nsteps = config["nsteps"]
        
    data_dict, src_list, models = setup_run(src_name, shared_params, config=config)

    out_name = src_name

    f = fit(data_dict, seed=seed, nsteps=nsteps, nwalkers=nwalkers, nchains=nchains, src_param=src_list, models=models, out_name=out_name, folder=out_dir, test_llh=args.test_llh)

    f.run_all_chains(resume=(not args.reseed_run), init=initial_run, have_proton=(not args.have_proton))
