import numpy as np
from functions import *

def custom_log_bins(E_min, E_low_break, E_high_break, E_max,
                                    n_low, n_mid, n_high):
    """
    Create log-spaced energy bin edges with two custom break points.

    Parameters:
        E_min (float): Minimum energy (eV)
        E_low_break (float): First break energy (eV)
        E_high_break (float): Second break energy (eV)
        E_max (float): Maximum energy (eV)
        n_low (int): Number of bins from E_min to E_low_break
        n_mid (int): Number of bins from E_low_break to E_high_break
        n_high (int): Number of bins from E_high_break to E_max

    Returns:
        np.ndarray: Full set of log-spaced bin edges
    """
    assert E_min < E_low_break < E_high_break < E_max, "Check energy boundaries"

    bins_low = np.logspace(np.log10(E_min), np.log10(E_low_break), n_low + 1)
    bins_mid = np.logspace(np.log10(E_low_break), np.log10(E_high_break), n_mid + 1)[1:]
    bins_high = np.logspace(np.log10(E_high_break), np.log10(E_max), n_high + 1)[1:]

    return np.concatenate([bins_low, bins_mid, bins_high])

class Parameter:
    def __init__(self, name, initial, bounds, is_free=True):
        self.name = name
        self.value = initial      # Initial value
        self.min = bounds[0]      # Lower bound
        self.max = bounds[1]      # Upper bound
        self.is_free = is_free    # Free/fixed status

    def __repr__(self):
        return f"Parameter({self.name}, value={self.value}, free={self.is_free}, bounds=({self.min}, {self.max}))"

shared_params_BPL = {
    'log_xi_p': {'values': [0., -1, 2], 'free': True},
    'log_xi_B': {'values': [0., -1, 2], 'free': True},
    'log_xi_e': {'values': [0., -1, 2], 'free': True},
    'log_Radius': {'values': [14., 13., 16.], 'free': True},
    'Gamma_j': {'values': [5., 2., 10.], 'free': True},
    'spectral_p': {'values': [2., 1.9, 2.2], 'free': False},
    'spectral_l_e': {'values': [1., 0., 2.1], 'free': False},
    'spectral_h_e': {'values': [3., 2.1, 4.], 'free': True},
    'log_eps_e_break': {'values': [9., 8, 12], 'free': False},
}

shared_params = {
    'log_xi_B': {'values': [0., -2, 2], 'free': True},
    #'log_xi_e': {'values': [0., -2, 2], 'free': True},
    'log_xi_e': {'values': [0., -1, 2], 'free': True},
    'log_Radius': {'values': [14., 11.5, 16], 'free': True},
    'Gamma_j': {'values': [5., 2, 20.], 'free': True},
    'Gamma_sh': {'values': [2, 1.5, 10], 'free': False},
    'spectral_e': {'values': [2.1, 2., 3.], 'free': True},
    #'log_gamma_e_min': {'values': [3., 1.5, 4], 'free': True},
    'eps_e': {'values': [0.2, 0.01, 0.3], 'free': True},
    'log_xi_p': {'values': [0., -2, 2], 'free': True},
    'spectral_p': {'values': [2., 1., 3.], 'free': False},
    'eta': {'values': [10., 1., 20], 'free': False},
}

def create_parameter_dict(shared_param):
    theta_dict = {}
    for name, config in shared_param.items():
        initial, min_val, max_val = config['values']
        theta_dict[name] = Parameter(
            name=name,
            initial=initial,
            bounds=(min_val, max_val),
            is_free=config['free']
        )
    return theta_dict

theta_dict = create_parameter_dict(shared_params)

Mpc = 3.08e24
source_dict = {
'190829A': {
    'name' :    '190829A',
    'z'    :    0.0785,
    'dl'   :    358.4*Mpc,
    'logLiso': 48.3, #syu
#    'logLiso':  np.log10(1.68e-8*4*np.pi*(358.4*Mpc)**2),
    #'logLiso':  49.47, #Chand+20
    'T_90' :    56.9, # how to got it? it is 10s in Chand+20
    #'E_min': 8e3, #eV #Chand+20 GBM
    'E_min': 1e3, #syu
    'E_max': 10000e3,#syu
    #'E_max': 1000e3, #eV #Chand+20 GBM
    'alpha': 1.0, #syu
    'Epk_obs':  25.2e3,
    'func_form': 'CPL',
    #'alpha': 1.61, #Chand+20
    #'beta': 2.67, #Chand+20
    #'Epk_obs':  12e3, #eV #Chand+20
    #'func_form': 'Band',
    'shared_param': theta_dict,
    'Dec': -9.0,
},

'120422A': {
    'name' :    '120422A',
    'z'    :    0.283,
    'dl'   :    1464.5*Mpc,
    'logLiso':  np.log10(1.8e48),
    'T_90' :    60.4,
    'Epk_obs': 33.3e3, #eV ##
    'E_min': 1e3, #eV
    'E_max': 10000e3, #eV
    'alpha': 1.2,
    'func_form': 'CPL',
    'shared_param': theta_dict,
    'Dec': 14.0,
},

'201015A': {
    'name' :    '201015A',
    'z'    :    0.426,
    'dl'   :    2363.6*Mpc,
    'logLiso':  np.log10(5.9e49), #slchen, 2.25E-07 erg/cm^2 
    'T_90' :    9.8, # s
    'Epk_obs': 21.39e3, #eV 
    'E_min': 1e3, #eV
    'E_max': 10000e3, #eV
    'alpha': 1.9, #temperal value, no low energy measure
    'func_form': 'CPL',
    'shared_param': theta_dict,
    'Dec': 53.4,
},

'171205A': {
    'name' :    '171205A',
    'z'    :    0.0368,
    'dl'   :    163.*Mpc,
    'logLiso':  46.9, #47.2,
    'T_90' :    190.5,
    'Epk_obs': 134.1e3, #eV
    'E_min': 1e3, #eV
    'E_max': 10000e3, #eV
    'alpha': 1.0,
    'func_form': 'CPL',
    'shared_param': theta_dict,
    'Dec': -12.6,
},

'100316D': {
    'name' :    '100316D',
    'z'    :    0.059,
    'dl'   :    261.*Mpc,
    'logLiso':  46.5,
    'T_90' :    521.9,
    'Epk_obs': 21.6e3, #eV
    'E_min': 1e3, #eV
    'E_max': 10000e3, #eV
    'alpha': 1.0,
    'func_form': 'CPL',
#custom_log_bins(E_min=15e3, E_low_break=20e3, E_high_break=80e3, E_max=140e3, n_low=2, n_mid=10, n_high=4),
    'shared_param': theta_dict,
    'Dec': -56.3,
},

'060218' : {
    'name' :    '060218',
    'z'    :    0.0331,
    'dl'   :    147.*Mpc,
    'logLiso':  np.log10(1.2e46), #3.84e-9*4*np.pi*(147*Mpc)**2),
    'T_90' :    2100,
    'Epk_obs': 11.9e3, #eV
    'E_min': 1e3, #eV
    'E_max': 10000e3, #eV
    'alpha': -0.6,
    'func_form': 'CPL',
    'shared_param': theta_dict,
    'Dec': 16.9,
},

'050826': {
    'name' :    '050826',
    'z'    :    0.297,
    'dl'   :    1517.*Mpc,
    'logLiso':  49.1, #48.41,
    'T_90' :    35.5,
    'Epk_obs': 500, #178.701e3, #eV
    'E_min': 1e3, #1e4, #eV
    'E_max': 10000e3, #1e5, #eV
    'alpha': 1.0,
    'func_form': 'CPL',
    'shared_param': theta_dict,
    'Dec': 20.7,
}
}

sbo_dict = {
'100316D': source_dict['100316D'],
'060218': source_dict['060218']
}
