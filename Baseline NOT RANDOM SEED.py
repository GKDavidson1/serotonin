
from __future__ import division
import argparse
from datetime import datetime
import json
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
import numpy.matlib

# %matplotlib inline
import matplotlib.pyplot as plt
import pandas
import scipy.io as sio
import brian2
import brian2 as b2
from joblib import Parallel, delayed, wrap_non_picklable_objects

try:
    import cma
except ImportError:  # pragma: no cover - handled when optimisation is invoked
    cma = None

_ANATOMY_CACHE = None
_FC_TARGET_CACHE = {}
DEFAULT_GLOBAL_SEED = 12345*10
DEFAULT_BATCH_RUNS = 1
DEFAULT_SEEDED_LOG_ROOT = Path('SEEDED LOGS Test')


PARAMS = {
    # local connectivity strengths

          # local strengths E-->
          'g_e_self': 0.18 * brian2.nA,
          'g_e_cross': 0 * brian2.nA,
          'g_pv_e' : 0.174   * brian2.nA,
          'g_sst_e_self' : 0.0435   * brian2.nA,
          'g_sst_e_cross' : 0.0435   * brian2.nA,
          'g_vip_e' : 0.058   * brian2.nA,
          # local strengths PV-->
          'g_e_pv_min': -0.001 * brian2.nA, # dopamine dependent min PV->E strength
          'g_e_pv_max': -0.4 * brian2.nA, # dopamine dependent max PV->E strength #<------------------------
          'g_pv_self': -0.18 * brian2.nA,
          # local strengths SST-->
          'g_e_sst_min': -0.09 * brian2.nA, # dopamine dependent min SST->E strength #<--------------------------
          'g_e_sst_max': -0.11 * brian2.nA, # dopamine dependent max SST->E strength
          'g_pv_sst': -0.17 * brian2.nA,
          'g_vip_sst': -0.1 * brian2.nA,
          # local strengths VIP-->
          'g_sst_vip': -0.05 * brian2.nA,

    # Time constants
          'tau_nmda': 60 * brian2.ms,
          'tau_ampa': 2 * brian2.ms,
          'tau_gaba': 5 * brian2.ms,
          'tau_gaba_dend': 10 * brian2.ms,
          'tau_adapt': 0.1   * brian2.second,

    # synaptic rise constants
          'gamma_nmda': 1.282, # unitless
          'gamma_gaba': 2,
          'gamma_ampa': 5,

    # AMPA/(AMPA+NMDA) fraction
          'ampa_frac_pv': 0.2, # AMPA fraction (1-NMDA fraction) for PV cells
          'ampa_frac': 0.1,    # AMPA fraction (1-NMDA fraction) for all other cell types

    # dendrite I/O function parameters
          'c1': 120 * brian2.pA,
          'c2': 136.24 * brian2.pA,
          'c3': 7.0,
          'c4': 0 * brian2.pA,
          'c5': 9.64 * brian2.pA,
          'c6': 20 * brian2.pA,

    # adaptation strengths
          'g_adapt_e': -0.004 * brian2.nA,
          'g_adapt_sst': -0.004 * brian2.nA,
          'g_adapt_vip': -0.004 * brian2.nA,

    # f-I curve parameters - E populations
          'a_e': 0.135 * brian2.Hz / brian2.pA,
          'b_e': 54 * brian2.Hz,
          'd_e': 0.308 * brian2.second,

    # f-I curve parameters - I populations
          'c_I_sst': 132 * brian2.Hz / brian2.nA,
          'c_I_vip': 132 * brian2.Hz / brian2.nA,
          'c_I_pv': 330 * brian2.Hz / brian2.nA,
          'r_0_sst': -33 * brian2.Hz,
          'r_0_vip': -33 * brian2.Hz,
          'r_0_pv': -95 * brian2.Hz,


    # rescale FLN parameter
          'b1': 0.3, #default
          # 'b1': 0.25,

    # Long-range E targets
    #       'lr_e_self_dend': 0.9, #concern
          'lr_e_self_dend': 1.0,
          # 'lr_e_cross_dend': 0.1,

    # Long-range I targets
          'lr_pv_e': 0.31,
          'lr_sst_e_self': 0.22 ,
          'lr_vip_e_self': 0.47,

    # Long-range I targets in FEF
          'lr_pv_e_fef': 0.2,
          'lr_sst_e_self_fef': 0.1 ,
          'lr_vip_e_self_fef': 0.7,

    # parameters for D1 occupancy
          'slope_d1occ' : 2,
          'midpoint_d1occ' : 1,

    # parameters for DA modulation of NMDA
          'slope_nmda_da' : 10,
          'midpoint_nmda_da' : 0.35,
          'g_nmda_da': 0.6,

    # parameters for m current
          'slope_m' : 14,
          'midpoint_m' : 0.85,
          'g_m' : -0.5  * brian2.nA,

    # Simulation parameters
        # initial firing rates
        #   'r_0_e': 5 * brian2.Hz, #default
          'r_0_e': 0 * brian2.Hz,
        # timestep
        #   'dt': 0.5 * brian2.ms, #default
          'dt': 1.0 * brian2.ms,
        # trial length

         # target stimulus on time
          'stim_on': 1 * brian2.second,
         'stim_off': 1.4 * brian2.second,

        # distractor stimulus on time (if using)
          'distract_on': 2 * brian2.second,
        # distractor stimulus off time (if using)
          'distract_off': 2.4 * brian2.second,


    # background inputs

          # 'I_background_e': 310 * brian2.pA, #default
          'I_background_e': 306 * brian2.pA,
          # 'I_background_i': 300 * brian2.pA, #default
          'I_background_i': 280 * brian2.pA,
          # 'I_background_dend': 30 * brian2.pA, #default
          'I_background_dend': 24 * brian2.pA,

          # 'tau_noise': 2 * 8 * brian2.ms,

    # stimulus strength
    #       'stim_strength': 0.1 * brian2.nA, #default
          'stim_strength': 0.0 * brian2.nA,


    # dopamine release level for the current simulation
    #     'da_rel': 1.5, #default
          'da_rel': 0.5,

          'std_noise': 4 * brian2.pA,
          # 'std_noise': 0.0 * brian2.pA,

          # 'trial_length': 40 * brian2.second,
          'trial_length': 15 * brian2.second,
        # Long-range connectivity strengths
        #   'mu_ee': 1.45, #default
            'mu_ee': 1.46,
        #   'mu_ie': 2.24, #default
            'mu_ie': 4.1,

# Min excitatory gradient (spine count) value
#           'e_grad_min': 0.45, #default
          'e_grad_min': 0.235 #acceptable ranges from 0.1-0.2

}



def current_to_frequency(input_current,population_type,parameters):
    if population_type == 'E':
        a = parameters['a_e']
        b = parameters['b_e']
        d = parameters['d_e']
        return np.divide((a*input_current - b),(1 - np.exp(-d*(a*input_current - b))))
    if population_type == 'PV':
        c_I = parameters['c_I_pv']
        r_0 = parameters['r_0_pv']
        r = np.maximum(c_I*input_current + r_0,0)
        return r
    if population_type == 'SST':
        c_I = parameters['c_I_sst']
        r_0 = parameters['r_0_sst']
        r = np.maximum(c_I*input_current + r_0,0)
        return r
    if population_type == 'VIP':
        c_I = parameters['c_I_vip']
        r_0 = parameters['r_0_vip']
        r = np.maximum(c_I*input_current + r_0,0)
        return r

def dendrite_input_output(exc_current,inh_current,parameters):
    c1 = parameters['c1']
    c2 = parameters['c2']
    c3 = parameters['c3']
    c4 = parameters['c4']
    c5 = parameters['c5']
    c6 = parameters['c6']

    beta = c5*np.exp(-inh_current/c6)

    return c1*(np.tanh((exc_current +c3*inh_current + c4)/beta)) + c2

def NMDA_deriv(S_NMDA_prev,rate_now,parameters):

    return -S_NMDA_prev/parameters['tau_nmda'] + parameters['gamma_nmda']*(1 - S_NMDA_prev)*rate_now

def AMPA_deriv(S_AMPA_prev,rate_now,parameters):

    return -S_AMPA_prev/parameters['tau_ampa'] + parameters['gamma_ampa']*rate_now

def GABA_deriv(S_GABA_prev,rate_now,parameters,cell_section):
    if cell_section == 'soma':
        return -S_GABA_prev/parameters['tau_gaba'] + parameters['gamma_gaba']*rate_now
    elif cell_section == 'dendrite':
        return -S_GABA_prev/parameters['tau_gaba_dend'] + parameters['gamma_gaba']*rate_now
    
def adaptation_deriv(S_a_prev,rate_now,parameters):
    return -S_a_prev/parameters['tau_adapt'] + rate_now

def sigmoid_DA(height,midpoint,slope):
     return np.exp(slope*(height-midpoint))/(1 + np.exp(slope*(height-midpoint)))


def load_anatomy():
    global _ANATOMY_CACHE
    if _ANATOMY_CACHE is not None:
        return _ANATOMY_CACHE

    # Load in anatomical data file
    subgraph_data = sio.loadmat('anatomical_data/beta_bin_hierarchy_subgraph.mat')
    sln = subgraph_data['HierOrderedSLNsubgraph']
    fln = subgraph_data['HierOrderedFLNsubgraph']
    hierarchy = subgraph_data['hierarchy_vals_subgraph']


    temp_list = subgraph_data['subgraph_hierarchical_order']
    area_list_SLN = []
    for row in temp_list:
        v = '%s' % str(row[0][0])
        area_list_SLN.append(v)

    area_column_list  = ['from '+ mystring for mystring in area_list_SLN]
    area_row_list  = ['to '+ mystring for mystring in area_list_SLN]

    df_fln = pandas.DataFrame(fln , columns=area_column_list, index=area_row_list)

    df_sln = pandas.DataFrame(sln , columns=area_column_list, index=area_row_list)

    # load the receptor data
    D1R_data = sio.loadmat('anatomical_data/D1R_lyon_regions.mat')

    D1_density_raw = D1R_data['D1R_lyon_regions_40']

    # load the spine count data
    spine_data = sio.loadmat('anatomical_data/spine_count_lyon_regions.mat')

    spine_count_raw = spine_data['spine_count_lyon_regions_40']

    df_raw_anatomy = pandas.DataFrame(D1_density_raw, columns=['D1R'], index=area_list_SLN)
    df_raw_anatomy.loc[:,'spines'] = spine_count_raw
    df_raw_anatomy.loc[:,'hierarchy'] = hierarchy

    
    _ANATOMY_CACHE = (
        sln,
        fln,
        hierarchy,
        area_list_SLN,
        df_fln,
        df_sln,
        D1_density_raw,
        spine_count_raw,
        df_raw_anatomy,
    )
    return _ANATOMY_CACHE







def prepare_connectivity(parameters,spine_count_raw,fln,sln,d1_density_raw):

    d1_occ = sigmoid_DA(parameters['da_rel'],parameters['midpoint_d1occ'],parameters['slope_d1occ'])

    ######## Excitatory gradient ########
    # scale spine count to lie within [0,1] range
    min_spine_count = np.min(spine_count_raw)
    spine_count_rescaled = spine_count_raw-min_spine_count
    spine_grad = spine_count_rescaled/np.max(spine_count_rescaled)

    # define the excitatory gradient to lie according to the spine count 
    e_grad_scaling_factor = 1 - parameters['e_grad_min'] 
    e_grad = parameters['e_grad_min'] + e_grad_scaling_factor*spine_grad


    ######## Local connectivity ########
    # set up the local connectivity matrix
    J =  np.array([#soma dend pv sst vip
        [parameters['g_e_self'] , 0, parameters['g_pv_e'], parameters['g_sst_e_self'],parameters['g_vip_e']],

                [0,0,0,0,0],

                  [parameters['g_e_pv_min'],0,parameters['g_pv_self'], 0, 0],

                  [0,0,parameters['g_pv_sst'], 0,parameters['g_vip_sst']],


                  [0,0,0,parameters['g_sst_vip'],0],

                  ]).T * brian2.amp


    pops = ['E1soma','E1dend','PV','SST1','VIP1']
    pops_column_list  = ['from '+ mystring for mystring in pops]
    pops_row_list  = ['to '+ mystring for mystring in pops]

    J_display = J*(1/brian2.pA)
    df_J = pandas.DataFrame(J_display, columns=pops_column_list, index=pops_row_list)
    df_J

    ######### numbers of areas, populations ##########

    num_pops  = J.shape[0]
    num_e_pops = 1
    num_areas = fln.shape[0]

    ######### adaptation ###########
    g_adapt = np.array([parameters['g_adapt_e'],
                        0,
                        0,
                        parameters['g_adapt_sst'],
                        parameters['g_adapt_vip']])* brian2.amp

    
    g_m = np.array([parameters['g_m'],0,0,0,0])* brian2.amp
    
    ######### AMPA/(AMPA+NMDA) fraction ##########

    ampa_frac = np.array([parameters['ampa_frac'],
                          parameters['ampa_frac'],
                          parameters['ampa_frac_pv'],
                          parameters['ampa_frac'],
                          parameters['ampa_frac']])
    nmda_frac = 1 - ampa_frac

    J_nmda = J*((J>0).astype(int))
    J_ampa = J*((J>0).astype(int))
    J_gaba = J*((J<0).astype(int))

    J_gaba_dend =  np.array([[0,
                              0,
                              0,
                              parameters['g_e_sst_min'],
                              0],

                             ]) * brian2.amp

    ####### LONG-RANGE CONNECTIONS ########
    # Compress FLN
    fln_squish = np.power(fln,parameters['b1'])
    fln_rowtotal = np.sum(fln_squish,axis=1)
    fln_rowtotal_mat = np.matlib.repmat(fln_rowtotal, num_areas,1).T
    fln_squishnorm = fln_squish/fln_rowtotal_mat

    # Isolate long-range connections from superficial layers
    W_superficial = fln_squishnorm*sln
    # Isolate long-range connections from deep layers
    W_deep = fln_squishnorm*(1-sln)


    # This matrix splits the long-range current onto each local population of cells
    lr_targets = np.array([[0,
                            parameters['lr_e_self_dend'],
                            parameters['lr_pv_e'],
                            parameters['lr_sst_e_self'],
                            parameters['lr_vip_e_self']],
                           ]).T * brian2.nA
    
    # This matrix splits the long-range current onto each local population of cells - reflecting greater proportion of CR cells in FEF (Pouget et al., 2009)
    lr_targets_FEF = np.array([[0,
                                parameters['lr_e_self_dend'],
                                parameters['lr_pv_e_fef'],
                                parameters['lr_sst_e_self_fef'],
                                parameters['lr_vip_e_self_fef']],


                                ]).T * brian2.nA

    ##### Dopamine modulation #####
    # scale_receptors to lie within [0,1] range
    min_d1R = np.min(d1_density_raw)
    d1R_rescaled = np.squeeze(d1_density_raw)-min_d1R
    d1_grad = d1R_rescaled/np.max(d1R_rescaled)

    # strength of excitatory currents through NMDA receptors increases with dopamine (Seamans et al., PNAS, 2001)
    # To remove effect of dopamine on NMDA, while keeping other dopamine effects, set d1_occ here = 0
    nmda_da_grad = 1 + parameters['g_nmda_da']*sigmoid_DA(d1_occ*np.expand_dims(d1_grad,axis=1),parameters['midpoint_nmda_da'],parameters['slope_nmda_da'])

    # PV-->soma strength decreases with dopamine (Gao et al., J Neurosci, 2003)
    # To remove effect of dopamine on PV-->E connections, while keeping other dopamine effects, set d1_occ here = 0
    e_pv_da_grad = (parameters['g_e_pv_max'] + d1_occ*d1_grad*(parameters['g_e_pv_min'] - parameters['g_e_pv_max']))/parameters['g_e_pv_min']

    e_pv_da_mat = np.concatenate((np.expand_dims(e_pv_da_grad,axis=1),np.ones((num_areas,num_pops-num_e_pops))),axis=1)

    # SST-->dendrite strength increases with dopamine (Gao et al., J Neurosci, 2003)
    # To remove effect of dopamine on PV-->E connections, while keeping other dopamine effects, set d1_occ here = 0
    e_sst_da_grad = (parameters['g_e_sst_min'] + d1_occ*d1_grad*(parameters['g_e_sst_max'] - parameters['g_e_sst_min']))/parameters['g_e_sst_min']

    # e_sst_da_mat = np.concatenate((np.expand_dims(e_sst_da_grad,axis=1)))
    e_sst_da_mat = e_sst_da_grad[:, None]
    # High levels of D1 receptor stimulation engage an outward M-channel, reducing excitability (Arnsten et al., Neurobio. Stress., 2019)
    # To remove effect of dopamine on the M-channel, while keeping other dopamine effects, set d1_occ here = 0
    m_da_grad = sigmoid_DA(d1_occ*d1_grad,parameters['midpoint_m'],parameters['slope_m']).reshape(num_areas,1)

    return(pops, num_pops, num_e_pops, num_areas, e_grad, g_adapt, ampa_frac, nmda_frac, J_nmda, J_ampa, 
          J_gaba, J_gaba_dend, W_superficial, W_deep, lr_targets, nmda_da_grad, e_pv_da_mat, e_sst_da_mat, m_da_grad,g_m,lr_targets_FEF)






def initialise_simulation_state(PARAMS, num_areas, num_pops, area_list_SLN, pops, *, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    num_iterations = int(PARAMS['trial_length'] / PARAMS['dt'])

    R = np.matlib.repmat(
        np.array([PARAMS['r_0_e'], 0, PARAMS['r_0_e'], PARAMS['r_0_e'], PARAMS['r_0_e']]),
        num_areas,
        1,
    ) * brian2.Hz
    s_nmda = np.zeros((num_areas, num_pops))
    s_ampa = np.zeros((num_areas, num_pops))
    s_gaba = np.ones((num_areas, num_pops))
    s_gaba_dend = np.ones((num_areas, num_pops))
    s_adapt = np.zeros((num_areas, num_pops))

    I_0 = np.zeros((num_areas, num_pops)) * brian2.pA
    I_0[:, [pops.index('E1soma')]] = PARAMS['I_background_e']
    I_0[:, [pops.index('E1dend')]] = PARAMS['I_background_dend']
    I_0[:, [pops.index('PV'), pops.index('SST1'), pops.index('VIP1')]] = PARAMS['I_background_i']

    eta = rng.normal(loc=0.0, scale=1.0, size=(num_iterations, num_areas, num_pops))
    noise_rhs = eta * PARAMS["std_noise"] * np.sqrt(2.0 * PARAMS["dt"] / PARAMS["tau_ampa"])
    noise_rhs[:, :, 1:2] = 0
    I_noise = np.zeros((num_areas, num_pops)) * brian2.pA

    return {
        'num_iterations': num_iterations,
        'R': R,
        's_nmda': s_nmda,
        's_ampa': s_ampa,
        's_gaba': s_gaba,
        's_gaba_dend': s_gaba_dend,
        's_adapt': s_adapt,
        'I_0': I_0,
        'I_noise': I_noise,
        'noise_rhs': noise_rhs,
        'stim_start_idx': int(PARAMS['stim_on'] / PARAMS['dt']),
        'stim_end_idx': int(PARAMS['stim_off'] / PARAMS['dt']),
        'stim_area_idx': area_list_SLN.index('V1'),
        'stim_pop_idx': pops.index('E1dend'),
    }
        






def large_scale_da_model(
    pops,
    num_pops,
    num_areas,
    e_grad,
    g_adapt,
    g_m,
    ampa_frac,
    nmda_frac,
    J_nmda,
    J_ampa,
    J_gaba,
    J_gaba_dend,
    W_superficial,
    W_deep,
    lr_targets,
    nmda_da_grad,
    e_pv_da_mat,
    e_sst_da_mat,
    m_da_grad,
    state,
    parameters,
    lr_targets_FEF,
    area_list_SLN,
):

    num_iterations = state['num_iterations']
    R = state['R']
    s_nmda = state['s_nmda']
    s_ampa = state['s_ampa']
    s_gaba = state['s_gaba']
    s_gaba_dend = state['s_gaba_dend']
    s_adapt = state['s_adapt']
    I_0 = state['I_0']
    I_noise = state['I_noise']
    noise_rhs = state['noise_rhs']
    stim_start_idx = state['stim_start_idx']
    stim_end_idx = state['stim_end_idx']
    stim_area_idx = state['stim_area_idx']
    stim_pop_idx = state['stim_pop_idx']

    fef_idx = [area_list_SLN.index('8m'), area_list_SLN.index('8l')]
    area_drive_abs = np.zeros((num_iterations, num_areas)) * brian2.pA

    I_lr_nmda = np.zeros((num_areas, num_pops)) * brian2.pA
    I_lr_ampa = np.zeros((num_areas, num_pops)) * brian2.pA
    I_local_nmda = np.zeros((num_areas, num_pops)) * brian2.pA
    I_local_ampa = np.zeros((num_areas, num_pops)) * brian2.pA
    I_local_gaba = np.zeros((num_areas, num_pops)) * brian2.pA
    I_soma_dend = np.zeros((num_areas, num_pops)) * brian2.pA
    I_exc_dend = np.zeros((num_areas, 1)) * brian2.pA
    I_inh_dend = np.zeros((num_areas, 1)) * brian2.pA
    I_local_gaba_dend = np.zeros((num_areas, 1)) * brian2.pA
    I_adapt = np.zeros((num_areas, num_pops)) * brian2.pA
    I_total = np.zeros((num_areas, num_pops)) * brian2.pA
    R_next = np.zeros_like(R)

    dt = parameters['dt']
    tau_ampa = parameters['tau_ampa']
    stim_strength = parameters['stim_strength']

    for step in range(num_iterations):

        I_noise = I_noise + -I_noise * (dt / tau_ampa) + noise_rhs[step, :, :]

        I_lr_nmda[:, :] = 0 * brian2.pA
        I_lr_nmda[:, :2] = ((e_grad * parameters['mu_ee'] * nmda_da_grad) * W_superficial).dot(s_nmda[:, :1]).dot(
            nmda_frac[:2] * lr_targets[:2, :].T
        )
        I_lr_nmda[:, 2:] = parameters['mu_ie'] * e_grad * nmda_da_grad * (W_deep.dot(s_nmda[:, :1])).dot(
            nmda_frac[2:] * lr_targets[2:, :].T
        )
        I_lr_nmda[fef_idx, 2:] = (
            parameters['mu_ie']
            * e_grad[fef_idx]
            * nmda_da_grad[fef_idx]
            * (W_deep[fef_idx, :].dot(s_nmda[:, :1])).dot(nmda_frac[2:] * lr_targets_FEF[2:, :].T)
        )

        I_lr_ampa[:, :] = 0 * brian2.pA
        I_lr_ampa[:, :2] = ((e_grad * parameters['mu_ee']) * W_superficial).dot(s_ampa[:, :1]).dot(
            ampa_frac[:2] * lr_targets[:2, :].T
        )
        I_lr_ampa[:, 2:] = parameters['mu_ie'] * e_grad * (W_deep.dot(s_ampa[:, :1])).dot(
            ampa_frac[2:] * lr_targets[2:, :].T
        )
        I_lr_ampa[fef_idx, 2:] = (
            parameters['mu_ie']
            * e_grad[fef_idx]
            * (W_deep[fef_idx, :].dot(s_ampa[:, :1])).dot(ampa_frac[2:] * lr_targets_FEF[2:, :].T)
        )

        I_local_nmda[:, :] = nmda_frac * nmda_da_grad * e_grad * J_nmda.dot(s_nmda.T).T
        I_local_ampa[:, :] = ampa_frac * e_grad * J_ampa.dot(s_ampa.T).T
        I_local_gaba[:, :] = e_pv_da_mat * (J_gaba.dot(s_gaba.T).T)
        I_local_gaba_dend[:, :] = e_sst_da_mat * (J_gaba_dend.dot(s_gaba_dend.T).T)

        I_exc_dend[:, :] = (
            I_local_nmda[:, 1:2]
            + I_lr_nmda[:, 1:2]
            + I_local_ampa[:, 1:2]
            + I_lr_ampa[:, 1:2]
            + I_0[:, 1:2]
            + I_noise[:, 1:2]
        )
        if stim_start_idx <= step < stim_end_idx:
            I_exc_dend[stim_area_idx, 0] += stim_strength

        I_inh_dend[:, :] = I_local_gaba_dend
        I_soma_dend[:, :] = 0 * brian2.pA
        I_soma_dend[:, :1] = dendrite_input_output(I_exc_dend, I_inh_dend, parameters)
        I_adapt[:, :] = (g_adapt + g_m * m_da_grad) * s_adapt

        I_total[:, :] = (
            I_local_nmda
            + I_local_ampa
            + I_local_gaba
            + I_0
            + I_noise
            + I_lr_nmda
            + I_lr_ampa
            + I_soma_dend
            + I_adapt
        )
        if stim_start_idx <= step < stim_end_idx:
            I_total[stim_area_idx, stim_pop_idx] += stim_strength

        area_drive_abs[step, :] = (
            abs(I_local_nmda)
            + abs(I_local_ampa)
            + abs(I_local_gaba)
            + abs(I_0)
            + abs(I_noise)
            + abs(I_lr_nmda)
            + abs(I_lr_ampa)
            + abs(I_soma_dend)
            + abs(I_adapt)
        ).sum(axis=1)
        if stim_start_idx <= step < stim_end_idx:
            area_drive_abs[step, stim_area_idx] += abs(stim_strength)

        if step == num_iterations - 1:
            break

        R_next[:, :1] = R[:, :1] + dt * current_to_frequency(I_total[:, :1], 'E', parameters) / tau_ampa - dt * R[:, :1] / tau_ampa
        R_next[:, 1:2] = 0 * brian2.Hz
        R_next[:, 2] = R[:, 2] + dt * current_to_frequency(I_total[:, 2], 'PV', parameters) / tau_ampa - dt * R[:, 2] / tau_ampa
        R_next[:, 3:4] = R[:, 3:4] + dt * current_to_frequency(I_total[:, 3:4], 'SST', parameters) / tau_ampa - dt * R[:, 3:4] / tau_ampa
        R_next[:, 4:] = R[:, 4:] + dt * current_to_frequency(I_total[:, 4:], 'VIP', parameters) / tau_ampa - dt * R[:, 4:] / tau_ampa

        s_nmda[:, :1] = s_nmda[:, :1] + dt * NMDA_deriv(s_nmda[:, :1], R_next[:, :1], parameters)
        s_ampa[:, :1] = s_ampa[:, :1] + dt * AMPA_deriv(s_ampa[:, :1], R_next[:, :1], parameters)
        s_gaba[:, 2:] = s_gaba[:, 2:] + dt * GABA_deriv(s_gaba[:, 2:], R_next[:, 2:], parameters, 'soma')
        s_gaba_dend[:, 2:] = s_gaba_dend[:, 2:] + dt * GABA_deriv(s_gaba_dend[:, 2:], R_next[:, 2:], parameters, 'dendrite')
        s_adapt[:, :] = s_adapt[:, :] + dt * adaptation_deriv(s_adapt[:, :], R_next[:, :], parameters)
        R, R_next = R_next, R

    return area_drive_abs


# Analysis helpers used by the FC / plotting workflow.
def plot_all_areas_2col(
    R,
    area_names,
    pops,
    pops_to_show,
    *,
    dt,
    t_start=0.0,
    t_end=None,
    stim_on=None,
    stim_off=None,
    ylim=(0, 15),
    figsize_per_row=2.6,
    dpi=120,
):
    import math

    # --- dt in seconds (float)
    dt_s = float(dt / b2.second) if hasattr(dt, "unit") else float(dt)

    R = np.asarray(R)
    T, A, P = R.shape

    # --- helper: convert time-like to seconds float
    def to_s(x):
        if x is None:
            return None
        return float(x / b2.second) if hasattr(x, "unit") else float(x)

    t_start_s = to_s(t_start)
    t_end_s = to_s(t_end) if t_end is not None else (T - 1) * dt_s
    stim_on_s, stim_off_s = to_s(stim_on), to_s(stim_off)

    # --- indices from times
    i0 = max(0, int(np.round(t_start_s / dt_s)))
    i1 = min(T, int(np.round(t_end_s / dt_s)))
    if i1 <= i0:
        raise ValueError(
            f"Empty plotting window: t_start={t_start_s}s, t_end={t_end_s}s, dt={dt_s}s"
        )

    # --- time axis aligned to samples
    t = np.arange(i0, i1) * dt_s

    # --- pop indices once
    pop_idx = {}
    for name in pops_to_show:
        if name not in pops:
            raise ValueError(f"Population '{name}' not found in pops list.")
        pop_idx[name] = pops.index(name)

    # --- 2-column layout
    ncols = 2
    nrows = math.ceil(A / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(16, figsize_per_row * nrows),
        sharex=False,
        dpi=dpi,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for a in range(A):
        r = a // ncols
        c = a % ncols
        ax = axes[r, c]

        ax.set_title(area_names[a])

        for pop_name in pops_to_show:
            p = pop_idx[pop_name]
            ax.plot(t, R[i0:i1, a, p], label=pop_name)

        if stim_on_s is not None and stim_off_s is not None:
            ax.axvspan(stim_on_s, stim_off_s, color="red", alpha=0.15)

        ax.set_ylim(*ylim)
        ax.set_ylabel("Rate (Hz)")
        ax.legend(loc="upper right", fontsize=9)

    # hide unused axes (if odd number of areas)
    for a in range(A, nrows * ncols):
        r = a // ncols
        c = a % ncols
        axes[r, c].axis("off")

    # ✅ DIFFERENT METHOD: reserve bottom margin + label bottom-row axes
    # fig.subplots_adjust(bottom=0.08, hspace=0.35, wspace=0.25)

    bottom_row = nrows - 1
    for c in range(ncols):
        ax = axes[bottom_row, c]
        if ax.axison:  # don’t label hidden axes
            ax.set_xlabel("Time (s)", labelpad=10)
            ax.tick_params(axis="x", labelbottom=True)

    return fig, axes






def acf_1d(x, max_lag, *, fft=True, missing="none"):
    from statsmodels.tsa.api import acf as sm_acf

    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= 1:
        return np.full(max_lag + 1, np.nan)

    # constant / zero-variance => undefined ACF (match your previous behavior)
    if np.allclose(x, x[0]):
        return np.full(max_lag + 1, np.nan)

    # Statsmodels includes lag 0 and returns length nlags+1
    out = sm_acf(
        x,
        nlags=max_lag,
        adjusted=False,   # denom uses (n-k) like your code
        fft=fft,
        missing=missing  # "none" (default). Use "drop"/"conservative" if NaNs exist.
    )

    # (usually unnecessary, but keeps it robust)
    if out.size != max_lag + 1:
        out = np.pad(out, (0, max_lag + 1 - out.size), constant_values=np.nan)

    return out

def plot_acf_oneplot_gradient(
    R, area_names, pops, pop_name="E1soma", *, dt, max_lag_ms=1000,
    cmap_name="viridis", alpha=0.45, lw=1.0, ylim=(-0.2, 1.0),
    add_colorbar=True, show_name_ticks=True
):
    pop_idx = pops.index(pop_name)

    dt_s = float(dt / b2.second) if hasattr(dt, "unit") else float(dt)

    R_np = np.asarray(R)
    rates = R_np[:, :, pop_idx]
    if hasattr(rates, "unit"):
        rates = rates / b2.Hz

    T, A = rates.shape
    max_lag = int(round((max_lag_ms / 1000.0) / dt_s))
    max_lag = min(max_lag, T - 1)
    lags_ms = (np.arange(max_lag + 1) * dt_s) * 1000.0

    acfs = np.zeros((A, max_lag + 1), dtype=float)
    for a in range(A):
        acfs[a] = acf_1d(rates[:, a], max_lag=max_lag)

    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=0, vmax=A-1)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    for a in range(A):
        ax.plot(lags_ms, acfs[a], color=cmap(norm(a)), alpha=alpha, lw=lw)

    ax.axhline(0, linewidth=0.8)
    ax.set_xlim(0, max_lag_ms)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("ACF (normalized, demeaned)")
    ax.set_title(f"Autocorrelation — all {A} areas (pop: {pop_name})")

    if add_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Area index")

        if show_name_ticks and area_names is not None:
            # For 40 areas, show ~8 labeled ticks
            tick_idx = np.linspace(0, A-1, num=8, dtype=int)
            cbar.set_ticks(tick_idx)
            cbar.set_ticklabels([str(area_names[i]) for i in tick_idx])

    fig.tight_layout()
    plt.show()


def bin_firing_rate(
    r, *, dt_ms=5.0, bin_ms=50.0, mode="mean"
):
    """
    Bin/aggregate a firing-rate time series onto coarser bins.

    Parameters
    ----------
    r : array-like
        Firing rate over time. Can be:
          - shape (T,) for a single series
          - shape (T, A) for multiple areas
          - shape (T, A, P) for areas x populations
    dt_ms : float
        Original timestep in milliseconds (e.g., 5.0).
    bin_ms : float
        Desired bin width in milliseconds (e.g., 50.0).
    mode : {"mean", "sum"}
        How to aggregate within each bin:
          - "mean": average rate within the bin (keeps units as Hz)
          - "sum": sum within the bin (not usually meaningful for rates unless you convert to counts)

    Returns
    -------
    r_binned : ndarray
        Binned array with time dimension reduced to T_bin.
    dt_bin_ms : float
        New effective timestep in ms (approximately bin_ms, exact multiple of dt_ms).
    """
    r = np.asarray(r, dtype=float)

    if dt_ms <= 0:
        raise ValueError("dt_ms must be > 0")
    if bin_ms <= 0:
        raise ValueError("bin_ms must be > 0")

    bin_steps = int(round(bin_ms / dt_ms))
    bin_steps = max(1, bin_steps)

    # Actual bin width after rounding to an integer number of steps
    dt_bin_ms = bin_steps * dt_ms

    T = r.shape[0]
    T_use = (T // bin_steps) * bin_steps  # truncate to full bins
    if T_use < bin_steps:
        raise ValueError("Time series too short for the requested bin size.")

    r_trim = r[:T_use]
    new_shape = (T_use // bin_steps, bin_steps) + r.shape[1:]
    r_reshaped = r_trim.reshape(new_shape)

    if mode == "mean":
        r_binned = r_reshaped.mean(axis=1)
    elif mode == "sum":
        r_binned = r_reshaped.sum(axis=1)
    else:
        raise ValueError("mode must be 'mean' or 'sum'")

    return r_binned, dt_bin_ms




def itotal_to_drive_abs(I_total, *, unit=brian2.pA, gain=1.0):
    """
    Translate synaptic currents I_total(t, area, pop) into a per-area neural drive x(t, area).

    You requested:
      drive(t, area) ∝ sum_p | I_total(t, area, p) |

    Parameters
    ----------
    I_total : array-like (T, A, P)
        Brian2 quantity array with units of current (e.g. pA).
    unit : brian2 unit (default: pA)
        Unit used to convert I_total into plain floats. Using pA keeps numbers nicely scaled.
    gain : float
        Scalar mapping from (current in 'unit') to dimensionless drive.

    Returns
    -------
    drive : np.ndarray (T, A), dtype=float
        Dimensionless neural drive suitable as input to a Balloon/BOLD forward model.
    """

    if I_total.ndim != 3:
        raise ValueError(f"I_total must be 3D (T,A,P). Got shape {I_total.shape}")

    # Absolute current, preserve Brian2 units
    I_abs = np.abs(I_total)               # (T, A, P)

    # Sum across populations -> per-area magnitude
    I_area_abs = I_abs.sum(axis=2)        # (T, A), still Brian2 quantity
    # print(I_area_abs)
    # Convert to floats in chosen unit, then scale to drive
    drive = gain * (I_area_abs / unit).astype(float)   # (T, A)

    return drive


def balloon_bold_per_area(drive, dt, *, method="rk4", **balloon_params):
    """
    Run the Balloon/BOLD forward model independently for each area.

    Parameters
    ----------
    drive : np.ndarray (T, A)
        Dimensionless neural drive per area.
    dt : float
        Time-step in seconds for integration.
    method : str
        "euler" or "rk4"
    balloon_params : dict
        Parameters passed to balloon_bold (tau, alpha, E0, V0, TE, epsilon, etc.)

    Returns
    -------
    bold : np.ndarray (T, A)
        BOLD signal (ΔS/S0) per area.
    states : list of dict
        Per-area state histories returned by balloon_bold.
    """
    X = np.asarray(drive, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"drive must be 2D (T,A). Got shape {X.shape}")

    T, A = X.shape
    bold = np.zeros((T, A), dtype=float)
    states = []

    for a in range(A):
        b, st = balloon_bold(X[:, a], dt, method=method, **balloon_params)
        bold[:, a] = b
        states.append(st)

    return bold, states



def balloon_bold(
    current,
    dt,
    *,
    kappa=0.65,
    gamma=0.41,
    tau=0.98,
    alpha=0.32,
    E0=0.34,
    V0=0.02,
    TE=0.04,
    epsilon=1.0,
    theta0=40.3,
    r0=25.0,
    neural_gain=1.0,
    method="rk4",
    max_step=0.05,
):
    u = np.asarray(current, dtype=float)
    n = u.size

    # Initial conditions (rest)
    s, f, v, q = 0.0, 1.0, 1.0, 1.0

    x_hist = np.empty(n)
    s_hist = np.empty(n)
    f_hist = np.empty(n)
    v_hist = np.empty(n)
    q_hist = np.empty(n)
    bold = np.empty(n)

    k1 = 4.3 * theta0 * E0 * TE
    k2 = epsilon * r0 * E0 * TE
    k3 = 1.0 - epsilon

    def oxygen_extraction(f_):
        f_safe = max(1e-6, float(f_))
        return 1.0 - (1.0 - E0) ** (1.0 / f_safe)

    def deriv(state, x_):
        s_, f_, v_, q_ = state

        ds = x_ - kappa * s_ - gamma * (f_ - 1.0)
        df = s_

        # Guard v before using fractional power (prevents nan in intermediate RK4 states)
        v_safe = max(1e-6, float(v_))
        v_out = v_safe ** (1.0 / alpha)

        dv = (f_ - v_out) / tau

        E = oxygen_extraction(f_)
        dq = (f_ * (E / E0) - v_out * (q_ / v_safe)) / tau

        return np.array([ds, df, dv, dq], dtype=float)

    def bold_from_vq(v_, q_):
        v_safe = max(1e-6, float(v_))
        return V0 * (k1 * (1.0 - q_) + k2 * (1.0 - q_ / v_safe) + k3 * (1.0 - v_safe))

    # Internal step splitting
    dt = float(dt)
    max_step = float(max_step)
    n_sub = max(1, int(np.ceil(dt / max_step)))
    h = dt / n_sub  # internal step

    def rk4_step(y, x_, h_):
        k_1 = deriv(y, x_)
        k_2 = deriv(y + 0.5 * h_ * k_1, x_)
        k_3 = deriv(y + 0.5 * h_ * k_2, x_)
        k_4 = deriv(y + h_ * k_3, x_)
        return y + (h_ / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

    for t in range(n):
        x = neural_gain * u[t]

        x_hist[t] = x
        s_hist[t] = s
        f_hist[t] = f
        v_hist[t] = v
        q_hist[t] = q
        bold[t] = bold_from_vq(v, q)

        # integrate over dt using substeps
        y = np.array([s, f, v, q], dtype=float)
        for _ in range(n_sub):
            if method.lower() == "euler":
                y = y + h * deriv(y, x)
            elif method.lower() == "rk4":
                y = rk4_step(y, x, h)
            else:
                raise ValueError("method must be 'euler' or 'rk4'")

            # clamp each substep so intermediate states stay physical
            y[1] = max(1e-6, y[1])  # f
            y[2] = max(1e-6, y[2])  # v
            y[3] = max(1e-6, y[3])  # q

        s, f, v, q = y.tolist()

    states = {"x": x_hist, "s": s_hist, "f": f_hist, "v": v_hist, "q": q_hist}
    return bold, states



def downsample_by_block_mean(x, k, *, axis=0):
    """
    Downsample by averaging over non-overlapping blocks of length k.

    Parameters
    ----------
    x : array-like
        Data with time on `axis`. Common shapes: (T, A) or (T, A, P).
    k : int
        Number of original timesteps per new timestep.
    axis : int
        Time axis (default 0).

    Returns
    -------
    y : np.ndarray
        Downsampled array with length floor(T/k) along `axis`.
    """
    x = np.asarray(x)
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if x.shape[axis] < k:
        raise ValueError(f"Not enough samples along axis {axis} to downsample by k={k}.")

    # Move time axis to front
    x0 = np.moveaxis(x, axis, 0)  # shape (T, ...)
    T = x0.shape[0]
    T2 = T // k  # truncate remainder
    x0 = x0[:T2 * k]

    # Reshape into blocks and average
    new_shape = (T2, k) + x0.shape[1:]
    y0 = x0.reshape(new_shape).mean(axis=1)

    # Move time axis back
    y = np.moveaxis(y0, 0, axis)
    return y


def downsample_to_dt(x, dt_old, dt_new, *, axis=0, require_integer_ratio=True):
    """
    Downsample `x` from dt_old to dt_new by block-mean averaging.

    Parameters
    ----------
    x : array-like
        Data with time on `axis`.
    dt_old : float or Brian2 quantity
        Original timestep.
    dt_new : float or Brian2 quantity
        Desired timestep (must be >= dt_old).
    axis : int
        Time axis (default 0).
    require_integer_ratio : bool
        If True, enforces dt_new/dt_old to be (near) an integer.

    Returns
    -------
    y : np.ndarray
        Downsampled data.
    k : int
        The block size used (number of old steps per new step).
    dt_new_s : float
        New timestep in seconds (float).
    """
    # Convert dt to seconds floats
    dt_old_s = float(dt_old / b2.second) if hasattr(dt_old, "unit") else float(dt_old)
    dt_new_s = float(dt_new / b2.second) if hasattr(dt_new, "unit") else float(dt_new)

    if dt_new_s < dt_old_s:
        raise ValueError(f"dt_new ({dt_new_s}) must be >= dt_old ({dt_old_s}).")

    ratio = dt_new_s / dt_old_s
    k = int(np.round(ratio))

    if require_integer_ratio and not np.isclose(ratio, k, rtol=1e-6, atol=1e-12):
        raise ValueError(
            f"dt_new/dt_old must be an integer for block-mean downsampling. "
            f"Got ratio={ratio} (nearest int {k})."
        )

    y = downsample_by_block_mean(x, k, axis=axis)
    return y, k, dt_new_s




def plot_bold(bold, dt, *, area_names=None, percent=True, max_areas=40):
    """
    Plot BOLD time series.

    bold : (T, A) array
    dt : timestep in seconds
    area_names : list of str, optional
    percent : plot in % signal change if True
    max_areas : limit number of plotted areas (for readability)
    """
    bold = np.asarray(bold)
    T, A = bold.shape
    t = np.arange(T) * dt

    if percent:
        bold_plot = 100 * bold
        ylabel = "BOLD (% ΔS/S₀)"
    else:
        bold_plot = bold
        ylabel = "BOLD (ΔS/S₀)"

    if area_names is None:
        area_names = [f"Area {i}" for i in range(A)]

    plt.figure(figsize=(10, 4))
    for a in range(min(A, max_areas)):
        plt.plot(t, bold_plot[:, a], label=area_names[a])

    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title("BOLD signal per area")
    # plt.legend()
    plt.tight_layout()
    plt.show()


def seed_based_fc(bold, seed_idx):
    bold = np.asarray(bold, dtype=float)
    seed = bold[:, seed_idx]
    if np.ndim(seed) == 1:
        seed_ts = seed
    else:
        seed_ts = np.mean(seed, axis=1)

    seed_std = np.std(seed_ts)

    fc = np.full(bold.shape[1], np.nan, dtype=float)
    if not np.isfinite(seed_std) or np.isclose(seed_std, 0.0):
        return fc

    for a in range(bold.shape[1]):
        area_ts = bold[:, a]
        area_std = np.std(area_ts)
        if not np.isfinite(area_std) or np.isclose(area_std, 0.0):
            continue
        corr = np.corrcoef(seed_ts, area_ts)[0, 1]
        if np.isfinite(corr):
            fc[a] = float(corr)

    return fc


import pandas as pd

def save_bold_csv(bold, dt, filename, *, area_names=None, percent=True):
    """
    Save BOLD time series to CSV.

    bold : (T, A) array
    dt : timestep in seconds
    filename : output CSV file
    area_names : list of str, optional
    percent : save in % signal change if True
    """
    bold = np.asarray(bold)
    T, A = bold.shape
    time = np.arange(T) * dt

    if percent:
        bold_out = 100 * bold
        unit = "%_BOLD"
    else:
        bold_out = bold
        unit = "BOLD"

    if area_names is None:
        area_names = [f"area_{i}" for i in range(A)]

    data = {"time_s": time}
    for a in range(A):
        data[f"{area_names[a]}_{unit}"] = bold_out[:, a]

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def drive_abs_to_balloon_input(
    drive_abs,
    *,
    baseline_idx,     # (i0, i1) indices defining baseline window
    gain=1.0,
    clamp_nonnegative=False,
    per_area=True
):
    """
    Convert a strictly-positive absolute drive into a baseline-referenced Balloon input x(t).

    x(t) = gain * (drive_abs(t) - drive0)

    Parameters
    ----------
    drive_abs : array (T, A) or (T,)
    baseline_idx : tuple(int,int)
        Baseline window indices (i0,i1)
    gain : float
        Scaling into Balloon's x units
    clamp_nonnegative : bool
        If True, x = max(0, x). (Not recommended for FC; use only if you insist on purely excitatory coupling.)
    per_area : bool
        If True, compute drive0 separately for each area. If False, use a single scalar baseline.

    Returns
    -------
    x : np.ndarray same shape as drive_abs
        Baseline-referenced Balloon input.
    drive0 : baseline level (A,) or scalar
    """
    d = np.asarray(drive_abs, dtype=float)
    i0, i1 = baseline_idx
    if d.ndim == 1:
        d0 = d[i0:i1].mean()
        x = gain * (d - d0)
        if clamp_nonnegative:
            x = np.maximum(0.0, x)
        return x, d0

    # d is (T, A)
    if per_area:
        d0 = d[i0:i1, :].mean(axis=0, keepdims=True)  # (1, A)
        x = gain * (d - d0)
        if clamp_nonnegative:
            x = np.maximum(0.0, x)
        return x, d0.squeeze()
    else:
        d0 = d[i0:i1, :].mean()  # scalar
        x = gain * (d - d0)
        if clamp_nonnegative:
            x = np.maximum(0.0, x)
        return x, d0
FAILURE_PENALTY = 1e9
DEFAULT_FC_TARGET_CSV = Path(r'C:\Users\GlenA\Downloads\MB_NS_acc.left_subgraph_L.csv')
CHECKPOINT_FILENAME = 'cmaes_state.pkl'
CHECKPOINT_EVAL_INTERVAL = 10

#TESTING HERE
PARAMETER_SPACE = [
    ('I_background_e', 100.0, 600.0, brian2.pA, 0.5, float(PARAMS['I_background_e'] / brian2.pA)),
    ('I_background_i', 100.0, 600.0, brian2.pA, 0.5, float(PARAMS['I_background_i'] / brian2.pA)),
    ('I_background_dend', 0.0, 80.0, brian2.pA, 0.25, float(PARAMS['I_background_dend'] / brian2.pA)),
    ('mu_ee', 0.5, 6.0, None, 0.03, float(PARAMS['mu_ee'])),
    ('mu_ie', 0.5, 6.0, None, 0.03, float(PARAMS['mu_ie'])),
    ('e_grad_min', 0.1, 0.45, None, 0.005, float(PARAMS['e_grad_min'])),
]

BOLD_PARAMETER_SPACE = [
    ('balloon_kappa', 0.10, 2.00, 0.02),
    ('balloon_gamma', 0.05, 1.50, 0.015),
    ('balloon_tau', 0.30, 2.50, 0.02),
]

FITNESS_CONFIG = {
    'target_csv': str(DEFAULT_FC_TARGET_CSV) if DEFAULT_FC_TARGET_CSV.exists() else None,
    'target_column': 'mean_connectivity',
    'seed_idx': None,
    'seed_area': None,
    'seed_areas': ['24c', '32'],
    'fc_distance': 'pearson',
    'fc_trim_seconds': 2.0,
    'fc_balloon_dt': 100 * brian2.ms,
    'fc_drive_gain': 1.0,
    'fc_baseline_start_seconds': 5.0,
    'fc_baseline_end_seconds': 40.0,
    'fc_balloon_gain': 0.025,
    'fc_clamp_nonnegative': False,
    'fc_fisher_z_clip': 2.0,
    'balloon_kappa': 0.65,
    'balloon_gamma': 0.41,
    'balloon_tau': 0.98,
    'balloon_alpha': 0.32,
    'balloon_E0': 0.4,
    'balloon_V0': 0.04,
    'balloon_epsilon': 2.0,
    'balloon_TE': 0.0254,
    'balloon_theta0': 80.6,
    'balloon_r0': 25.0,
    'balloon_neural_gain': 1.0,
    'balloon_max_step': 0.05,
    'balloon_method': 'rk4',
}


def initial_parameter_vector():
    return np.concatenate(
        (
            initial_neural_parameter_vector(),
            initial_bold_parameter_vector(),
        )
    )


def sigma_vector():
    return np.concatenate((neural_sigma_vector(), bold_sigma_vector()))


def parameter_bounds():
    neural_lower, neural_upper = neural_parameter_bounds()
    bold_lower, bold_upper = bold_parameter_bounds()
    lower = list(neural_lower) + list(bold_lower)
    upper = list(neural_upper) + list(bold_upper)
    return [lower, upper]


def initial_neural_parameter_vector():
    return np.array([curr for (_, _, _, _, _, curr) in PARAMETER_SPACE], dtype=float)


def neural_sigma_vector():
    return np.array([sigma for (_, _, _, _, sigma, _) in PARAMETER_SPACE], dtype=float)


def neural_parameter_bounds():
    lower = [lo for (_, lo, _, _, _, _) in PARAMETER_SPACE]
    upper = [hi for (_, _, hi, _, _, _) in PARAMETER_SPACE]
    return [lower, upper]


def bold_parameter_bounds():
    lower = [lo for (_, lo, _, _) in BOLD_PARAMETER_SPACE]
    upper = [hi for (_, _, hi, _) in BOLD_PARAMETER_SPACE]
    return [lower, upper]


def vector_to_param_dict(param_vector):
    param_dict = {}
    for (name, _, _, unit, _, _), value in zip(PARAMETER_SPACE, param_vector):
        if unit is None:
            param_dict[name] = float(value)
        else:
            param_dict[name] = float(value) * unit
    return param_dict


def vector_to_plain_dict(param_vector):
    return {
        name: float(value)
        for (name, _, _, _, _, _), value in zip(PARAMETER_SPACE, param_vector)
    }


def plain_dict_to_parameter_vector(param_dict):
    return np.array([float(param_dict[name]) for (name, _, _, _, _, _) in PARAMETER_SPACE], dtype=float)


def initial_bold_parameter_vector(config=None):
    merged = FITNESS_CONFIG.copy()
    if config:
        merged.update(config)
    return np.array([float(merged[name]) for (name, _, _, _) in BOLD_PARAMETER_SPACE], dtype=float)


def bold_sigma_vector():
    return np.array([sigma for (_, _, _, sigma) in BOLD_PARAMETER_SPACE], dtype=float)


def vector_to_bold_config(param_vector):
    return {
        name: float(value)
        for (name, _, _, _), value in zip(BOLD_PARAMETER_SPACE, param_vector)
    }


def load_result_summary(summary_path):
    summary_path = Path(summary_path).expanduser().resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Result summary not found: {summary_path}")

    with summary_path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)

    neural_params = payload.get('best_params')
    if not isinstance(neural_params, dict):
        raise ValueError("Result summary is missing a valid 'best_params' object.")

    bold_params = payload.get('best_bold_params')
    if bold_params is not None and not isinstance(bold_params, dict):
        raise ValueError("Result summary has an invalid 'best_bold_params' object.")

    missing_neural = [
        name for (name, _, _, _, _, _) in PARAMETER_SPACE
        if name not in neural_params
    ]
    if missing_neural:
        raise ValueError(
            "Result summary is missing neural parameters: "
            + ", ".join(missing_neural)
        )

    if bold_params is None:
        bold_params = {}
    missing_bold = [
        name for (name, _, _, _) in BOLD_PARAMETER_SPACE
        if name not in bold_params
    ]
    if missing_bold:
        raise ValueError(
            "Result summary is missing BOLD parameters: "
            + ", ".join(missing_bold)
        )

    return {
        'path': str(summary_path),
        'neural_vector': plain_dict_to_parameter_vector(neural_params),
        'bold_vector': np.array([float(bold_params[name]) for (name, _, _, _) in BOLD_PARAMETER_SPACE], dtype=float),
        'joint_vector': np.concatenate(
            (
                plain_dict_to_parameter_vector(neural_params),
                np.array([float(bold_params[name]) for (name, _, _, _) in BOLD_PARAMETER_SPACE], dtype=float),
            )
        ),
        'neural_params': {
            name: float(neural_params[name]) for (name, _, _, _, _, _) in PARAMETER_SPACE
        },
        'bold_params': {
            name: float(bold_params[name]) for (name, _, _, _) in BOLD_PARAMETER_SPACE
        },
    }


def split_joint_parameter_vector(param_vector):
    vector = np.asarray(param_vector, dtype=float)
    neural_dim = len(PARAMETER_SPACE)
    bold_dim = len(BOLD_PARAMETER_SPACE)
    expected_dim = neural_dim + bold_dim
    if vector.size != expected_dim:
        raise ValueError(
            f"Expected joint parameter vector of length {expected_dim}, got {vector.size}."
        )
    return vector[:neural_dim], vector[neural_dim:neural_dim + bold_dim]


def joint_vector_to_plain_dicts(param_vector):
    neural_vector, bold_vector = split_joint_parameter_vector(param_vector)
    return vector_to_plain_dict(neural_vector), vector_to_bold_config(bold_vector)


def save_default_params(path):
    with path.open('wb') as handle:
        pickle.dump(PARAMS, handle)


def checkpoint_path_for_log_dir(log_dir):
    return Path(log_dir) / CHECKPOINT_FILENAME


def save_cmaes_state(
    checkpoint_path,
    *,
    es,
    generation,
    fitness_config,
    log_dir,
    workers=None,
    start_from_summary=None,
    resumed_from_checkpoint=None,
):
    checkpoint_path = Path(checkpoint_path)
    payload = {
        'version': 1,
        'es': es,
        'generation': int(generation),
        'countevals': int(es.countevals),
        'fitness_config': dict(fitness_config or {}),
        'log_dir': str(Path(log_dir).resolve()),
        'workers': None if workers is None else int(workers),
        'start_from_summary': None if start_from_summary is None else str(start_from_summary),
        'resumed_from_checkpoint': (
            None if resumed_from_checkpoint is None else str(resumed_from_checkpoint)
        ),
    }
    with checkpoint_path.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return checkpoint_path


def load_cmaes_state(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CMA-ES checkpoint not found: {checkpoint_path}")

    with checkpoint_path.open('rb') as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("CMA-ES checkpoint is not a valid payload.")

    es = payload.get('es')
    if es is None or not hasattr(es, 'N') or not hasattr(es, 'countevals'):
        raise ValueError("CMA-ES checkpoint is missing a valid strategy object.")

    expected_dim = len(initial_parameter_vector())
    if int(es.N) != expected_dim:
        raise ValueError(
            f"CMA-ES checkpoint dimension {int(es.N)} does not match current model dimension {expected_dim}."
        )

    fitness_config = payload.get('fitness_config') or {}
    if not isinstance(fitness_config, dict):
        raise ValueError("CMA-ES checkpoint contains an invalid fitness_config payload.")

    return {
        'path': str(checkpoint_path),
        'es': es,
        'generation': int(payload.get('generation', 0)),
        'countevals': int(es.countevals),
        'fitness_config': dict(fitness_config),
        'log_dir': str(payload.get('log_dir')) if payload.get('log_dir') is not None else None,
        'workers': payload.get('workers'),
        'start_from_summary': payload.get('start_from_summary'),
        'resumed_from_checkpoint': payload.get('resumed_from_checkpoint'),
    }


def _mean_rate_in_window(rate, t, t_start, t_end):
    mask = (t >= t_start) & (t < t_end)
    if not np.any(mask):
        return 0.0
    return float(np.mean(rate[mask]))


def _to_seconds(value):
    return float(value / brian2.second) if hasattr(value, 'unit') else float(value)


def _derive_simulation_seed(master_seed, *components):
    if master_seed is None:
        return None

    spawn_key = [int(master_seed)]
    spawn_key.extend(int(component) for component in components)
    seed_seq = np.random.SeedSequence(spawn_key)
    return int(seed_seq.generate_state(1, dtype=np.uint32)[0])


def _resolve_model_seed_indices(area_names, *, seed_areas=None, seed_area=None, seed_indices=None, seed_idx=None):
    if seed_areas is not None:
        resolved_seed_areas = list(seed_areas)
        missing = [name for name in resolved_seed_areas if name not in area_names]
        if missing:
            raise ValueError(f"seed_areas not found in model areas: {', '.join(missing)}")
        indices = [int(area_names.index(name)) for name in resolved_seed_areas]
        return indices, resolved_seed_areas

    if seed_area is not None:
        if seed_area not in area_names:
            raise ValueError(f"seed_area '{seed_area}' not found in model areas.")
        return [int(area_names.index(seed_area))], [str(seed_area)]

    if seed_indices is not None:
        indices = [int(value) for value in seed_indices]
        for index in indices:
            if index < 0 or index >= len(area_names):
                raise ValueError(f"seed index {index} is out of bounds for {len(area_names)} areas.")
        return indices, [str(area_names[index]) for index in indices]

    if seed_idx is None:
        raise ValueError("Provide seed_area, seed_areas, seed_idx, or seed_indices for FC fitness.")

    seed_idx = int(seed_idx)
    if seed_idx < 0 or seed_idx >= len(area_names):
        raise ValueError(f"seed_idx {seed_idx} is out of bounds for {len(area_names)} areas.")
    return [seed_idx], [str(area_names[seed_idx])]


def _clip_window_to_indices(n_samples, dt_s, start_s, end_s):
    i0 = max(0, int(round(start_s / dt_s)))
    i1 = min(n_samples, int(round(end_s / dt_s)))
    if i1 <= i0:
        if n_samples < 2:
            raise ValueError("Not enough samples to construct a baseline window.")
        return 0, n_samples
    return i0, i1


def _balloon_kwargs_from_config(config):
    return {
        'kappa': float(config['balloon_kappa']),
        'gamma': float(config['balloon_gamma']),
        'tau': float(config['balloon_tau']),
        'alpha': float(config['balloon_alpha']),
        'E0': float(config['balloon_E0']),
        'V0': float(config['balloon_V0']),
        'TE': float(config['balloon_TE']),
        'epsilon': float(config['balloon_epsilon']),
        'theta0': float(config['balloon_theta0']),
        'r0': float(config['balloon_r0']),
        'neural_gain': float(config['balloon_neural_gain']),
        'max_step': float(config['balloon_max_step']),
    }


def _correlation_to_clipped_fisher_z(values, clip_value=2.0):
    arr = np.asarray(values, dtype=float)
    z = np.full(arr.shape, np.nan, dtype=float)
    finite = np.isfinite(arr)
    if np.any(finite):
        clipped_r = np.clip(arr[finite], -1.0 + 1e-6, 1.0 - 1e-6)
        z[finite] = np.arctanh(clipped_r)
        if clip_value is not None:
            z[finite] = np.clip(z[finite], -float(clip_value), float(clip_value))
    return z


def load_fc_target(csv_path, *, value_column='mean_connectivity'):
    target_path = Path(csv_path).expanduser().resolve()
    cache_key = (str(target_path), str(value_column))
    cached = _FC_TARGET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not target_path.exists():
        raise FileNotFoundError(f"Target FC CSV not found: {target_path}")

    df = pandas.read_csv(target_path)
    required_columns = {'area_name', value_column}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            f"Target FC CSV is missing required columns: {', '.join(missing_columns)}"
        )

    area_names = df['area_name'].astype(str).tolist()
    if len(area_names) != len(set(area_names)):
        raise ValueError("Target FC CSV contains duplicate area_name values.")

    target_values = pandas.to_numeric(df[value_column], errors='coerce').to_numpy(dtype=float)
    target_spec = {
        'path': str(target_path),
        'area_names': area_names,
        'values': target_values,
        'value_column': value_column,
        'seed_label': str(df['seed'].iloc[0]) if 'seed' in df.columns and not df.empty else None,
        'condition': str(df['condition'].iloc[0]) if 'condition' in df.columns and not df.empty else None,
    }
    _FC_TARGET_CACHE[cache_key] = target_spec
    return target_spec


def align_target_fc(target_spec, area_names):
    value_by_area = {
        area_name: float(value)
        for area_name, value in zip(target_spec['area_names'], target_spec['values'])
    }
    missing = [area_name for area_name in area_names if area_name not in value_by_area]
    if missing:
        raise ValueError(
            "Target FC CSV does not contain all model areas. "
            f"Missing: {', '.join(missing)}"
        )

    return np.array([value_by_area[area_name] for area_name in area_names], dtype=float)


def _pearson_from_valid_fc(model_fc_z, target_fc_z, valid_mask):
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid_mask):
        return np.nan

    model_fc_z = np.asarray(model_fc_z, dtype=float)[valid_mask]
    target_fc_z = np.asarray(target_fc_z, dtype=float)[valid_mask]
    if model_fc_z.size < 2:
        return np.nan

    pearson_r = np.corrcoef(model_fc_z, target_fc_z)[0, 1]
    return float(pearson_r) if np.isfinite(pearson_r) else np.nan


def compute_seed_fc_from_drive(area_drive_abs, params_run, area_names, *, fitness_config):
    config = FITNESS_CONFIG.copy()
    config.update(fitness_config or {})

    seed_indices, seed_area_names = _resolve_model_seed_indices(
        area_names,
        seed_areas=config.get('seed_areas'),
        seed_area=config.get('seed_area'),
        seed_idx=config.get('seed_idx'),
    )

    dt_old = params_run['dt']
    dt_old_s = _to_seconds(dt_old)
    trim_steps = max(0, int(round(float(config['fc_trim_seconds']) / dt_old_s)))

    if trim_steps >= area_drive_abs.shape[0] - 1:
        raise ValueError("fc_trim_seconds removes the full simulation.")

    drive = np.asarray(area_drive_abs[trim_steps:, :] / brian2.pA, dtype=float) * float(config['fc_drive_gain'])
    drive_ds, _, dt_balloon_s = downsample_to_dt(drive, dt_old, config['fc_balloon_dt'])

    i0, i1 = _clip_window_to_indices(
        drive_ds.shape[0],
        dt_balloon_s,
        float(config['fc_baseline_start_seconds']),
        float(config['fc_baseline_end_seconds']),
    )

    balloon_input, drive0 = drive_abs_to_balloon_input(
        drive_ds,
        baseline_idx=(i0, i1),
        gain=float(config['fc_balloon_gain']),
        clamp_nonnegative=bool(config['fc_clamp_nonnegative']),
    )

    bold, _ = balloon_bold_per_area(
        balloon_input,
        dt=dt_balloon_s,
        method=str(config.get('balloon_method', 'rk4')),
        **_balloon_kwargs_from_config(config),
    )
    fc_vector = seed_based_fc(bold, seed_indices if len(seed_indices) > 1 else seed_indices[0])

    return {
        'seed_idx': seed_indices[0] if len(seed_indices) == 1 else None,
        'seed_area': seed_area_names[0] if len(seed_area_names) == 1 else '+'.join(seed_area_names),
        'seed_indices': seed_indices,
        'seed_areas': seed_area_names,
        'seed_fc': fc_vector,
        'dt_balloon_s': dt_balloon_s,
        'drive_baseline': drive0,
    }


def run_simulation(param_vector, override_params=None, *, simulation_seed=None):
    neural_vector, _ = split_joint_parameter_vector(param_vector)
    params_run = PARAMS.copy()
    params_run.update(vector_to_param_dict(neural_vector))
    if override_params:
        params_run.update(override_params)

    try:
        (sln, fln, hierarchy, area_list_SLN,
         df_fln, df_sln, d1_density_raw, spine_count_raw, df_raw_anatomy) = load_anatomy()

        (pops, num_pops, num_e_pops, num_areas, e_grad, g_adapt, ampa_frac, nmda_frac,
         J_nmda, J_ampa, J_gaba, J_gaba_dend, W_superficial, W_deep, lr_targets,
         nmda_da_grad, e_pv_da_mat, e_sst_da_mat, m_da_grad, g_m, lr_targets_FEF) = prepare_connectivity(
            params_run, spine_count_raw, fln, sln, d1_density_raw
        )

        rng = np.random.default_rng(simulation_seed)
        state = initialise_simulation_state(
            params_run, num_areas, num_pops, area_list_SLN, pops, rng=rng
        )

        area_drive_abs = large_scale_da_model(
            pops, num_pops, num_areas, e_grad, g_adapt, g_m, ampa_frac, nmda_frac,
            J_nmda, J_ampa, J_gaba, J_gaba_dend, W_superficial, W_deep, lr_targets,
            nmda_da_grad, e_pv_da_mat, e_sst_da_mat, m_da_grad, state, params_run,
            lr_targets_FEF, area_list_SLN
        )

        return {
            'params': params_run,
            'area_names': area_list_SLN,
            'area_drive_abs': area_drive_abs,
        }
    except Exception as e:
        import traceback
        print("run_simulation failed:", repr(e))
        traceback.print_exc()
        raise


def summarise_simulation(simulation, fitness_config=None):
    if simulation is None:
        return None

    config = FITNESS_CONFIG.copy()
    if fitness_config:
        config.update(fitness_config)

    target_csv = config.get('target_csv')
    if not target_csv:
        raise ValueError("target_csv is required because the deprecated rate-based fitness path has been removed.")

    target_spec = load_fc_target(target_csv, value_column=config['target_column'])
    target_fc = align_target_fc(target_spec, simulation['area_names'])
    model_fc = compute_seed_fc_from_drive(
        simulation['area_drive_abs'],
        simulation['params'],
        simulation['area_names'],
        fitness_config=config,
    )
    seed_indices = [int(value) for value in model_fc['seed_indices']]
    exclude_mask = np.zeros(len(simulation['area_names']), dtype=bool)
    exclude_mask[seed_indices] = True

    model_fc_r = np.asarray(model_fc['seed_fc'], dtype=float)
    model_fc_z = _correlation_to_clipped_fisher_z(
        model_fc_r,
        clip_value=config.get('fc_fisher_z_clip', 2.0),
    )
    target_fc_z = np.asarray(target_fc, dtype=float)
    valid_mask = np.isfinite(model_fc_z) & np.isfinite(target_fc_z) & (~exclude_mask)
    if not np.any(valid_mask):
        return None

    pearson_r = _pearson_from_valid_fc(model_fc_z, target_fc_z, valid_mask)
    pearson_fitness = 1.0 - pearson_r if np.isfinite(pearson_r) else FAILURE_PENALTY

    return {
        'target_fc': target_fc_z,
        'seed_fc': model_fc_z,
        'seed_fc_r': model_fc_r,
        'seed_idx': model_fc['seed_idx'],
        'seed_area': model_fc['seed_area'],
        'seed_indices': model_fc['seed_indices'],
        'seed_areas': model_fc['seed_areas'],
        'fc_valid_count': int(np.count_nonzero(valid_mask)),
        'valid_mask': valid_mask,
        'excluded_seed_count': int(np.count_nonzero(exclude_mask)),
        'fc_target_seed': target_spec['seed_label'],
        'fc_target_condition': target_spec['condition'],
        'fc_target_column': target_spec['value_column'],
        'fc_target_path': target_spec['path'],
        'dt_balloon_s': model_fc['dt_balloon_s'],
        'pearson_r': float(pearson_r) if np.isfinite(pearson_r) else None,
        'pearson_fitness': float(pearson_fitness) if np.isfinite(pearson_fitness) else float(FAILURE_PENALTY),
    }



def fitness_from_summary(summary, fitness_config=None):
    if summary is None:
        return FAILURE_PENALTY

    config = FITNESS_CONFIG.copy()
    if fitness_config:
        config.update(fitness_config)

    valid_mask = np.asarray(summary.get('valid_mask'), dtype=bool)
    if not np.any(valid_mask):
        return FAILURE_PENALTY

    model_fc_z = np.asarray(summary['seed_fc'], dtype=float)[valid_mask]
    target_fc_z = np.asarray(summary['target_fc'], dtype=float)[valid_mask]
    diff = model_fc_z - target_fc_z
    metric = str(config.get('fc_distance', 'mse')).lower()
    if metric == 'pearson':
        pearson_r = summary.get('pearson_r')
        if pearson_r is None:
            pearson_r = _pearson_from_valid_fc(summary['seed_fc'], summary['target_fc'], valid_mask)
        if not np.isfinite(pearson_r):
            return FAILURE_PENALTY
        fitness = 1.0 - pearson_r
    elif metric == 'mse':
        fitness = np.mean(diff ** 2)
    elif metric == 'rmse':
        fitness = np.sqrt(np.mean(diff ** 2))
    elif metric == 'mae':
        fitness = np.mean(np.abs(diff))
    elif metric == 'l2':
        fitness = np.linalg.norm(diff)
    else:
        raise ValueError(f"Unsupported fc_distance metric: {config['fc_distance']}")
    return float(fitness) if np.isfinite(fitness) else FAILURE_PENALTY


def evaluate_fitness(param_vector, fitness_config=None, simulation_seed=None):
    _, bold_vector = split_joint_parameter_vector(param_vector)
    config = FITNESS_CONFIG.copy()
    config.update(fitness_config or {})
    config.update(vector_to_bold_config(bold_vector))

    simulation = run_simulation(param_vector, simulation_seed=simulation_seed)
    if simulation is None:
        return FAILURE_PENALTY

    summary = summarise_simulation(simulation, fitness_config=config)
    if summary is None:
        return FAILURE_PENALTY

    return fitness_from_summary(summary, fitness_config=config)


def evaluate_candidate_record(param_vector, fitness_config=None, simulation_seed=None):
    vector = np.asarray(param_vector, dtype=float)
    _, bold_vector = split_joint_parameter_vector(vector)
    config = FITNESS_CONFIG.copy()
    config.update(fitness_config or {})
    config.update(vector_to_bold_config(bold_vector))

    simulation = run_simulation(vector, simulation_seed=simulation_seed)
    if simulation is None:
        return {
            'vector': vector,
            'fitness': FAILURE_PENALTY,
            'simulation': None,
            'summary': None,
        }

    summary = summarise_simulation(simulation, fitness_config=config)
    fitness = fitness_from_summary(summary, fitness_config=config)
    return {
        'vector': vector,
        'fitness': float(fitness),
        'simulation': simulation,
        'summary': summary,
    }


EVALUATE_CANDIDATE_RECORD_JOB = wrap_non_picklable_objects(evaluate_candidate_record)


def parallel_candidate_records(population, workers=None, fitness_config=None):
    if workers is None:
        workers = min(mp.cpu_count(), len(population))
    config = FITNESS_CONFIG.copy()
    config.update(fitness_config or {})
    master_seed = config.get('simulation_seed')
    simulation_seeds = [master_seed for _ in range(len(population))]
    records = Parallel(n_jobs=workers, backend='loky')(
        delayed(EVALUATE_CANDIDATE_RECORD_JOB)(
            candidate,
            fitness_config=fitness_config,
            simulation_seed=simulation_seeds[idx],
        )
        for idx, candidate in enumerate(population)
    )
    for idx, record in enumerate(records):
        record['simulation_seed'] = simulation_seeds[idx]
    return records


def append_generation_log(log_path, generation, population, fitness_values):
    best_index = int(np.argmin(fitness_values))
    best_vector = np.asarray(population[best_index], dtype=float)
    best_neural_params, best_bold_params = joint_vector_to_plain_dicts(best_vector)
    record = {
        'generation': int(generation),
        'best_fitness': float(fitness_values[best_index]),
        'mean_fitness': float(np.mean(fitness_values)),
        'best_vector': best_vector.tolist(),
        'best_params': best_neural_params,
        'best_bold_params': best_bold_params,
    }
    with log_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record) + '\n')


def export_best_fc_csvs(log_dir, best_vector, fitness_config=None, *, simulation=None, summary=None):
    config = FITNESS_CONFIG.copy()
    config.update(fitness_config or {})
    _, bold_vector = split_joint_parameter_vector(best_vector)
    config.update(vector_to_bold_config(bold_vector))
    if simulation is None:
        simulation = run_simulation(
            best_vector,
            simulation_seed=config.get('simulation_seed'),
        )
    if summary is None and simulation is not None:
        summary = summarise_simulation(simulation, fitness_config=config)
    if summary is None or 'seed_fc' not in summary or 'target_fc' not in summary:
        return None

    area_names = simulation['area_names']
    seed_area = summary['seed_area']
    seed_indices = [int(value) for value in summary.get('seed_indices', [])]
    model_fc_z = np.asarray(summary['seed_fc'], dtype=float)
    model_fc_r = np.asarray(summary.get('seed_fc_r'), dtype=float)
    target_fc = np.asarray(summary['target_fc'], dtype=float)
    valid_mask = np.asarray(summary.get('valid_mask'), dtype=bool)
    diff = model_fc_z - target_fc
    squared_error = diff ** 2

    common_columns = {
        'area_name': area_names,
        'seed_area': [seed_area] * len(area_names),
        'seed_indices': [','.join(str(value) for value in seed_indices)] * len(area_names),
    }

    seed_fc_path = log_dir / 'seed_fc.csv'
    target_fc_path = log_dir / 'target_fc.csv'
    comparison_path = log_dir / 'seed_vs_target_fc.csv'

    pandas.DataFrame({
        **common_columns,
        'model_seed_fc_r': model_fc_r,
        'model_seed_fc_z': model_fc_z,
    }).to_csv(seed_fc_path, index=False)

    pandas.DataFrame({
        **common_columns,
        'target_seed_fc_z': target_fc,
    }).to_csv(target_fc_path, index=False)

    pandas.DataFrame({
        **common_columns,
        'model_seed_fc_r': model_fc_r,
        'model_seed_fc_z': model_fc_z,
        'target_seed_fc_z': target_fc,
        'difference': diff,
        'squared_error': squared_error,
        'valid': valid_mask,
    }).to_csv(comparison_path, index=False)

    return {
        'seed_fc_csv': str(seed_fc_path),
        'target_fc_csv': str(target_fc_path),
        'seed_vs_target_fc_csv': str(comparison_path),
    }


def write_final_summary(
    log_dir,
    result,
    stop_reasons,
    *,
    generation,
    best_fitness_override=None,
    best_pearson_r=None,
    best_pearson_fitness=None,
    reevaluated_fitness=None,
    reevaluated_pearson_r=None,
    reevaluated_pearson_fitness=None,
    final_evaluation_seed=None,
    initialised_from_summary=None,
    initialised_from_checkpoint=None,
    exported_files=None,
):
    best_vector = np.asarray(result.xbest, dtype=float)
    best_params, best_bold_params = joint_vector_to_plain_dicts(best_vector)
    summary = {
        'best_fitness': float(result.fbest) if best_fitness_override is None else float(best_fitness_override),
        'best_vector': best_vector.tolist(),
        'best_params': best_params,
        'best_bold_params': best_bold_params,
        'stop_reasons': stop_reasons,
        'generations_completed': int(generation),
        'best_pearson_r': None if best_pearson_r is None else float(best_pearson_r),
        'best_pearson_fitness': None if best_pearson_fitness is None else float(best_pearson_fitness),
        'reevaluated_fitness': None if reevaluated_fitness is None else float(reevaluated_fitness),
        'reevaluated_pearson_r': None if reevaluated_pearson_r is None else float(reevaluated_pearson_r),
        'reevaluated_pearson_fitness': (
            None if reevaluated_pearson_fitness is None else float(reevaluated_pearson_fitness)
        ),
        'final_evaluation_seed': None if final_evaluation_seed is None else int(final_evaluation_seed),
        'initialised_from_summary': (
            str(initialised_from_summary) if initialised_from_summary is not None else None
        ),
        'initialised_from_checkpoint': (
            str(initialised_from_checkpoint) if initialised_from_checkpoint is not None else None
        ),
        'exported_files': exported_files or {},
    }
    with (log_dir / 'result_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)


def make_seeded_run_log_dir(log_root, run_number, seed):
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'run_{int(run_number):03d}_seed_{int(seed)}_{timestamp}'
    log_dir = log_root / base_name
    suffix = 1
    while log_dir.exists():
        log_dir = log_root / f'{base_name}_{suffix:02d}'
        suffix += 1
    return log_dir


def run_cmaes(
    *,
    sigma0=None,
    popsize=None,
    maxfevals=None,
    workers=None,
    ftarget=None,
    seed=None,
    log_dir='cmaes_logs',
    fitness_config=None,
    start_from_summary=None,
    resume_state=None,
):
    if cma is None:
        raise ImportError("The 'cma' package is required to run optimisation.")
    requested_sigma0 = sigma0
    requested_popsize = popsize
    requested_maxfevals = maxfevals
    requested_ftarget = ftarget
    default_sigma0 = 20.0
    default_popsize = 12
    default_maxfevals = 2000

    start_summary = None
    resume_payload = None
    if resume_state is not None:
        if start_from_summary is not None:
            raise ValueError("Use either resume_state or start_from_summary, not both.")
        if fitness_config:
            raise ValueError("Cannot override fitness_config when resuming from a saved CMA-ES state.")

        resume_payload = load_cmaes_state(resume_state)
        log_dir_value = resume_payload.get('log_dir') or log_dir
        log_dir = Path(log_dir_value)
        generation_log = log_dir / 'generations.jsonl'
        effective_fitness_config = dict(resume_payload['fitness_config'])
        workers = resume_payload.get('workers') if workers is None else int(workers)
        es = resume_payload['es']
        generation = int(resume_payload['generation'])
        if requested_maxfevals is not None:
            es.opts['maxfevals'] = int(requested_maxfevals)
            if hasattr(es, '_stopdict'):
                es._stopdict.clear()
        if requested_ftarget is not None:
            es.opts['ftarget'] = float(requested_ftarget)
            if hasattr(es, '_stopdict'):
                es._stopdict.clear()
        initialised_from_checkpoint = resume_payload['path']
    else:
        sigma0 = default_sigma0 if requested_sigma0 is None else float(requested_sigma0)
        popsize = default_popsize if requested_popsize is None else int(requested_popsize)
        maxfevals = default_maxfevals if requested_maxfevals is None else int(requested_maxfevals)
        log_dir = Path(log_dir)
        generation_log = log_dir / 'generations.jsonl'
        if start_from_summary is not None:
            start_summary = load_result_summary(start_from_summary)

        effective_fitness_config = FITNESS_CONFIG.copy()
        effective_fitness_config.update(fitness_config or {})
        if seed is None:
            seed = DEFAULT_GLOBAL_SEED
        if effective_fitness_config.get('simulation_seed') is None:
            effective_fitness_config['simulation_seed'] = int(seed)

        x0 = initial_parameter_vector() if start_summary is None else start_summary['joint_vector']
        cma_options = {
            'bounds': parameter_bounds(),
            'popsize': popsize,
            'maxfevals': maxfevals,
            'verb_log': 1,
            'verb_disp': 1,
            'CMA_stds': sigma_vector(),
            'verb_filenameprefix': str(log_dir / 'outcmaes'),
            'seed': int(seed),
        }
        if requested_ftarget is not None:
            cma_options['ftarget'] = float(requested_ftarget)

        es = cma.CMAEvolutionStrategy(x0, sigma0, inopts=cma_options)
        generation = 0
        initialised_from_checkpoint = None

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_path_for_log_dir(log_dir)
    save_default_params(log_dir / 'default_params.pck')
    if resume_payload is None and generation_log.exists():
        generation_log.unlink()
    last_checkpoint_eval = (
        int(resume_payload['countevals']) if resume_payload is not None else 0
    )
    best_observed_record = None
    best_observed_fitness = float(es.result.fbest) if np.isfinite(getattr(es.result, 'fbest', np.nan)) else np.inf

    while not es.stop():
        population = es.ask()
        generation_fitness_config = dict(effective_fitness_config)
        generation_fitness_config['generation'] = generation
        candidate_records = parallel_candidate_records(
            population,
            workers=workers,
            fitness_config=generation_fitness_config,
        )
        fitness_values = [float(record['fitness']) for record in candidate_records]
        es.tell(population, fitness_values)
        es.logger.add()
        es.disp()

        best_index = int(np.argmin(fitness_values))
        best_vector = np.asarray(population[best_index], dtype=float)
        best_fitness = float(fitness_values[best_index])
        if best_fitness < best_observed_fitness:
            best_observed_fitness = best_fitness
            best_observed_record = candidate_records[best_index]

        np.save(log_dir / 'best_params_so_far.npy', best_vector)
        np.save(log_dir / 'best_fitness_so_far.npy', np.array(best_fitness))
        append_generation_log(generation_log, generation, population, fitness_values)
        generation += 1
        if es.countevals - last_checkpoint_eval >= CHECKPOINT_EVAL_INTERVAL:
            save_cmaes_state(
                checkpoint_path,
                es=es,
                generation=generation,
                fitness_config=effective_fitness_config,
                log_dir=log_dir,
                workers=workers,
                start_from_summary=None if start_summary is None else start_summary['path'],
                resumed_from_checkpoint=initialised_from_checkpoint,
            )
            last_checkpoint_eval = int(es.countevals)

    best_vector = np.asarray(es.result.xbest, dtype=float)
    final_evaluation_seed = effective_fitness_config.get('simulation_seed')
    final_record = evaluate_candidate_record(
        best_vector,
        fitness_config=effective_fitness_config,
        simulation_seed=final_evaluation_seed,
    )
    reevaluated_fitness = float(final_record['fitness'])
    final_simulation = final_record.get('simulation')
    final_summary = final_record.get('summary')
    reevaluated_pearson_r = None if final_summary is None else final_summary.get('pearson_r')
    reevaluated_pearson_fitness = None if final_summary is None else final_summary.get('pearson_fitness')
    observed_summary = None if best_observed_record is None else best_observed_record.get('summary')
    best_pearson_r = None if observed_summary is None else observed_summary.get('pearson_r')
    best_pearson_fitness = None if observed_summary is None else observed_summary.get('pearson_fitness')
    best_fitness = float(es.result.fbest)
    best_neural_params, best_bold_params = joint_vector_to_plain_dicts(best_vector)
    np.save(log_dir / 'best_params_final.npy', best_vector)
    np.save(log_dir / 'best_fitness_final.npy', np.array(best_fitness))
    with (log_dir / 'best_bold_params_final.json').open('w', encoding='utf-8') as handle:
        json.dump(best_bold_params, handle, indent=2)
    np.save(log_dir / 'best_neural_params_final.npy', split_joint_parameter_vector(best_vector)[0])

    exported_files = {
        'best_bold_params_json': str(log_dir / 'best_bold_params_final.json'),
        'best_neural_params_npy': str(log_dir / 'best_neural_params_final.npy'),
    }
    final_checkpoint = save_cmaes_state(
        checkpoint_path,
        es=es,
        generation=generation,
        fitness_config=effective_fitness_config,
        log_dir=log_dir,
        workers=workers,
        start_from_summary=None if start_summary is None else start_summary['path'],
        resumed_from_checkpoint=initialised_from_checkpoint,
    )
    exported_files['cmaes_state_pickle'] = str(final_checkpoint)
    try:
        exported = export_best_fc_csvs(
            log_dir,
            best_vector,
            fitness_config=effective_fitness_config,
            simulation=final_simulation,
            summary=final_summary,
        )
        if exported is not None:
            exported_files.update(exported)
    except Exception as exc:
        exported_files['fc_export_error'] = repr(exc)

    if final_simulation is not None:
        np.save(
            log_dir / 'best_area_drive_abs_pA.npy',
            np.asarray(final_simulation['area_drive_abs'] / brian2.pA, dtype=float),
        )
        exported_files['best_area_drive_abs_pA_npy'] = str(log_dir / 'best_area_drive_abs_pA.npy')

    write_final_summary(
        log_dir,
        es.result,
        es.stop(),
        generation=generation,
        best_fitness_override=best_fitness,
        best_pearson_r=best_pearson_r,
        best_pearson_fitness=best_pearson_fitness,
        reevaluated_fitness=reevaluated_fitness,
        reevaluated_pearson_r=reevaluated_pearson_r,
        reevaluated_pearson_fitness=reevaluated_pearson_fitness,
        final_evaluation_seed=final_evaluation_seed,
        initialised_from_summary=None if start_summary is None else start_summary['path'],
        initialised_from_checkpoint=initialised_from_checkpoint,
        exported_files=exported_files,
    )

    print("Stop reasons:", es.stop())
    print("Best fitness:", best_fitness)
    if best_pearson_r is not None:
        print("Best Pearson r:", float(best_pearson_r))
        print("Best Pearson fitness (1-r):", float(best_pearson_fitness))
    print("Reevaluated fitness:", reevaluated_fitness)
    if reevaluated_pearson_r is not None:
        print("Reevaluated Pearson r:", float(reevaluated_pearson_r))
        print("Reevaluated Pearson fitness (1-r):", float(reevaluated_pearson_fitness))
    print("Best neural parameters:", best_neural_params)
    print("Best BOLD parameters:", best_bold_params)
    return best_vector


def parse_args():
    parser = argparse.ArgumentParser(description='Run CMA-ES for the OPT baseline model.')
    # Command-line usage:
    #   Run the default seeded CMA-ES batch:
    #     py -3 "Baseline NOT RANDOM SEED.py"
    #   Run a custom number of fresh seeded runs:
    #     py -3 "Baseline NOT RANDOM SEED.py" --num-runs 5
    parser.add_argument('--sigma0', type=float, default=None)
    parser.add_argument('--popsize', type=int, default=None)
    parser.add_argument('--maxfevals', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--ftarget', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--log-dir', default='cmaes_logs')
    parser.add_argument('--start-from-summary', default=None)
    parser.add_argument('--resume-state', default=None)
    parser.add_argument('--target-csv', default=None)
    parser.add_argument('--target-column', default=None)
    parser.add_argument('--seed-area', default=None)
    parser.add_argument('--seed-idx', type=int, default=None)
    parser.add_argument('--fc-distance', choices=('pearson', 'mse', 'rmse', 'mae', 'l2'), default=None)
    parser.add_argument('--simulation-seed', type=int, default=None)
    parser.add_argument('--num-runs', type=int, default=DEFAULT_BATCH_RUNS)
    parser.add_argument('--seed-log-root', default=str(DEFAULT_SEEDED_LOG_ROOT))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_runs < 1:
        raise ValueError('--num-runs must be at least 1.')
    if args.log_dir != 'cmaes_logs':
        raise ValueError('Seeded batch mode writes logs under "SEEDED LOGS"; do not pass --log-dir.')
    if args.resume_state is not None:
        raise ValueError('Seeded batch mode does not support --resume-state; each run must start fresh.')
    if args.start_from_summary is not None:
        raise ValueError('Seeded batch mode does not support --start-from-summary; each run must start fresh.')
    if args.seed is not None:
        raise ValueError('Seeded batch mode sets the CMA-ES seed automatically; do not pass --seed.')
    if args.simulation_seed is not None:
        raise ValueError('Seeded batch mode sets the simulation seed automatically; do not pass --simulation-seed.')

    fitness_config = {}
    if args.target_csv is not None:
        fitness_config['target_csv'] = args.target_csv
    if args.target_column is not None:
        fitness_config['target_column'] = args.target_column
    if args.seed_area is not None:
        fitness_config['seed_area'] = args.seed_area
    if args.seed_idx is not None:
        fitness_config['seed_idx'] = args.seed_idx
    if args.fc_distance is not None:
        fitness_config['fc_distance'] = args.fc_distance

    run_summaries = []
    for run_number in range(1, args.num_runs + 1):
        seed = 12345 * run_number
        run_log_dir = make_seeded_run_log_dir(args.seed_log_root, run_number, seed)
        run_fitness_config = dict(fitness_config)
        run_fitness_config['simulation_seed'] = seed
        print(f'Starting CMA-ES run {run_number}/{args.num_runs} with seed {seed}')
        run_cmaes(
            sigma0=args.sigma0,
            popsize=args.popsize,
            maxfevals=args.maxfevals,
            workers=args.workers,
            ftarget=args.ftarget,
            seed=seed,
            log_dir=run_log_dir,
            fitness_config=run_fitness_config or None,
            start_from_summary=None,
            resume_state=None,
        )
        run_summaries.append({
            'run_number': run_number,
            'seed': seed,
            'log_dir': str(run_log_dir),
        })

    return run_summaries


if __name__ == '__main__':
    # Command-line usage from PowerShell / cmd:
    #   py -3 "D:\New folder\serotonin\Baseline NOT RANDOM SEED.py"
    #   py -3 "D:\New folder\serotonin\Baseline NOT RANDOM SEED.py" --num-runs 5
    #   py -3 "D:\New folder\serotonin\Baseline NOT RANDOM SEED.py" --popsize 24 --maxfevals 1200 --workers 8
    #
    # Optional CLI flags:
    #   --num-runs INT             Number of fresh CMA-ES runs to execute. Seed for run N is 12345 * N.
    #   --seed-log-root PATH       Parent directory for seeded run logs. Defaults to "SEEDED LOGS".
    #   --sigma0 FLOAT              Global CMA-ES step size used when creating a fresh run.
    #   --popsize INT               CMA-ES population size.
    #   --maxfevals INT             Maximum total fitness evaluations.
    #   --workers INT               Parallel worker count for candidate evaluation.
    #   --ftarget FLOAT             Stop early if fitness reaches this target.
    #   --target-csv PATH           CSV file containing the empirical FC target.
    #   --target-column NAME        Column name to read from --target-csv.
    #   --seed-area NAME            Single model area to use as the FC seed.
    #   --seed-idx INT              Single model area index to use as the FC seed.
    #   --fc-distance {pearson,mse,rmse,mae,l2}
    #                               Fitness metric for comparing model FC against target FC.
    main()
