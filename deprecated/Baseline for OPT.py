
from __future__ import division
import argparse
import json
import multiprocessing as mp
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.matlib

# %matplotlib inline
import pandas
import scipy.io as sio
import brian2
import brian2 as b2
from joblib import Parallel, delayed

try:
    import cma
except ImportError:  # pragma: no cover - handled when optimisation is invoked
    cma = None

# np.random.seed(1234)


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

        # target stimulus off time
        #   'stim_on': 8 * brian2.second,
        #   'stim_off': 8.4 * brian2.second,


        # distractor stimulus on time (if using)
          'distract_on': 2 * brian2.second,
        # distractor stimulus off time (if using)
          'distract_off': 2.4 * brian2.second,


    # background inputs

          'I_background_e': 310 * brian2.pA, #default
          # 'I_background_e': 155 * brian2.pA,
          'I_background_i': 300 * brian2.pA, #default
          # 'I_background_i': 180 * brian2.pA,
          'I_background_dend': 30 * brian2.pA, #default

          'tau_noise': 2 * 8 * brian2.ms, #NEW TESTING

    # stimulus strength
    #       'stim_strength': 0.1 * brian2.nA, #default
          'stim_strength': 0.0 * brian2.nA,


    # dopamine release level for the current simulation
    #     'da_rel': 1.5, #default
          'da_rel': 0.5,

          # 'std_noise': 8 * brian2.pA,
          'std_noise': 0.0 * brian2.pA,

          'trial_length': 5 * brian2.second,
        # Long-range connectivity strengths
        #   'mu_ee': 1.45, #default
            'mu_ee': 1.55,
        #   'mu_ie': 2.24, #default
            'mu_ie': 2.6,

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


@lru_cache(maxsize=1)
def load_anatomy():
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

    
    return (sln, fln, hierarchy, area_list_SLN,
        df_fln, df_sln, D1_density_raw, spine_count_raw, df_raw_anatomy)







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






def initialise_variables(PARAMS,num_areas,num_pops,num_e_pops,area_list_SLN,pops):

    # Initialise
    num_iterations = int(PARAMS['trial_length']/PARAMS['dt'])

    # Choose initial values for rates and synapse variables
    R0 = np.matlib.repmat(np.array([PARAMS['r_0_e'],0,PARAMS['r_0_e'],PARAMS['r_0_e'],PARAMS['r_0_e']]), num_areas, 1) * brian2.Hz
    R = np.zeros((num_iterations,num_areas,num_pops)) * brian2.Hz
    R[0,:,:] = R0

    s_nmda = np.zeros((num_iterations,num_areas,num_pops))
    s_ampa = np.zeros((num_iterations,num_areas,num_pops))
    s_gaba = np.zeros((num_iterations,num_areas,num_pops))
    s_gaba_dend = np.zeros((num_iterations,num_areas,num_pops))
    s_gaba_dend[0,:,:] = 1
    s_gaba[0,:,:] = 1
    s_adapt = np.zeros((num_iterations,num_areas,num_pops))

    # # Preassign external inputs
    I_ext    = np.zeros((num_iterations,num_areas,num_pops)) * brian2.amp

    # Let's apply external stimulation to V1 populations E1 & E2
    I_ext[int(PARAMS['stim_on']/PARAMS['dt']):int(PARAMS['stim_off']/PARAMS['dt']),area_list_SLN.index('V1'),pops.index('E1dend')] = PARAMS['stim_strength']
    # I_ext[int(PARAMS['stim_on']/PARAMS['dt']):int(PARAMS['stim_off']/PARAMS['dt']),area_list_SLN.index('32'),pops.index('E1dend')] = PARAMS['stim_strength']


    # I_ext[int(PARAMS['stim_on2']/PARAMS['dt']):int(PARAMS['stim_off2']/PARAMS['dt']),area_list_SLN.index('V1'),pops.index('E1dend')] = PARAMS['stim_strength']
    # I_ext[int(PARAMS['stim_on3']/PARAMS['dt']):int(PARAMS['stim_off3']/PARAMS['dt']),area_list_SLN.index('V1'),pops.index('E1dend')] = PARAMS['stim_strength']

    # No distractor
#     I_ext[int(PARAMS['distract_on']/PARAMS['dt']):int(PARAMS['distract_off']/PARAMS['dt']),area_list_SLN.index('V1'),pops.index('E2dend')] = PARAMS['stim_strength']

    # Create matrices in which we can store the currents
    I_lr_nmda    =  np.zeros((num_iterations,num_areas,num_pops)) * brian2.pA
    I_lr_ampa    =  np.zeros((num_iterations,num_areas,num_pops)) * brian2.pA
    I_local_nmda =  np.zeros((num_iterations,num_areas,num_pops)) * brian2.pA
    I_local_ampa =  np.zeros((num_iterations,num_areas,num_pops)) * brian2.pA
    I_local_gaba =  np.zeros((num_iterations,num_areas,num_pops)) * brian2.pA
    I_soma_dend  =  np.zeros((num_iterations,num_areas,num_pops)) * brian2.pA
    I_total      =  np.zeros((num_iterations,num_areas,num_pops)) * brian2.pA
    I_total_abs  =  np.zeros((num_iterations,num_areas,num_pops)) * brian2.pA
    I_exc_dend   = np.zeros((num_iterations,num_areas,num_e_pops)) * brian2.pA
    I_inh_dend   = np.zeros((num_iterations,num_areas,num_e_pops)) * brian2.pA
    I_local_gaba_dend =  np.zeros((num_iterations,num_areas,num_e_pops)) * brian2.pA
    I_adapt = np.zeros((num_iterations,num_areas,num_pops)) * brian2.pA

    # Define background inputs
    I_0 = np.zeros((num_areas,num_pops)) * brian2.pA
    I_0[:,[pops.index('E1soma')]] = PARAMS['I_background_e']
    I_0[:,[pops.index('E1dend')]] = PARAMS['I_background_dend']
    I_0[:,[pops.index('PV'),pops.index('SST1'),pops.index('VIP1')]] = PARAMS['I_background_i']

    # Let's set up the noise. We will model the noise as an Ornstein-Uhlenbeck process.
    # Gaussian noise. mean 0, std 1. Dims: timesteps, local populations, areas
    eta = np.random.normal(loc=0.0, scale=1.0, size=(num_iterations,num_areas,num_pops))
    # eta[:,:,1:] = 0 #remove noise from non-soma                                 # NEW LINE

    # prepare the right hand side of the above equation
    # noise_rhs = eta*((np.sqrt(PARAMS['tau_ampa']*np.power(PARAMS['std_noise'],2))*np.sqrt(PARAMS['dt']))/PARAMS['tau_ampa'])
    noise_rhs = eta * PARAMS["std_noise"] * np.sqrt(2.0 * PARAMS["dt"] / PARAMS["tau_ampa"]) #NEW LINE
    # noise_rhs = eta * PARAMS["std_noise"] * np.sqrt(2.0 * PARAMS["dt"] / PARAMS["tau_noise"]) #NEW LINE

    noise_rhs[:,:,1:2] = 0 # remove noise from dendrites
    # noise_rhs[:,1:,:] = 0 # remove noise from everywhere but v1                    # NEW LINE
    I_noise = np.zeros((num_areas , num_pops )) *brian2.pA

    return(num_iterations,R,s_nmda,s_ampa,s_gaba,s_gaba_dend,s_adapt
           ,I_ext,I_lr_nmda,I_lr_ampa,I_local_nmda,I_local_ampa,I_local_gaba
           ,I_soma_dend,I_total,I_exc_dend,I_inh_dend,I_local_gaba_dend,I_adapt
           ,I_0,I_noise,noise_rhs,I_total_abs)
        






def large_scale_da_model(pops, num_pops, num_e_pops, num_areas, e_grad, g_adapt, ampa_frac, nmda_frac
                          , J_nmda, J_ampa, J_gaba, J_gaba_dend, W_superficial, W_deep, lr_targets
                          , nmda_da_grad, e_pv_da_mat, e_sst_da_mat, m_da_grad,num_iterations,R,s_nmda
                          ,s_ampa,s_gaba,s_gaba_dend,s_adapt
                          ,I_ext,I_lr_nmda,I_lr_ampa,I_local_nmda,I_local_ampa,I_local_gaba
                          ,I_soma_dend,I_total,I_exc_dend,I_inh_dend,I_local_gaba_dend,I_adapt
                          ,I_0,I_noise,noise_rhs,parameters,lr_targets_FEF,I_total_abs,area_list_SLN):

    # noise_delay = 8
    fef_idx = [area_list_SLN.index('8m'), area_list_SLN.index('8l')]
    for i_t in range(1,num_iterations):

        # update noise - dims = num local pops x num areas
        I_noise = I_noise + -I_noise*(parameters['dt']/(parameters['tau_ampa'])) + noise_rhs[i_t-1,:,:]
        # I_noise = I_noise + -I_noise*(parameters['dt']/(parameters['tau_noise'])) + noise_rhs[i_t-1,:,:] #NEW LINE

        # Long range NMDA to E populations
        I_lr_nmda[i_t-1,:,:2]   = ((e_grad*parameters['mu_ee']*nmda_da_grad)*W_superficial).dot(s_nmda[i_t-1,:,:1]).dot(nmda_frac[:2]*lr_targets[:2,:].T)
        # Long range NMDA to I populations 
        I_lr_nmda[i_t-1,:,2:]   = parameters['mu_ie']*e_grad*nmda_da_grad*(W_deep.dot(s_nmda[i_t-1,:,:1])).dot(nmda_frac[2:]*lr_targets[2:,:].T)
        # Long range NMDA to I populations in FEF
        I_lr_nmda[i_t-1,fef_idx,2:]   = parameters['mu_ie']*e_grad[fef_idx]*nmda_da_grad[fef_idx]*(W_deep[fef_idx,:].dot(s_nmda[i_t-1,:,:1])).dot(nmda_frac[2:]*lr_targets_FEF[2:,:].T)

        
        # Long range AMPA to E populations 
        I_lr_ampa[i_t-1,:,:2]   = ((e_grad*parameters['mu_ee'])*W_superficial).dot(s_ampa[i_t-1,:,:1]).dot(ampa_frac[:2]*lr_targets[:2,:].T)
        # Long range AMPA to I populations 
        I_lr_ampa[i_t-1,:,2:]   = parameters['mu_ie']*e_grad*(W_deep.dot(s_ampa[i_t-1,:,:1])).dot(ampa_frac[2:]*lr_targets[2:,:].T)
        # Long range AMPA to I populations in FEF
        I_lr_ampa[i_t-1,fef_idx,2:]   = parameters['mu_ie']*e_grad[fef_idx]*(W_deep[fef_idx,:].dot(s_ampa[i_t-1,:,:1])).dot(ampa_frac[2:]*lr_targets_FEF[2:,:].T)

        
        # local NMDA
        I_local_nmda[i_t-1,:,:] = nmda_frac*nmda_da_grad*e_grad*J_nmda.dot(s_nmda[i_t-1,:,:].T).T

        # local AMPA
        I_local_ampa[i_t-1,:,:] = ampa_frac*e_grad*J_ampa.dot(s_ampa[i_t-1,:,:].T).T

        # sum up all the local GABA current onto E and I cell somas

        I_local_gaba[i_t-1,:,:] = e_pv_da_mat*(J_gaba.dot(s_gaba[i_t-1,:,:].T).T)

        # sum up all the local GABA current onto dendrites
        I_local_gaba_dend[i_t-1,:,:] = e_sst_da_mat*(J_gaba_dend.dot(s_gaba_dend[i_t-1,:,:].T).T)

        # calculate the dendrite-to-soma current
        I_exc_dend[i_t-1,:,:] = I_local_nmda[i_t-1,:,1:2] + I_lr_nmda[i_t-1,:,1:2] + I_local_ampa[i_t-1,:,1:2] + I_lr_ampa[i_t-1,:,1:2] +I_0[:,1:2] + I_ext[i_t-1,:,1:2] + I_noise[:,1:2] #Noise term is always 0 here

        I_inh_dend[i_t-1,:,:] = I_local_gaba_dend[i_t-1,:,:] 

        I_soma_dend[i_t-1,:,:1]  = dendrite_input_output(I_exc_dend[i_t-1,:,:],I_inh_dend[i_t-1,:,:],parameters)

        # adaptation current
        I_adapt[i_t-1,:,:] = (g_adapt+g_m*m_da_grad)*s_adapt[i_t-1,:,:]

        # Define total input current as sum of local NMDA & GABA inputs, with background and external currents, 
        # noise and long-range NMDA inputs, and an adaptation current
        I_total[i_t-1,:,:] = I_local_nmda[i_t-1,:,:] + I_local_ampa[i_t-1,:,:] +  I_local_gaba[i_t-1,:,:] + I_0 + I_ext[i_t-1,:,:] + I_noise + I_lr_nmda[i_t-1,:,:] + I_lr_ampa[i_t-1,:,:] + I_soma_dend[i_t-1,:,:] + I_adapt[i_t-1,:,:]

        I_total_abs[i_t-1,:,:] = abs(I_local_nmda[i_t-1,:,:]) + abs(I_local_ampa[i_t-1,:,:]) +  abs(I_local_gaba[i_t-1,:,:]) + abs(I_0) + abs(I_ext[i_t-1,:,:]) + abs(I_noise) + abs(I_lr_nmda[i_t-1,:,:]) + abs(I_lr_ampa[i_t-1,:,:]) + abs(I_soma_dend[i_t-1,:,:]) + abs(I_adapt[i_t-1,:,:])

        
        # Update the firing rates of the two excitatory populations.
        R[i_t,:,:1] = R[i_t-1,:,:1] + parameters['dt']*current_to_frequency(I_total[i_t-1,:,:1],'E',parameters)/parameters['tau_ampa'] -parameters['dt']*R[i_t-1,:,:1]/parameters['tau_ampa']

        # Update the firing rates of the PV population. 
        R[i_t,:,2] =  R[i_t-1,:,2] + parameters['dt']*current_to_frequency(I_total[i_t-1,:,2],'PV',parameters)/parameters['tau_ampa'] -parameters['dt']*R[i_t-1,:,2]/parameters['tau_ampa']

        # Update the firing rates of the SST populations. 
        R[i_t,:,3:4] =  R[i_t-1,:,3:4] + parameters['dt']*current_to_frequency(I_total[i_t-1,:,3:4],'SST',parameters)/parameters['tau_ampa'] -parameters['dt']*R[i_t-1,:,3:4]/parameters['tau_ampa']

        # Update the firing rates of the VIP populations. 
        R[i_t,:,4:] =  R[i_t-1,:,4:] +  parameters['dt']*current_to_frequency(I_total[i_t-1,:,4:],'VIP',parameters)/parameters['tau_ampa'] -parameters['dt']*R[i_t-1,:,4:]/parameters['tau_ampa']

        # Update the NMDA synapses
        s_nmda[i_t,:,:1] = s_nmda[i_t-1,:,:1] + parameters['dt']*NMDA_deriv(s_nmda[i_t-1,:,:1],R[i_t,:,:1],parameters)

        # Update the AMPA synapses
        s_ampa[i_t,:,:1] = s_ampa[i_t-1,:,:1] + parameters['dt']*AMPA_deriv(s_ampa[i_t-1,:,:1],R[i_t,:,:1],parameters)

        # Update the GABA synapses onto the somata
        s_gaba[i_t,:,2:] = s_gaba[i_t-1,:,2:] + parameters['dt']*GABA_deriv(s_gaba[i_t-1,:,2:],R[i_t,:,2:],parameters,'soma')

        # Update the GABA synapses onto the dendrites
        s_gaba_dend[i_t,:,2:] = s_gaba_dend[i_t-1,:,2:] + parameters['dt']*GABA_deriv(s_gaba_dend[i_t-1,:,2:],R[i_t,:,2:],parameters,'dendrite')

        # Update the adaptation variable
        s_adapt[i_t,:,:] = s_adapt[i_t-1,:,:] + parameters['dt']*adaptation_deriv(s_adapt[i_t-1,:,:],R[i_t,:,:],parameters)

    return(R)


LEGACY_ANALYSIS_CODE = r'''
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
    seed = bold[:, seed_idx]
    return np.array([
        np.corrcoef(seed, bold[:, a])[0, 1]
        for a in range(bold.shape[1])
    ])


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
'''


FAILURE_PENALTY = 1e9
# FAILURE_PENALTY = 0

PARAMETER_SPACE = [
    ('g_e_sst_min', -0.135, -0.03, brian2.nA, 0.002, float(PARAMS['g_e_sst_min'] / brian2.nA)),
    ('g_e_sst_max', -0.17, -0.09, brian2.nA, 0.002, float(PARAMS['g_e_sst_max'] / brian2.nA)),
    ('g_e_pv_max', -0.6, -0.3, brian2.nA, 0.01, float(PARAMS['g_e_pv_max'] / brian2.nA)),
    ('g_e_pv_min', -0.1, -0.0001, brian2.nA, 0.01, float(PARAMS['g_e_pv_min'] / brian2.nA)),
    ('ampa_frac_pv', 0.10, 0.90, None, 0.01, float(PARAMS['ampa_frac_pv'])),
    ('e_grad_min', 0.05, 0.65, None, 0.01, float(PARAMS['e_grad_min'])),
    ('I_background_e', 230.0, 390.0, brian2.pA, 0.5, float(PARAMS['I_background_e'] / brian2.pA)),
    ('I_background_i', 230.0, 390.0, brian2.pA, 0.5, float(PARAMS['I_background_i'] / brian2.pA)),
    ('I_background_dend', 0.0, 40.0, brian2.pA, 0.25, float(PARAMS['I_background_dend'] / brian2.pA)),
    ('b1', 0.2, 3.0, None, 0.3, float(PARAMS['b1'])),
    ('mu_ee', 0.5, 6.0, None, 0.03, float(PARAMS['mu_ee'])),
    ('mu_ie', 0.5, 6.0, None, 0.03, float(PARAMS['mu_ie'])),
]

FITNESS_CONFIG = {

}


def initial_parameter_vector():
    return np.array([curr for (_, _, _, _, _, curr) in PARAMETER_SPACE], dtype=float)


def sigma_vector():
    return np.array([sigma for (_, _, _, _, sigma, _) in PARAMETER_SPACE], dtype=float)


def parameter_bounds():
    lower = [lo for (_, lo, _, _, _, _) in PARAMETER_SPACE]
    upper = [hi for (_, _, hi, _, _, _) in PARAMETER_SPACE]
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


def save_default_params(path):
    with path.open('wb') as handle:
        pickle.dump(PARAMS, handle)


def _mean_rate_in_window(rate, t, t_start, t_end):
    mask = (t >= t_start) & (t < t_end)
    if not np.any(mask):
        return 0.0
    return float(np.mean(rate[mask]))


def run_simulation(param_vector, override_params=None):
    params_run = PARAMS.copy()
    params_run.update(vector_to_param_dict(param_vector))
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

        (num_iterations, R, s_nmda, s_ampa, s_gaba, s_gaba_dend, s_adapt,
         I_ext, I_lr_nmda, I_lr_ampa, I_local_nmda, I_local_ampa, I_local_gaba,
         I_soma_dend, I_total, I_exc_dend, I_inh_dend, I_local_gaba_dend, I_adapt,
         I_0, I_noise, noise_rhs, I_total_abs) = initialise_variables(
            params_run, num_areas, num_pops, num_e_pops, area_list_SLN, pops
        )

        R = large_scale_da_model(
            pops, num_pops, num_e_pops, num_areas, e_grad, g_adapt, ampa_frac, nmda_frac,
            J_nmda, J_ampa, J_gaba, J_gaba_dend, W_superficial, W_deep, lr_targets,
            nmda_da_grad, e_pv_da_mat, e_sst_da_mat, m_da_grad, num_iterations, R, s_nmda,
            s_ampa, s_gaba, s_gaba_dend, s_adapt, I_ext, I_lr_nmda, I_lr_ampa,
            I_local_nmda, I_local_ampa, I_local_gaba, I_soma_dend, I_total, I_exc_dend,
            I_inh_dend, I_local_gaba_dend, I_adapt, I_0, I_noise, noise_rhs, params_run,
            lr_targets_FEF, I_total_abs, area_list_SLN
        )

        return {
            'params': params_run,
            'area_names': area_list_SLN,
            'E1_V2': np.asarray(R[:, area_list_SLN.index('V2'), 0] / brian2.Hz, dtype=float),
            'E1_46': np.asarray(R[:, area_list_SLN.index('9/46d'), 0] / brian2.Hz, dtype=float),
            'E2_all': np.asarray(R[:, :, 1] / brian2.Hz, dtype=float),
        }
    except Exception:
        return None


def summarise_simulation(simulation, fitness_config=None):
    if simulation is None:
        return None

    config = FITNESS_CONFIG.copy()
    if fitness_config:
        config.update(fitness_config)

    e1_v2 = simulation['E1_V2']
    e1_46 = simulation['E1_46']
    e2_all = simulation['E2_all']

    if (
        not np.all(np.isfinite(e1_v2))
        or not np.all(np.isfinite(e1_46))
        or not np.all(np.isfinite(e2_all))
    ):
        return None

    n_steps = e1_v2.shape[0]
    t = np.arange(n_steps) * simulation['params']['dt']
    t_ignore = config['ignore_initial']

    stim_e1_v2 = _mean_rate_in_window(
        e1_v2,
        t,
        simulation['params']['stim_on'],
        simulation['params']['stim_off'],
    )

    off_mask_v2 = (
        (t >= t_ignore)
        & ((t < simulation['params']['stim_on']) | (t >= simulation['params']['stim_off']))
    )
    mean_offstim_e1_v2 = float(np.mean(e1_v2[off_mask_v2])) if np.any(off_mask_v2) else 0.0

    persistent_e1_46 = _mean_rate_in_window(
        e1_46,
        t,
        simulation['params']['stim_off'],
        t[-1] + simulation['params']['dt'],
    )
    mean_pre_e1_46 = _mean_rate_in_window(
        e1_46,
        t,
        t_ignore,
        simulation['params']['stim_on'],
    )

    mask_e2 = (t >= t_ignore)
    if np.any(mask_e2):
        mean_e2_per_area = np.mean(e2_all[mask_e2, :], axis=0)
        e2_metric = float(np.max(mean_e2_per_area))
    else:
        e2_metric = 0.0

    return {

    }


def fitness_from_summary(summary, fitness_config=None):
    if summary is None:
        return FAILURE_PENALTY

    config = FITNESS_CONFIG.copy()
    if fitness_config:
        config.update(fitness_config)

    baseline_max = config['baseline_max']


    fitness = (

    )
    return float(fitness) if np.isfinite(fitness) else FAILURE_PENALTY


def evaluate_fitness(param_vector, fitness_config=None):
    simulation = run_simulation(param_vector)
    summary = summarise_simulation(simulation, fitness_config=fitness_config)
    return fitness_from_summary(summary, fitness_config=fitness_config)


def parallel_fitness(population, workers=None, fitness_config=None):
    if workers is None:
        workers = min(mp.cpu_count(), len(population))
    return Parallel(n_jobs=workers, backend='threading')(
        delayed(evaluate_fitness)(candidate, fitness_config=fitness_config)
        for candidate in population
    )


def append_generation_log(log_path, generation, population, fitness_values):
    best_index = int(np.argmin(fitness_values))
    best_vector = np.asarray(population[best_index], dtype=float)
    record = {
        'generation': int(generation),
        'best_fitness': float(fitness_values[best_index]),
        'mean_fitness': float(np.mean(fitness_values)),
        'best_vector': best_vector.tolist(),
        'best_params': vector_to_plain_dict(best_vector),
    }
    with log_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record) + '\n')


def write_final_summary(log_dir, result, stop_reasons):
    best_vector = np.asarray(result.xbest, dtype=float)
    summary = {
        'best_fitness': float(result.fbest),
        'best_vector': best_vector.tolist(),
        'best_params': vector_to_plain_dict(best_vector),
        'stop_reasons': stop_reasons,
    }
    with (log_dir / 'result_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)


def run_cmaes(
    *,
    sigma0=20.0,
    popsize=6,
    maxfevals=2200,
    workers=None,
    ftarget=None,
    seed=None,
    log_dir='cmaes_logs',
    fitness_config=None,
):
    if cma is None:
        raise ImportError("The 'cma' package is required to run optimisation.")

    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    save_default_params(log_dir / 'default_params.pck')

    x0 = initial_parameter_vector()
    cma_options = {
        'bounds': parameter_bounds(),
        'popsize': popsize,
        'maxfevals': maxfevals,
        'verb_log': 1,
        'verb_disp': 1,
        'CMA_stds': sigma_vector(),
        'verb_filenameprefix': str(log_dir / 'outcmaes'),
    }
    if ftarget is not None:
        cma_options['ftarget'] = float(ftarget)
    if seed is not None:
        cma_options['seed'] = int(seed)

    es = cma.CMAEvolutionStrategy(x0, sigma0, inopts=cma_options)
    generation_log = log_dir / 'generations.jsonl'
    generation = 0

    while not es.stop():
        population = es.ask()
        fitness_values = parallel_fitness(population, workers=workers, fitness_config=fitness_config)
        es.tell(population, fitness_values)
        es.logger.add()
        es.disp()

        best_index = int(np.argmin(fitness_values))
        best_vector = np.asarray(population[best_index], dtype=float)
        best_fitness = float(fitness_values[best_index])

        np.save(log_dir / 'best_params_so_far.npy', best_vector)
        np.save(log_dir / 'best_fitness_so_far.npy', np.array(best_fitness))
        append_generation_log(generation_log, generation, population, fitness_values)
        generation += 1

    best_vector = np.asarray(es.result.xbest, dtype=float)
    best_fitness = float(es.result.fbest)
    np.save(log_dir / 'best_params_final.npy', best_vector)
    np.save(log_dir / 'best_fitness_final.npy', np.array(best_fitness))
    write_final_summary(log_dir, es.result, es.stop())

    print("Stop reasons:", es.stop())
    print("Best fitness:", best_fitness)
    print("Best parameters:", vector_to_plain_dict(best_vector))
    return best_vector


def parse_args():
    parser = argparse.ArgumentParser(description='Run CMA-ES for the OPT baseline model.')
    parser.add_argument('--sigma0', type=float, default=20.0)
    parser.add_argument('--popsize', type=int, default=40)
    parser.add_argument('--maxfevals', type=int, default=2200)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--ftarget', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--log-dir', default='cmaes_logs')
    return parser.parse_args()


def main():
    args = parse_args()
    return run_cmaes(
        sigma0=args.sigma0,
        popsize=args.popsize,
        maxfevals=args.maxfevals,
        workers=args.workers,
        ftarget=args.ftarget,
        seed=args.seed,
        log_dir=args.log_dir,
    )


if __name__ == '__main__':
    main()
