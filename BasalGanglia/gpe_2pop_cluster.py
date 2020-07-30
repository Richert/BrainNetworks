import pandas as pd
import numpy as np
from pyrates.utility import ClusterGridSearch
import os
import scipy.io as scio

# parameter definitions
#######################

# simulation parameters
dt = 1e-2
dts = 1.0
T = 300000.0

# stimulation parameters
stim_periods = np.linspace(58.0, 85.0, 100)
stim_amps = np.linspace(50.0, 90.0, 40)
n_infreqs = len(stim_periods)

# model parameters
k_gp = 30.0
k_p = 1.5
k_i = 0.75
k_pi = 1.0
k = 10.0
param_grid = {
        'k_ae': [100.0*k],
        'k_pe': [100.0*k],
        'k_pp': [k*k_gp*k_p/k_i],
        'k_ap': [k*k_gp*k_p*k_i*k_pi],
        'k_aa': [k*k_gp/(k_p*k_i)],
        'k_pa': [k*k_gp*k_i/(k_p*k_pi)],
        'k_ps': [200.0*k],
        'k_as': [200.0*k],
        'eta_e': [0.02],
        'eta_p': [5.5],
        'eta_a': [-6.0],
        'eta_s': [0.002],
        'delta_p': [90.0],
        'delta_a': [120.0],
        'tau_p': [25],
        'tau_a': [20],
        'omega': np.asarray(stim_periods),
        'alpha': np.asarray(stim_amps),
    }

param_map = {
    'k_ae': {'vars': ['weight'], 'edges': [('stn', 'gpe_a')]},
    'k_pe': {'vars': ['weight'], 'edges': [('stn', 'gpe_p')]},
    'k_pp': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_p')]},
    'k_ap': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_a')]},
    'k_aa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_a')]},
    'k_pa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_p')]},
    'k_ps': {'vars': ['weight'], 'edges': [('str', 'gpe_p')]},
    'k_as': {'vars': ['weight'], 'edges': [('str', 'gpe_a')]},
    'eta_p': {'vars': ['gpe_proto_syns_op/eta_i'], 'nodes': ['gpe_p']},
    'eta_a': {'vars': ['gpe_arky_syns_op/eta_a'], 'nodes': ['gpe_a']},
    'eta_e': {'vars': ['stn_dummy_op/eta_e'], 'nodes': ['stn']},
    'eta_s': {'vars': ['str_dummy_op/eta_s'], 'nodes': ['str']},
    'delta_p': {'vars': ['gpe_proto_syns_op/delta_i'], 'nodes': ['gpe_p']},
    'delta_a': {'vars': ['gpe_arky_syns_op/delta_a'], 'nodes': ['gpe_a']},
    'tau_p': {'vars': ['gpe_proto_syns_op/tau_i'], 'nodes': ['gpe_p']},
    'tau_a': {'vars': ['gpe_arky_syns_op/tau_a'], 'nodes': ['gpe_a']},
    'omega': {'vars': ['sl_op/t_off'], 'nodes': ['driver']},
    'alpha': {'vars': ['sl_op/alpha'], 'nodes': ['driver']}
}

# set up cluster
################

nodes = [
            'carpenters',
            'osttimor',
            'spanien',
            'animals',
            'kongo',
            'tschad',
            'uganda',
            'tiber',
            'giraffe',
            'lech',
            'rilke',
            #'dinkel',
            #'rosmarin',
            #'mosambik',
        ]
compute_dir = f"{os.getcwd()}/stn_gpe_syns_opt"
cgs = ClusterGridSearch(nodes=nodes, compute_dir=compute_dir)

# simulations
#############

chunk_size = [
            15,  # carpenters
            20,  # osttimor
            15,  # spanien
            20,  # animals
            10,  # kongo
            10,  # tschad
            15,  # uganda
            10,  # tiber
            15,  # giraffe
            10,  # lech
            5,  # rilke
            #50,  # dinkel
            #10,  # rosmarin
            #10,  # mosambik
        ]

res_file = cgs.run(
            circuit_template="config/stn_gpe/gpe_2pop_driver",
            param_grid=param_grid,
            param_map=param_map,
            simulation_time=T,
            dt=dt,
            inputs={},
            outputs={'r_i': 'gpe_p/gpe_proto_syns_op/R_i',
                     'r_a': 'gpe_a/gpe_arky_syns_op/R_a',
                     'd': 'driver/sl_op/Z1'},
            sampling_step_size=dts,
            permute=True,
            chunk_size=chunk_size,
            worker_env="/data/u_rgast_software/anaconda3/envs/pyrates/bin/python3",
            worker_file=f'{os.getcwd()}/stn_gpe_syns_worker.py',
            worker_kwargs={},
            gs_kwargs={'init_kwargs': {'backend': 'numpy', 'solver': 'scipy', 'step_size': dt, 'matrix_sparseness': 1.0}
                       },
            method='RK45',
            result_concat_axis=0)

results = pd.read_hdf(res_file, key=f'/Results/results')
result_map = pd.read_hdf(res_file, key=f'/Results/result_map')

results_dict = {}
for key in result_map.index:
    data1, data2 = results.loc[:, ('d', key)].values, results.loc[:, ('r_i', key)].values
    results_dict[key] = {"omega": result_map.loc[key, 'omega'], 'alpha': result_map.loc[key, 'w'],
                         "data": np.asarray([data1, data2])}
scio.savemat('lc_data.mat', mdict=results_dict, long_field_names=True)