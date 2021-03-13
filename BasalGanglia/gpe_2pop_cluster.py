import pandas as pd
import numpy as np
from pyrates.utility.grid_search import ClusterGridSearch
import os
import scipy.io as scio

# parameter definitions
#######################

# simulation parameters
dt = 1e-2
dts = 1.0
T = 320000.0

# stimulation parameters
stim_periods = np.linspace(60.0, 85.0, 25)
stim_amps = np.linspace(0.0, 60.0, 30)
n_infreqs = len(stim_periods)

# model parameters
k_gp = 1.0
k = 10.0
param_grid = {
        'k_ae': [k*1.5],
        'k_pe': [k*5.0],
        'k_pp': [1.5*k*k_gp],
        'k_ap': [2.0*k*k_gp],
        'k_aa': [0.1*k*k_gp],
        'k_pa': [0.5*k*k_gp],
        'k_ps': [k*10.0],
        'k_as': [k*1.0],
        'eta_e': [0.02],
        'eta_p': [12.0],
        'eta_a': [26.0],
        'eta_s': [0.002],
        'delta_p': [9.0],
        'delta_a': [3.0],
        'tau_p': [18],
        'tau_a': [32],
        'omega': stim_periods,
        'a2': np.asarray(stim_amps),
        'a1': [0.0]
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
    'a1': {'vars': ['weight'], 'edges': [('driver', 'gpe_p', 0)]},
    'a2': {'vars': ['weight'], 'edges': [('driver', 'gpe_p', 1)]}
}

# set up cluster
################

nodes = [
            'carpenters',
            #'osttimor',
            'spanien',
            #'animals',
            'kongo',
            'tschad',
            'uganda',
            'tiber',
            'giraffe',
            #'lech',
            #'rilke',
            #'dinkel',
            'rosmarin',
            #'mosambik',
        ]
compute_dir = f"{os.getcwd()}/gpe_2pop_forced"
cgs = ClusterGridSearch(nodes=nodes, compute_dir=compute_dir)

# simulations
#############

chunk_size = [
            50,  # carpenters
            #20,  # osttimor
            30,  # spanien
            #50,  # animals
            20,  # kongo
            20,  # tschad
            50,  # uganda
            20,  # tiber
            30,  # giraffe
            #20,  # lech
            #10,  # rilke
            #50,  # dinkel
            10,  # rosmarin
            #10,  # mosambik
        ]

res_file = cgs.run(
            circuit_template="/data/u_rgast_software/PycharmProjects/BrainNetworks/BasalGanglia/config/stn_gpe/gpe_2pop_driver",
            param_grid=param_grid,
            param_map=param_map,
            simulation_time=T,
            dt=dt,
            inputs={},
            outputs={'r_i': 'gpe_p/gpe_proto_syns_op/R_i',
                     'd': 'driver/sl_op/Z1'},
            sampling_step_size=dts,
            permute_grid=True,
            chunk_size=chunk_size,
            worker_env="/data/u_rgast_software/anaconda3/envs/pyrates/bin/python3",
            worker_file=f'{os.getcwd()}/gpe_2pop_worker.py',
            worker_kwargs={'time_lim': 4000.0, 'cpu_lim': True, 'nproc_lim': False, 'memory_lim': False},
            gs_kwargs={'init_kwargs': {'backend': 'numpy', 'solver': 'scipy', 'step_size': dt, 'matrix_sparseness': 0.1}
                       },
            method='RK45',
            result_concat_axis=1)

results = pd.read_hdf(res_file, key=f'/Results/results')
result_map = pd.read_hdf(res_file, key=f'/Results/result_map')

results_dict = {}
cutoff = 20.0
for key in result_map.index:
    data1, data2 = results.loc[cutoff:, ('d', key)].values, results.loc[cutoff:, ('r_i', key)].values
    results_dict[key] = {"omega": result_map.loc[key, 'omega'], 'alpha': result_map.loc[key, 'alpha'],
                         "data": np.asarray([data1, data2])}
scio.savemat('ss_data.mat', mdict=results_dict, long_field_names=True)
