import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrates.utility.grid_search import grid_search
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter1d
from pyrates.utility import plot_timeseries, create_cmap

# parameter definitions
#######################

# param_grid = {
#         'k_ee': [10.9],
#         'k_ae': [292.4],
#         'k_pe': [352.0],
#         'k_pp': [81.4],
#         'k_ep': [487.8],
#         'k_ap': [65.6],
#         'k_aa': [417.8],
#         'k_pa': [120.2],
#         'k_ps': [516.0],
#         'k_as': [143.6],
#         'eta_e': [-3.7],
#         'eta_p': [-4.7],
#         'eta_a': [-11.6],
#         'delta_e': [0.75],
#         'delta_p': [0.89],
#         'delta_a': [0.72],
#         'tau_e': [13],
#         'tau_p': [25],
#         'tau_a': [20],
#     }

k = 1.0
param_grid = {
        'k_ee': [2.0],
        'k_ae': [6.0],
        'k_pe': [200.0],
        'k_pp': [7.0],
        'k_ep': [50.0],
        'k_ap': [10.0*k],
        'k_aa': [5.0],
        'k_pa': [10.0],
        'k_ps': [8.0],
        'k_as': [24.0],
        'eta_e': [10.0],
        'eta_p': [-6.0],
        'eta_a': [-4.0],
        'delta_e': [0.05],
        'delta_p': [0.6/k],
        'delta_a': [0.3/k],
        'tau_e': [13],
        'tau_p': [25],
        'tau_a': [20],
    }

# fname = "/stn_gpe_healthy_opt1/PopulationDrops/PopulationDrop_0.h5"
# param_grid_vars = ['k_ee', 'k_ae', 'k_pe', 'k_pp', 'k_ep', 'k_ap', 'k_aa', 'k_pa', 'k_ps', 'k_as', 'eta_e', 'eta_p',
#                    'eta_a', 'delta_e', 'delta_p', 'delta_a', 'tau_e' 'tau_p', 'tau_a']
#
# param_grid = pd.read_hdf(fname)
# param_grid = param_grid.loc[:, param_grid_vars]

param_map = {
    'k_ee': {'vars': ['weight'], 'edges': [('stn', 'stn')]},
    'k_ae': {'vars': ['weight'], 'edges': [('stn', 'gpe_a')]},
    'k_pe': {'vars': ['weight'], 'edges': [('stn', 'gpe_p')]},
    'k_pp': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_p')]},
    'k_ep': {'vars': ['weight'], 'edges': [('gpe_p', 'stn')]},
    'k_ap': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_a')]},
    'k_aa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_a')]},
    'k_pa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_p')]},
    'k_ps': {'vars': ['weight'], 'edges': [('str', 'gpe_p')]},
    'k_as': {'vars': ['weight'], 'edges': [('str', 'gpe_a')]},
    'eta_e': {'vars': ['stn_op/eta_e'], 'nodes': ['stn']},
    'eta_p': {'vars': ['gpe_proto_op/eta_i'], 'nodes': ['gpe_p']},
    'eta_a': {'vars': ['gpe_arky_op/eta_a'], 'nodes': ['gpe_a']},
    'delta_e': {'vars': ['stn_op/delta_e'], 'nodes': ['stn']},
    'delta_p': {'vars': ['gpe_proto_op/delta_i'], 'nodes': ['gpe_p']},
    'delta_a': {'vars': ['gpe_arky_op/delta_a'], 'nodes': ['gpe_a']},
    'tau_e': {'vars': ['stn_op/tau_e'], 'nodes': ['stn']},
    'tau_p': {'vars': ['gpe_proto_op/tau_i'], 'nodes': ['gpe_p']},
    'tau_a': {'vars': ['gpe_arky_op/tau_a'], 'nodes': ['gpe_a']},
}

param_scalings = [
            ('delta_e', 'tau_e', 2.0),
            ('delta_p', 'tau_p', 2.0),
            ('delta_a', 'tau_a', 2.0),
            ('k_ee', 'delta_e', 0.5),
            ('k_ep', 'delta_e', 0.5),
            ('k_pe', 'delta_p', 0.5),
            ('k_pp', 'delta_p', 0.5),
            ('k_pa', 'delta_p', 0.5),
            ('k_ps', 'delta_p', 0.5),
            ('k_ae', 'delta_a', 0.5),
            ('k_ap', 'delta_a', 0.5),
            ('k_aa', 'delta_a', 0.5),
            ('k_as', 'delta_a', 0.5),
            ('eta_e', 'delta_e', 1.0),
            ('eta_p', 'delta_p', 1.0),
            ('eta_a', 'delta_a', 1.0),
            ]

conditions = [{},  # healthy control -> GPe_p: 60 Hz, STN: 20 Hz, GPe_a: 30 Hz
              {'k_pe': 0.2, 'k_ae': 0.2},  # AMPA blockade in GPe -> GPe_p: 40 Hz
              {'k_pe': 0.2, 'k_pp': 0.2, 'k_pa': 0.2, 'k_ae': 0.2, 'k_aa': 0.2, 'k_ap': 0.2,
               'k_ps': 0.2, 'k_as': 0.2},  # AMPA blockade and GABAA blockade in GPe -> GPe_p: 70 Hz
              {'k_pp': 0.2, 'k_pa': 0.2, 'k_aa': 0.2, 'k_ap': 0.2, 'k_ps': 0.2,
               'k_as': 0.2},  # GABAA blockade in GPe -> GPe_p: 100 Hz
              #{'k_pe': 0.0, 'k_ae': 0.0},  # STN blockade -> GPe_p: 20 HZ
              #{'k_pe': 0.0, 'k_ae': 0.0, 'k_pp': 0.2, 'k_pa': 0.2, 'k_aa': 0.2, 'k_ap': 0.2,
              # 'k_ps': 0.2, 'k_as': 0.2},  # STN blockade + GABAA blockade in GPe -> GPe_p: 60 Hz
              {'k_ep': 0.2}  # GABAA blocker in STN -> STN: 40 Hz, GPe_p: 100 Hz
              ]

T = 200.
dt = 1e-2
dts = 1e-1

# sim_steps = int(np.round(T/dt, decimals=0))
# ctx = np.zeros((sim_steps, 1))
# ctx[int(1000/dt), 0] = 20000.0
# ctx = gaussian_filter1d(ctx, 100, axis=0)
# stria = np.zeros((sim_steps, 1))
# stria[int((1000+20.0)/dt), 0] = 400.0
# stria = gaussian_filter1d(stria, 100, axis=0)

# plt.figure()
# plt.plot(ctx)
# plt.plot(stria)
# plt.show()

# simulations
#############

# param_grid_final = pd.DataFrame()
# for c_dict in deepcopy(conditions):
#     for key in param_grid:
#         if key in c_dict:
#             c_dict[key] = np.asarray(param_grid[key]) * c_dict[key]
#         elif key in param_grid:
#             c_dict[key] = np.asarray(param_grid[key])
#     for key, key_tmp, power in param_scalings:
#         c_dict[key] = c_dict[key] * c_dict[key_tmp] ** power
#     param_grid_tmp = pd.DataFrame.from_dict(c_dict)
#     param_grid_final = param_grid_final.append(param_grid_tmp)
# param_grid_final.index = np.arange(0, param_grid_final.shape[0])
# results, result_map = grid_search(
#         circuit_template="config/stn_gpe/stn_gpe",
#         param_grid=param_grid_final,
#         param_map=param_map,
#         simulation_time=T,
#         step_size=dt,
#         permute=False,
#         sampling_step_size=dts,
#         inputs={},
#         outputs={'r_i': 'gpe_p/gpe_proto_op/R_i'},
#         init_kwargs={
#             'backend': 'numpy', 'solver': 'scipy', 'step_size': dt}
#     )
# results = results*1e3
# results.plot()
# plt.show()

for c_dict in deepcopy(conditions):
    for key in param_grid:
        if key in c_dict:
            c_dict[key] = np.asarray(param_grid[key]) * c_dict[key]
        elif key in param_grid:
            c_dict[key] = np.asarray(param_grid[key])
    for key, key_tmp, power in param_scalings:
        c_dict[key] = c_dict[key] * c_dict[key_tmp] ** power
    param_grid_tmp = pd.DataFrame.from_dict(c_dict)
    results, result_map = grid_search(
        circuit_template="config/stn_gpe/stn_gpe",
        param_grid=param_grid_tmp,
        param_map=param_map,
        simulation_time=T,
        step_size=dt,
        permute=True,
        sampling_step_size=dts,
        inputs={
            #'stn/stn_op/ctx': ctx, 'str/str_dummy_op/I': stria
            },
        outputs={'r_e': 'stn/stn_op/R_e', 'r_i': 'gpe_p/gpe_proto_op/R_i', 'r_a': 'gpe_a/gpe_arky_op/R_a'},
        init_kwargs={
            'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
        method='RK45'
    )

    results = results*1e3
    results.plot()
    #plt.xlim(0, 100)
    #plt.ylim(0, 200)
    plt.show()
