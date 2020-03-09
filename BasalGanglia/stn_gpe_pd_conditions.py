from pyrates.utility import grid_search
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import scipy.io as scio

conditions = [{'k_ie': 0.2},  # PD + AMPA blockade in GPe
              {'k_ii': 0.2, 'k_str': 0.2},  # PD + GABAA bloacked in GPe
              {'k_ee': 0.2},  # PD + AMPA blockade in STN
              {'k_ei': 0.0}  # PD + GPe blockade
              ]

target = [[30, 40],   # healthy control
          [np.nan, 20],  # ampa blockade in GPe
          [np.nan, 90],  # ampa and gabaa blockade in GPe
          [30.0, np.nan],  # GABAA blockade in GPe
          [50.0, np.nan],  # STN blockade
          ],

param_map = {
        'k_ee': {'vars': ['stn_op/k_ee'], 'nodes': ['stn']},
        'k_ei': {'vars': ['stn_op/k_ei'], 'nodes': ['stn']},
        'k_ie': {'vars': ['gpe_proto_op/k_ie'], 'nodes': ['gpe']},
        'k_ii': {'vars': ['gpe_proto_op/k_ii'], 'nodes': ['gpe']},
        'eta_e': {'vars': ['stn_op/eta_e'], 'nodes': ['stn']},
        'eta_i': {'vars': ['gpe_proto_op/eta_i'], 'nodes': ['gpe']},
        'k_str': {'vars': ['gpe_proto_op/k_str'], 'nodes': ['gpe']},
        'delta_e': {'vars': ['stn_op/delta_e'], 'nodes': ['stn']},
        'delta_i': {'vars': ['gpe_proto_op/delta_i'], 'nodes': ['gpe']}
    }

k = 1.5
d = 0.4
param_grid_orig = {
        'k_ee': [0.0*k],
        'k_ei': [40.3*k],
        'k_ie': [81.3*k],
        'k_ii': [33.7*k],
        'k_str': [93.2*k + 200.0],
        'eta_e': [0.1],
        'eta_i': [11.0],
        'delta_e': [10.6*d],
        'delta_i': [14.7*d]
    }

param_grid = param_grid_orig.copy()
for c in conditions:
    for key in param_grid_orig.keys():
        param_grid[key].append(param_grid[key][0]*c[key] if key in c else param_grid[key][0])

# define simulation parameters
dt = 1e-2
T = 3000.0
dts = 1e-1

# perform simulation
results, result_map = grid_search(circuit_template="config/stn_gpe/stn_gpe_basic",
                                  param_grid=param_grid,
                                  param_map=param_map,
                                  simulation_time=T,
                                  step_size=dt,
                                  sampling_step_size=dts,
                                  permute_grid=False,
                                  inputs={},
                                  outputs={'R_e': 'stn/stn_op/R_e',
                                           'R_i': 'gpe/gpe_proto_op/R_i'},
                                  init_kwargs={'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
                                  )
results = results * 1e3

for i, id in enumerate(result_map.index):
    r_e = results['R_e'][id]
    r_i = results['R_i'][id]
    plt.plot(r_e)
    plt.plot(r_i)
    #plt.plot(np.ones_like(r_e)*target[i][1], '--')
    plt.legend(['R_e', 'R_i'])
    plt.show()
