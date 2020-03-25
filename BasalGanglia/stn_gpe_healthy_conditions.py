from pyrates.utility import grid_search
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import scipy.io as scio
from pandas import read_hdf

conditions = [{'k_ie': 0.2},  # AMPA blockade in GPe
              {'k_ie': 0.2, 'k_ii': 0.2, 'k_str': 0.2},  # AMPA blockade and GABAA blockade in GPe
              {'k_ii': 0.2, 'k_str': 0.2},  # GABAA blockade in GPe
              {'k_ie': 0.0},  # STN blockade
              {'k_ie': 0.0},  # STN blockade and AMPA+NMDA blockade in GPe
              {'k_ei': 0.2}  # GABAA blocker in STN
              ]

target = [[20, 60],   # healthy control
          [np.nan, 40],  # ampa blockade in GPe
          [np.nan, 70],  # ampa and gabaa blockade in GPe
          [np.nan, 100],  # GABAA blockade in GPe
          [np.nan, 30],  # STN blockade
          [np.nan, 15],  # STN blockade and AMPA+NMDA blockade in GPe
          [np.nan, 120]  # GABAA blockade in STN
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

path = "/data/u_rgast_software/PycharmProjects/BrainNetworks/BasalGanglia/stn_gpe_healthy_opt/PopulationDrops/pop_summary_64.h5"
param_grid_orig = read_hdf(path).to_dict()

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
